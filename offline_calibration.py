"""CPU-only tier research protocols; no training, serving, weights, or test IO.

Nested LOGO assesses the *selection procedure*, not a single final deployment
configuration. All gates are explicit prototype protocol numbers, not statistical
risk guarantees. The existing A/B/C training calibration remains unchanged.
"""
from __future__ import annotations
import collections
from fractions import Fraction
import math
import random
from calibration_groups import PlanError, fit_temperature
from evaluation_metrics import per_class, calibration_bins
from raw_logit_contract import (TIERS, align_sources, digest, exact, number,
                                source_binding, validate_costs, validate_prior, validate_source)

WISE_KEYS = ('schema_version', 'alpha_grid', 'temperature_bounds', 'fit_iterations', 'fit_tolerance',
             'min_fit_groups', 'min_class_rows', 'min_class_groups', 'min_predicted_groups',
             'min_flagship_recall', 'max_flagship_fpr', 'min_fast_precision')
TAU_GRID = [0., .5, 1., 1.5, 2.]
TAU_KEYS = ('schema_version', 'tau_grid', 'fixed_temperature', 'comparison_tau', 'min_class_rows',
            'min_class_groups', 'min_uncertainty_groups', 'bootstrap_samples', 'bootstrap_seed',
            'min_flagship_recall', 'max_flagship_fpr')


def _integer(obj, key, minimum, maximum=100000):
    if type(obj[key]) is not int or not minimum <= obj[key] <= maximum:
        raise PlanError(f'protocol {key} must be an integer in [{minimum},{maximum}]')


def _rate(obj, key, positive=False):
    if not number(obj[key]) or not (0 < obj[key] <= 1 if positive else 0 <= obj[key] <= 1):
        raise PlanError(f'protocol {key} must be a finite probability')


def validate_wise_protocol(protocol):
    exact(protocol, WISE_KEYS, 'nested LOGO protocol')
    if protocol['schema_version'] != 'router-nested-logo/v1':
        raise PlanError('unsupported nested LOGO protocol version')
    grid = protocol['alpha_grid']
    if not isinstance(grid, list) or not 1 <= len(grid) <= 21 or not all(number(a) and 0 <= a <= 1 for a in grid) \
            or len(set(grid)) != len(grid):
        raise PlanError('alpha_grid requires 1..21 distinct finite numbers in [0,1]')
    bounds = protocol['temperature_bounds']
    if not isinstance(bounds, list) or len(bounds) != 2 or not all(number(v) for v in bounds) \
            or not .05 <= bounds[0] < bounds[1] <= 100:
        raise PlanError('temperature bounds require 0.05 <= low < high <= 100')
    _integer(protocol, 'fit_iterations', 1, 1000)
    if not number(protocol['fit_tolerance']) or not 0 < protocol['fit_tolerance'] <= .1:
        raise PlanError('fit_tolerance must be finite in (0,.1]')
    for key in ('min_fit_groups', 'min_class_rows', 'min_class_groups', 'min_predicted_groups'):
        _integer(protocol, key, 2)
    _rate(protocol, 'min_flagship_recall', positive=True)
    _rate(protocol, 'max_flagship_fpr')
    _rate(protocol, 'min_fast_precision', positive=True)


def validate_tau_protocol(protocol):
    exact(protocol, TAU_KEYS, 'tau ablation protocol')
    if protocol['schema_version'] != 'router-tau-ablation/v1' or protocol['tau_grid'] != TAU_GRID \
            or not isinstance(protocol['tau_grid'], list) or not all(number(t) for t in protocol['tau_grid']):
        raise PlanError('tau grid must be the registered [0,0.5,1,1.5,2]')
    if not number(protocol['fixed_temperature']) or not .05 <= protocol['fixed_temperature'] <= 100:
        raise PlanError('fixed_temperature must be finite in [.05,100]; no temperature is fitted in tau-only ablation')
    if not number(protocol['comparison_tau']) or protocol['comparison_tau'] not in TAU_GRID[:-1]:
        raise PlanError('one smaller comparison_tau must be registered before inspecting the sweep')
    for key in ('min_class_rows', 'min_class_groups', 'min_uncertainty_groups'):
        _integer(protocol, key, 2)
    _integer(protocol, 'bootstrap_samples', 100, 10000)
    _integer(protocol, 'bootstrap_seed', 0, 2**32-1)
    _rate(protocol, 'min_flagship_recall', positive=True)
    _rate(protocol, 'max_flagship_fpr')


def support(rows):
    return {c: {'rows': sum(r['label'] == c for r in rows),
                'groups': len({r['split_group'] for r in rows if r['label'] == c})} for c in TIERS}


def supported(rows, protocol, fitting=False):
    table = support(rows)
    return (len({r['split_group'] for r in rows}) >= (protocol['min_fit_groups'] if fitting else 2)
            and all(v['rows'] >= protocol['min_class_rows'] and v['groups'] >= protocol['min_class_groups']
                    for v in table.values()))


def logo(rows):
    """Membership depends on group identities only, never labels or class balance."""
    for group in sorted({r['split_group'] for r in rows}):
        yield ([r for r in rows if r['split_group'] != group],
               [r for r in rows if r['split_group'] == group])


def probabilities(logits, temperature):
    scaled = [v / temperature for v in logits]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]


def _weights(rows):
    counts = collections.Counter(r['split_group'] for r in rows)
    return [1 / (len(counts) * counts[r['split_group']]) for r in rows]


def _group_units(rows):
    """Integer row masses with one common denominator for group-equal error.

    Float accumulation can make equal errors differ and defeat canonical ties
    (red team F1). Integer updates preserve equality AND arbitrarily close strict
    ordering, without an epsilon that would merge genuinely unequal objectives.
    """
    counts = collections.Counter(r['split_group'] for r in rows)
    scale = math.lcm(*counts.values())
    return [scale // counts[r['split_group']] for r in rows], len(counts) * scale


def _rates(matrix):
    flag_tp = matrix[2][2]
    flag_true = sum(matrix[2])
    flag_pred = sum(row[2] for row in matrix)
    fast_pred = sum(row[0] for row in matrix)
    nonflag = sum(sum(row) for row in matrix[:2])
    return {'flagship_recall': flag_tp / flag_true if flag_true else None,
            'flagship_precision': flag_tp / flag_pred if flag_pred else None,
            'flagship_fpr': (flag_pred - flag_tp) / nonflag if nonflag else None,
            'fast_precision': matrix[0][0] / fast_pred if fast_pred else None}


def _constraints(rates, protocol, fast=False):
    # Roundoff is not evidence: tolerate only machine-scale accumulated error.
    return (rates['flagship_recall'] is not None and rates['flagship_fpr'] is not None
            and rates['flagship_recall'] + 1e-12 >= protocol['min_flagship_recall']
            and rates['flagship_fpr'] <= protocol['max_flagship_fpr'] + 1e-12
            and (not fast or (rates['fast_precision'] is not None
                             and rates['fast_precision'] + 1e-12 >= protocol['min_fast_precision'])))


def choose_thresholds(rows, probs, protocol):
    """Exact empirical breakpoint search, not a coarse probability grid.

    Rule: flagship if p_flag >= theta_flag, otherwise fast if p_fast >=
    theta_fast, otherwise balanced. Flagship wins overlap/ties. All distinct
    prediction partitions on these finite rows are searched; no population
    optimum claim. Sentinel nextafter(1,+inf) explicitly allows 'select none'.
    O(n^2) updates, rather than recomputing n predictions for each pair.
    """
    if not supported(rows, protocol, fitting=True):
        return None
    weights = _weights(rows)
    units, total_units = _group_units(rows)
    truth = [TIERS.index(r['label']) for r in rows]
    best = None
    sentinel = math.nextafter(1., math.inf)
    flag_candidates = sorted({0., sentinel, *(p[2] for p in probs)})
    for theta_flag in flag_candidates:
        matrix = [[0.] * 3 for _ in TIERS]
        correct_units = 0
        flag_groups = set()
        remaining = []
        for i, (row, p) in enumerate(zip(rows, probs)):
            flag = p[2] >= theta_flag
            matrix[truth[i]][2 if flag else 1] += weights[i]
            if truth[i] == (2 if flag else 1):
                correct_units += units[i]
            if flag:
                flag_groups.add(row['split_group'])
            else:
                remaining.append(i)
        if len(flag_groups) < protocol['min_predicted_groups']:
            continue
        fast_groups = set()
        by_fast = collections.defaultdict(list)
        for i in remaining:
            by_fast[probs[i][0]].append(i)
        # Each unique p_fast is a canonical representative of one partition.
        for theta_fast in sorted(by_fast, reverse=True):
            for i in by_fast[theta_fast]:
                matrix[truth[i]][1] -= weights[i]
                matrix[truth[i]][0] += weights[i]
                correct_units += units[i] * ((truth[i] == 0) - (truth[i] == 1))
                fast_groups.add(rows[i]['split_group'])
            if len(fast_groups) < protocol['min_predicted_groups']:
                continue
            rates = _rates(matrix)
            if not _constraints(rates, protocol, fast=True):
                continue
            # Every candidate has denominator total_units: compare numerators,
            # not rounded errors. Float rates/gates above remain unchanged.
            key = (total_units - correct_units, theta_flag, theta_fast)
            if best is None or key < best[0]:
                best = (key, {'theta_flag': theta_flag, 'theta_fast': theta_fast})
    return None if best is None else best[1]


def predict(rows, temperature, thresholds=None):
    output = []
    for row in rows:
        p = probabilities(row['logits'], temperature)
        choice = (2 if p[2] >= thresholds['theta_flag'] else
                  0 if p[0] >= thresholds['theta_fast'] else 1) if thresholds else max(range(3), key=p.__getitem__)
        # Private identifiers exist in memory only; reports aggregate below.
        output.append({**row, 'probabilities': p, 'prediction': TIERS[choice]})
    return output


def _cost_means(rows, costs):
    """Exact selected-route spend, rounded once per mean to binary64.

    Scaling by an unused huge route erased representable positive spend (red
    team F1). Even selected-cost scaling/float weights can round subnormal terms
    to zero. Count routes with integers and retain rational group means until
    the final conversion: floats mean their exact binary values, ints are exact.
    The bounded CPU protocol has only three routes; no per-row Fraction is needed.
    """
    groups = collections.defaultdict(collections.Counter)
    for row in rows:
        groups[row['split_group']][row['prediction']] += 1
    selected = {tier for counts in groups.values() for tier in counts}
    if not all(number(costs[c]) and costs[c] > 0 for c in selected):
        raise PlanError('relative route costs must be finite positive nonboolean numbers')
    exact_costs = {c: Fraction(costs[c]) for c in selected}
    totals = [(sum(exact_costs[c]*n for c, n in counts.items()), sum(counts.values()))
              for counts in groups.values()]
    means = (sum(total for total, _ in totals) / len(rows),
             sum(total / n for total, n in totals) / len(groups))
    try:
        result = tuple(float(mean) for mean in means)
    except OverflowError:
        raise PlanError('relative cost means are not finite representable numbers') from None
    # Valid finite positive costs form convex means, including subnormals.
    # Do not clamp, add epsilon or silently certify zero on an arithmetic failure.
    if not all(math.isfinite(mean) and mean > 0 for mean in result):
        raise PlanError('relative cost means are not positive finite representable numbers')
    return result


def summarize(rows, costs=None):
    """Tier-only adapter reusing shared v2 class and calibration-bin metrics.

    Row-weighted confusion/F1/NLL/Brier and decision-confidence ECE; additionally
    group-equal NLL/error/rates for protocol selection. ECE uses the probability
    of the *threshold-selected* class, not an unrelated argmax confidence.
    Shared metrics import schema constants only (model loading is lazy). Avoid
    calling full metrics() here: it would nest 1000 resamples inside each tau
    bootstrap statistic, and ordinal routing loss is not predicted-route spend.
    """
    if not rows:
        raise PlanError('cannot summarize an empty prediction pool')
    matrix = [[0] * 3 for _ in TIERS]
    weighted = [[0.] * 3 for _ in TIERS]
    nll = brier = gnll = 0.
    row_cost, group_cost = _cost_means(rows, costs) if costs else (None, None)
    metric_rows = [{'label': {'tier': r['label']},
                    'answers': {'tier': {'choice': r['prediction'],
                                        'confidence': r['probabilities'][TIERS.index(r['prediction'])]}}}
                   for r in rows]
    pc = per_class(metric_rows, 'tier')
    bins = calibration_bins(metric_rows, 'tier')
    for row, weight in zip(rows, _weights(rows)):
        true, pred = TIERS.index(row['label']), TIERS.index(row['prediction'])
        p = row['probabilities']
        matrix[true][pred] += 1
        weighted[true][pred] += weight
        loss = -math.log(max(1e-12, p[true]))
        nll += loss
        gnll += weight * loss
        brier += sum((v - (i == true))**2 for i, v in enumerate(p))
    n = len(rows)
    return {'metric_definition': 'tier-v2; fixed 3-class macro-F1; NLL floor 1e-12; Brier sum; decision-confidence ECE',
            'metric_definition_version': 2, 'nll_probability_floor': 1e-12,
            'rows': n, 'groups': len({r['split_group'] for r in rows}), 'support': support(rows),
            'confusion': {a: dict(zip(TIERS, matrix[i])) for i, a in enumerate(TIERS)},
            'per_class': {c: pc[c] for c in TIERS},
            'accuracy': sum(matrix[i][i] for i in range(3)) / n, 'macro_f1': pc['macro_f1'],
            'nll': nll / n, 'brier': brier / n,
            'ece_5_bins': sum(b['contribution'] for b in bins),
            'ece_bin_counts': [b['count'] for b in bins],
            'mean_relative_cost': row_cost,
            'group_equal': {'nll': gnll, 'error': max(0., 1 - sum(weighted[i][i] for i in range(3))),
                            'mean_relative_cost': group_cost, **_rates(weighted)}}


def _fit(rows, protocol):
    if not supported(rows, protocol, fitting=True):
        return None
    temperature = fit_temperature([r['logits'] for r in rows], [TIERS.index(r['label']) for r in rows],
                                  [r['split_group'] for r in rows], bounds=protocol['temperature_bounds'],
                                  max_iterations=protocol['fit_iterations'], tolerance=protocol['fit_tolerance'])
    probs = [probabilities(r['logits'], temperature) for r in rows]
    thresholds = choose_thresholds(rows, probs, protocol)
    return None if thresholds is None else {'temperature': temperature, **thresholds}


def _ensemble(left, right, alpha):
    return [{**a, 'logits': [(1-alpha)*x + alpha*y for x, y in zip(a['logits'], b['logits'])]}
            for a, b in zip(left, right)]


def select_outer_training(left, right, protocol):
    """Only outer-training rows are accepted; inner validation labels select alpha.

    Inner validation groups never fit T or either threshold. After choosing
    alpha, refit T/thresholds on outer training only. These fits are NOT scores.
    """
    candidates, errors = [], {}
    for alpha in sorted(protocol['alpha_grid']):
        rows = _ensemble(left, right, alpha)
        oof = []
        folds = []
        for train, valid in logo(rows):
            fit = _fit(train, protocol)
            if fit is None:
                break
            oof.extend(predict(valid, fit['temperature'], fit))
            folds.append({'fit_groups': len({r['split_group'] for r in train}),
                          'fit_rows': len(train), 'validation_groups': 1, 'validation_rows': len(valid)})
        complete = len(oof) == len(rows)
        summary = summarize(oof) if complete else None
        if complete:
            units, total_units = _group_units(oof)
            errors[alpha] = Fraction(sum(u for r, u in zip(oof, units) if r['label'] != r['prediction']), total_units)
        feasible = complete and supported(oof, protocol) and _constraints(summary['group_equal'], protocol, fast=True)
        # Require >=2 groups supporting both predicted decisions in inner OOF.
        feasible = feasible and all(len({r['split_group'] for r in oof if r['prediction'] == c})
                                    >= protocol['min_predicted_groups'] for c in ('fast', 'flagship'))
        candidates.append({'alpha': alpha, 'feasible': bool(feasible), 'inner_oof': summary,
                           'inner_folds': folds})
    feasible = [c for c in candidates if c['feasible']]
    if not feasible:
        return None, candidates
    chosen = min(feasible, key=lambda c: (errors[c['alpha']],
                                         c['inner_oof']['group_equal']['nll'], c['alpha']))
    fit = _fit(_ensemble(left, right, chosen['alpha']), protocol)
    return (None if fit is None else {'alpha': chosen['alpha'], **fit}), candidates


def nested_logo(base, finetuned, protocol):
    validate_wise_protocol(protocol)
    left, right = align_sources(base, finetuned)
    binding = {'base': source_binding(base), 'finetuned': source_binding(finetuned), 'protocol_sha256': digest(protocol)}
    report = {'schema_version': 'router-nested-logo-report/v1', 'status': 'insufficient', 'binding': binding,
              'protocol': protocol, 'support': support(left), 'outer_folds': [], 'outer_oof': None,
              'deployment_configuration': None, 'test_used': False,
              'method': 'logit ensemble, NOT WiSE-FT weight interpolation; tier only',
              'claim': 'synthetic mechanics only' if base['data_kind'] == 'synthetic' else
                       'nested development estimate; not independent generalization or production qualification'}
    # Preflight EVERY inner fit before any selection. Do not keep only convenient
    # outer groups or silently relax support when a class is concentrated.
    if not supported(left, protocol):
        report['reason'] = 'development pool misses registered class/group support'
        return report
    for outer_train, _ in logo(left):
        if not supported(outer_train, protocol, fitting=True) or any(not supported(train, protocol, fitting=True)
                                                                  for train, _ in logo(outer_train)):
            report['reason'] = 'outer/inner fitting union misses registered class/group support; no fallback'
            return report
    right_by_id = {r['id']: r for r in right}
    oof = []
    for train, valid in logo(left):
        fit, candidates = select_outer_training(train, [right_by_id[r['id']] for r in train], protocol)
        fold = {'fold': len(report['outer_folds']), 'fit_rows': len(train), 'holdout_rows': len(valid),
                'fit_groups': len({r['split_group'] for r in train}), 'holdout_groups': 1,
                'selection': fit, 'candidates': candidates}
        report['outer_folds'].append(fold)
        if fit is None:
            report['reason'] = 'no feasible alpha/T/dual thresholds in an outer fold; no partial OOF claim or fallback'
            return report
        ens = _ensemble(valid, [right_by_id[r['id']] for r in valid], fit['alpha'])
        predicted = predict(ens, fit['temperature'], fit)
        oof.extend(predicted)
        fold['holdout_metrics'] = summarize(predicted)
    report.update(status='available', outer_oof=summarize(oof),
                  reason='every row predicted once without its group labels selecting alpha/T/thresholds; no full-pool fitted score')
    return report


def paired_group_interval(left, right, protocol, statistic):
    """Paired whole-group percentile diagnostic on FIXED development predictions.

    Repeated draws of one cluster get distinct bootstrap identities so the
    group-equal statistic counts multiplicity correctly. Does not refit or
    account for adaptive dev selection; one comparison is registered in advance.
    """
    groups = sorted({r['split_group'] for r in left})
    out = {'status': 'unavailable', 'estimate': None, 'low': None, 'high': None, 'groups': len(groups),
           'requested_resamples': protocol['bootstrap_samples'], 'valid_resamples': 0,
           'method': 'paired whole-group percentile bootstrap, 95%, fixed predictions',
           'warning': 'Approximate exploratory interval; few groups/boundary scores can understate uncertainty. No significance guarantee.'}
    if len(groups) < protocol['min_uncertainty_groups'] or not supported(left, protocol):
        out['reason'] = 'registered group/class support not met'
        return out
    lb = {g: [r for r in left if r['split_group'] == g] for g in groups}
    rb = {g: [r for r in right if r['split_group'] == g] for g in groups}
    rng = random.Random(protocol['bootstrap_seed'])
    values = []
    for _ in range(protocol['bootstrap_samples']):
        draw = [rng.choice(groups) for _ in groups]
        a = [{**r, 'split_group': str(i)} for i, g in enumerate(draw) for r in lb[g]]
        b = [{**r, 'split_group': str(i)} for i, g in enumerate(draw) for r in rb[g]]
        values.append(statistic(a) - statistic(b))
    values.sort()
    out.update(status='available', estimate=statistic(left)-statistic(right),
               low=values[int(.025*len(values))], high=values[min(len(values)-1, int(.975*len(values)))],
               valid_resamples=len(values))
    return out


def tau_ablation(source, prior, costs, protocol):
    validate_tau_protocol(protocol)
    rows = validate_source(source)
    prior_values, cost_table = validate_prior(prior, source), validate_costs(costs)
    candidates, predictions = [], {}
    for tau in TAU_GRID:
        adjusted = [{**r, 'logits': [z - tau*math.log(p) for z, p in zip(r['logits'], prior_values)]} for r in rows]
        predicted = predict(adjusted, protocol['fixed_temperature'])
        predictions[tau] = predicted
        metrics = summarize(predicted, cost_table)
        feasible = supported(rows, protocol) and _constraints(metrics['group_equal'], protocol)
        candidates.append({'tau': tau, 'metrics': metrics, 'selection_feasible': feasible})
    eligible = [c for c in candidates if c['selection_feasible']]
    selected = min(eligible, key=lambda c: (c['metrics']['group_equal']['nll'], c['tau']))['tau'] if eligible else None
    reference = protocol['comparison_tau']
    def stat(rs, key):
        return summarize(rs, cost_table)['group_equal'][key]
    intervals = {key: paired_group_interval(predictions[2.], predictions[reference], protocol,
                                           lambda rs, key=key: stat(rs, key))
                 for key in ('nll', 'error', 'mean_relative_cost')}
    # An interval computed after sweeping cannot certify a winner or resurrect
    # pristine holdout status. Synthetic inputs can NEVER support real claims.
    nll = intervals['nll']
    conclusion = ('synthetic_only_no_model_conclusion' if source['data_kind'] == 'synthetic' else
                  'uncertain' if nll['status'] != 'available' or nll['low'] <= 0 else
                  'exploratory_nll_evidence_tau_2_worse_than_registered_reference_not_a_significance_claim')
    return {'schema_version': 'router-tau-ablation-report/v1', 'status': 'available' if selected is not None else 'insufficient',
            'binding': {'input': source_binding(source), 'prior_sha256': digest(prior), 'cost_sha256': digest(costs),
                        'protocol_sha256': digest(protocol)}, 'protocol': protocol, 'cost_table': costs,
            'prior': prior, 'candidates': candidates, 'selected_development_tau': selected,
            'selection_rule': 'minimum group-equal development NLL subject to registered flagship recall/FPR; ties smaller tau',
            'comparison': {'tau': 2., 'registered_reference_tau': reference, 'delta': 'tau=2 minus reference',
                           'intervals': intervals, 'conclusion': conclusion},
            'test_used': False, 'training_causal_conclusion': 'unavailable: fixed weights cannot remove training oversampling; needs matched #19 training ablation',
            'claim': 'tau-only descriptive development sweep, fixed predeclared T; simulated tier cost is not billing or task success; no test tuning'}
