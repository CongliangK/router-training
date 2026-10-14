"""CPU-only tests for the issue #14 training-reproducibility fixes.

All fixtures are invented synthetic distributions (arbitrary tier counts,
opaque ids); nothing here loads torch, a model, real data, or a GPU. The real
train_eval.py is imported through the torch-stubbing loader defined in
test_calibration_groups (so its QUESTIONS constants stay source-parsed), and
every expected number is hand-computable.
"""
import argparse
import ast
import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V3C = json.loads((ROOT / 'configs' / 'v3c.json').read_text())

_FIXTURES = None


def fixtures():
    """test_calibration_groups' synthetic builders, loaded by file path so
    both `unittest discover -s tests` and `python -m unittest tests.test_...`
    work (the same pattern its own load_server_contract uses)."""
    global _FIXTURES
    if _FIXTURES is None:
        spec = importlib.util.spec_from_file_location(
            'test_calibration_groups_fixtures', ROOT / 'tests' / 'test_calibration_groups.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _FIXTURES = module
    return _FIXTURES


_STUBS = None


def train_eval():
    """The REAL train_eval.py with torch/peft/model_utils stubbed (the same
    loader the adapter tests use); the issue #14 helpers are pure Python and
    run unchanged under it."""
    global _STUBS
    if _STUBS is None:
        _STUBS = fixtures().import_train_eval_with_stubs()
    return _STUBS


def command_from(stdout):
    """Parse the launcher's printed `Command: [...]` list literal."""
    line = next(l for l in stdout.splitlines() if l.startswith('Command:'))
    return ast.literal_eval(line[len('Command:'):].strip())


class SchedulerStepCountTests(unittest.TestCase):
    """Defect 1: total steps must be counted over the OVERSAMPLED row stream
    (tail batch by ceil), not the raw pre-oversampling length."""

    def test_oversampled_epoch_rows_equalize_tier_counts_at_the_max_quota(self):
        # 16/7/5 rows per tier (the v3c fast-majority shape in miniature):
        # quota = 16, so every epoch trains 3x16 = 48 rows.
        rows = [{'id': f'fixture-{i}', 'label': {'tier': tier}}
                for i, tier in enumerate(['fast'] * 16 + ['balanced'] * 7 + ['flagship'] * 5)]
        random.seed(3)
        epoch_rows = train_eval().oversample_epoch_rows(rows)
        counts = Counter(r['label']['tier'] for r in epoch_rows)
        self.assertEqual(dict(counts), {'fast': 16, 'balanced': 16, 'flagship': 16})
        self.assertEqual(len(epoch_rows), 48)
        self.assertEqual({r['id'] for r in epoch_rows}, {r['id'] for r in rows})  # no row lost
        self.assertGreaterEqual(len(epoch_rows), len(rows))                       # minority repeated up

    def test_optimizer_updates_match_the_loop_step_rule_including_tail_ceil(self):
        # The loop steps when (i+1) % accumulation == 0 or i+1 == len(rows):
        # that is exactly ceil(n/accumulation), short tail included.
        for n, accumulation in [(0, 4), (1, 4), (7, 8), (24, 8), (25, 8), (30, 4), (1542, 8)]:
            loop_steps = sum(1 for i in range(1, n + 1)
                             if i % accumulation == 0 or i == n)
            self.assertEqual(train_eval().optimizer_updates(n, accumulation),
                             loop_steps, (n, accumulation))
        self.assertEqual(train_eval().optimizer_updates(1542, 8), 193)  # archived v3c per-epoch count
        self.assertEqual(train_eval().optimizer_updates(30, 4), 8)     # 7 full batches + a 2-row tail

    def test_total_updates_use_oversampled_rows_not_the_raw_length(self):
        # 16/7/5 rows -> 48 oversampled rows/epoch; accumulation 4, 2 epochs:
        # 12 updates/epoch, 24 total. The archived defect computed the plan
        # from the raw length (2*(28//4) = 14) and mis-scheduled the cosine.
        te = train_eval()
        epoch_counts = [48, 48]
        total = sum(te.optimizer_updates(n, 4) for n in epoch_counts)
        self.assertEqual(total, 24)
        self.assertNotEqual(total, 2 * (28 // 4))


class SchedulerCurveTests(unittest.TestCase):
    """Defect 1 (curve): clamped cosine + 10%-of-initial LR floor."""

    def test_warmup_then_clamped_cosine_is_hand_computed(self):
        multiplier = train_eval().cosine_lr_multiplier
        self.assertEqual(multiplier(0, 10, 2), 0.0)
        self.assertEqual(multiplier(1, 10, 2), 0.5)
        self.assertEqual(multiplier(2, 10, 2), 1.0)  # peak at the end of warmup
        # step 5: progress (5-2)/8 = 3/8 -> (1+cos(3pi/8))/2
        self.assertAlmostEqual(multiplier(5, 10, 2), (1 + math.cos(math.pi * 3 / 8)) / 2, places=12)

    def test_lr_never_rises_past_total_steps_and_floor_is_ten_percent(self):
        te = train_eval()
        values = [te.cosine_lr_multiplier(s, 10, 2) for s in range(2, 26)]
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(values, values[1:])),
                        'the multiplier must be non-increasing after warmup')
        self.assertEqual(te.cosine_lr_multiplier(10, 10, 2), 0.1)   # at the planned end: floor
        self.assertEqual(te.cosine_lr_multiplier(25, 10, 2), 0.1)   # past the end: clamped, no rise
        # The old unclamped curve would have returned (1+cos(pi*15/8))/2 ~= 0.96.
        self.assertLess(te.cosine_lr_multiplier(25, 10, 2), (1 + math.cos(math.pi * 15 / 8)) / 2)
        self.assertEqual(te.LR_FLOOR_FRACTION, 0.1)
        self.assertEqual(te.FULLFT_INITIAL_LR, 2e-6)
        self.assertEqual(te.FULLFT_INITIAL_LR * te.LR_FLOOR_FRACTION, 2e-7)  # official-recipe floor

    def test_reported_warmup_fraction_is_the_achieved_ratio_not_the_target(self):
        # Red-team LOW-2: .03 is only the warmup TARGET; after int() truncation
        # the achieved fraction differs, and the report must carry the real
        # warmup/total ratio. Archived v3c shape: 6 epochs * 193 updates = 1158
        # total, warmup = int(1158*.03) = 34, so 34/1158 ≈ 0.0294, NOT .03.
        te = train_eval()
        total, warmup, report = te.fullft_scheduler_plan(1158, 2e-6)
        self.assertEqual((total, warmup), (1158, 34))
        self.assertEqual(report['warmup_fraction_of_total'], 34 / 1158)
        self.assertAlmostEqual(report['warmup_fraction_of_total'], 0.02936, places=4)
        self.assertNotEqual(report['warmup_fraction_of_total'], .03)  # never the hardcoded target
        self.assertEqual(report['lr_floor'], 2e-7)
        # Tiny-run floor: warmup never drops below 1 step, and the fraction
        # still reports the honest achieved ratio (1 of 3), not .03.
        _, warmup1, report1 = te.fullft_scheduler_plan(3, 2e-6)
        self.assertEqual(warmup1, 1)
        self.assertEqual(report1['warmup_fraction_of_total'], 1 / 3)


class EffectiveConfigTests(unittest.TestCase):
    """Defect 3: the report must record the LR that actually ran (fullft
    fixes 2e-6) separately from the requested --lr."""

    def _args(self, **overrides):
        values = dict(epochs=2, rank=8, lr=1e-4, accumulation=4, seed=7, beta=.999,
                      label_smoothing=.05, synth=True, mlp=False, fullft=True, swa=True,
                      legacy_calibration=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_fullft_effective_lr_is_2e6_and_requested_lr_is_preserved(self):
        te = train_eval()
        self.assertEqual(te.effective_initial_lr(self._args(fullft=True)), 2e-6)
        self.assertEqual(te.effective_initial_lr(self._args(fullft=False, lr=1e-4)), 1e-4)
        config = te.effective_config(self._args(), raw_train_rows=28, epoch_row_counts=[48, 48],
                                     scheduler_report={'type': 'cosine_with_warmup'},
                                     swa_pool=[(1, 'fixture-a'), (2, 'fixture-b')])
        self.assertEqual(config['lr_effective'], 2e-6)   # what the optimizer used
        self.assertEqual(config['lr_requested'], 1e-4)   # raw CLI input, kept separate
        self.assertIn('fullft', config['lr_note'])
        self.assertEqual(config['epoch_optimizer_updates'], [12, 12])
        self.assertEqual(config['train_rows'],
                         {'pre_oversampling': 28, 'oversampled_per_epoch': [48, 48]})
        self.assertEqual(config['swa_pool_final_epochs'], [1, 2])
        self.assertIn('ceil', config['accumulation_tail_batch'])
        self.assertEqual(config['recipe']['epochs'], 2)
        json.dumps(config)  # fully serializable into report.json

    def test_legacy_runs_declare_the_scheduler_divergence_vs_archived_v3c(self):
        config = train_eval().effective_config(
            self._args(legacy_calibration=True, accumulation=8),
            raw_train_rows=669, epoch_row_counts=[1542] * 6,
            scheduler_report={'type': 'cosine_with_warmup'}, swa_pool=[(2, 'b'), (4, 'd'), (5, 'e')])
        note = config['archived_v3c_divergence_note']
        self.assertIn('OLD scheduler', note)
        self.assertIn('not comparable', note)
        self.assertEqual(config['epoch_optimizer_updates'], [193] * 6)  # 6*193 = 1158
        # Non-legacy runs carry no legacy note.
        self.assertNotIn('archived_v3c_divergence_note',
                         train_eval().effective_config(self._args(), 28, [48, 48], {}, []))


class SwaPoolTests(unittest.TestCase):
    """Defect 4: strict top-3 pool with eviction; soup = final top-3 average."""

    @staticmethod
    def _run_epochs(score_sequence):
        te = train_eval()
        history, pool, taken = [], [], []

        def snapshot(epoch):  # cheap stand-in for the CPU fp32 state copy
            taken.append(epoch)
            return {'w': float(epoch)}

        for epoch, f1 in score_sequence:
            history.append({'epoch': epoch, 'calibration_macro_f1': f1})
            pool = te.update_swa_pool(pool, history, snapshot)
        return te, history, pool, taken

    def test_fourth_epoch_evicts_the_first_from_the_pool(self):
        te, history, pool, taken = self._run_epochs([(1, .50), (2, .70), (3, .60), (4, .90)])
        self.assertEqual([epoch for epoch, _ in pool], [2, 3, 4])   # epoch 1 evicted
        self.assertEqual(sorted(taken), [1, 2, 3, 4])               # snapshots only on admission
        self.assertTrue(all(snap['w'] != 1.0 for _, snap in pool))  # its CPU reference was released
        self.assertEqual(set(te.top_swa_epochs(history)), {2, 3, 4})
        # Soup membership is consistent: the pool IS the final top-3, so the
        # soup average equals the average of exactly the top-3 snapshots.
        self.assertEqual(te.average_state_dicts([snap for _, snap in pool]),
                         {'w': 3.0})  # (2+3+4)/3, hand-computed
        self.assertEqual(te.average_state_dicts([{'w': float(e)} for e in te.top_swa_epochs(history)]),
                         {'w': 3.0})

    def test_ties_keep_earlier_epochs_and_a_fifth_epoch_evicts_the_third(self):
        _, _, pool, taken = self._run_epochs([(1, .50), (2, .70), (3, .60), (4, .90), (5, .90)])
        self.assertEqual([epoch for epoch, _ in pool], [2, 4, 5])   # 5 ties 4 (earlier wins), 3 evicted
        self.assertEqual(sorted(taken), [1, 2, 3, 4, 5])
        self.assertEqual(len(pool), 3)                              # never over-filled

    def test_soup_state_is_mathematically_the_top3_average(self):
        te = train_eval()

        class Fake:
            """Tensor stand-in: supports the arithmetic average_state_dicts
            performs (0 + v/n accumulation) and the .to()/.dtype/.device
            access soup_state_dict applies."""

            dtype = 'fixture-f32'
            device = 'fixture-cpu'

            def __init__(self, value):
                self.value = float(value)

            def __float__(self):
                return self.value

            def __add__(self, other):
                return Fake(self.value + float(other))

            def __radd__(self, other):
                return Fake(float(other) + self.value)

            def __truediv__(self, other):
                return Fake(self.value / other)

            def to(self, *args, **kwargs):
                return self

        snaps = [{'w': Fake(3.)}, {'w': Fake(6.)}, {'w': Fake(9.)}]  # pooled fp32 stand-ins
        avg = te.average_state_dicts(snaps)
        self.assertEqual(avg['w'].value, 6.0)                   # exact hand average (1+2+3)
        current = {'w': Fake(99.), 'bias': Fake(1.)}
        soup = te.soup_state_dict(current, avg)
        self.assertEqual(soup['w'].value, 6.0)                  # the average is what gets loaded
        self.assertIs(soup['bias'], current['bias'])            # non-pooled keys keep the live state


class LauncherRecipeTests(unittest.TestCase):
    """Defect 2: the grouped launcher passes the COMPLETE v3c recipe and
    deviations are declared only through an explicit --recipe override."""

    def setUp(self):
        source = fixtures()
        directory = tempfile.TemporaryDirectory(prefix='router-recipe-')
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name)
        source.write_jsonl(self.data_dir / 'train.jsonl', source.make_train_rows(n=30))
        source.write_jsonl(self.data_dir / 'calibration.jsonl', source.make_pool(n_groups=16))
        (self.data_dir / 'test.jsonl').write_text('NOT-VALID-JSON-SENTINEL\n')
        self.roles_path = self.data_dir / 'roles.json'
        self.roles_path.write_text(json.dumps(source.make_roles(source.make_pool(n_groups=16))))

    def _launch(self, *extra):
        return subprocess.run([sys.executable, str(ROOT / 'scripts' / 'train.py'),
                               '--run', 'fixture-recipe-echo', '--roles', str(self.roles_path),
                               '--data-dir', str(self.data_dir), '--dry-run', *extra],
                              capture_output=True, text=True)

    def test_grouped_dry_run_command_carries_the_full_v3c_recipe(self):
        result = self._launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Structural preflight', result.stdout)
        command = command_from(result.stdout)
        self.assertIn('--calibration-roles', command)
        for key in ['epochs', 'rank', 'lr', 'accumulation', 'seed', 'beta', 'label_smoothing']:
            flag = '--' + key.replace('_', '-')
            self.assertIn(flag, command)
            self.assertEqual(command[command.index(flag) + 1], str(V3C[key]), key)
        for flag in ['--synth', '--fullft', '--swa']:  # v3c booleans that are ON
            self.assertIn(flag, command)
        self.assertNotIn('--mlp', command)             # v3c ran without --mlp
        self.assertFalse((ROOT / 'artifacts' / 'fixture-recipe-echo').exists())

    def test_grouped_recipe_overrides_are_declared_and_validated(self):
        override = self.data_dir / 'override.json'
        override.write_text(json.dumps({'lr': 0.0005, 'fullft': False}))
        result = self._launch('--recipe', str(override))
        self.assertEqual(result.returncode, 0, result.stderr)
        command = command_from(result.stdout)
        self.assertEqual(command[command.index('--lr') + 1], '0.0005')  # the declared value
        self.assertNotIn('--fullft', command)                            # deviation honored
        self.assertIn('Recipe overrides vs configs/v3c.json:', result.stdout)
        self.assertIn('0.0001', result.stdout)                           # v3c default echoed as baseline
        # Unknown keys and wrongly-typed values are refused; nothing is assembled.
        for bad in [{'oops': 1}, {'epochs': True}, {'lr': '1e-4'}, {'swa': 1}, {'epochs': 2.5}]:
            path = self.data_dir / 'bad.json'
            path.write_text(json.dumps(bad))
            refused = self._launch('--recipe', str(path))
            self.assertNotEqual(refused.returncode, 0, bad)
            self.assertNotIn('Command:', refused.stdout)
        # --legacy refuses overrides entirely: it reproduces v3c unchanged.
        legacy = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'train.py'),
                                 '--run', 'fixture-recipe-echo', '--legacy',
                                 '--recipe', str(override), '--dry-run'],
                                capture_output=True, text=True)
        self.assertNotEqual(legacy.returncode, 0)
        self.assertIn('--legacy reproduces', legacy.stderr)

    def test_recipe_rejects_non_finite_numbers(self):
        # Red-team LOW-1: json.loads parses the JSON NaN/Infinity literals into
        # float nan/inf, which would poison optimizer hyperparameters and later
        # report JSON. All three must exit through parser.error (rc=2) with
        # nothing assembled; the message names the key and the reason but never
        # echoes the value itself.
        for literal in ['{"lr": NaN}', '{"lr": Infinity}', '{"lr": -Infinity}']:
            path = self.data_dir / 'nonfinite.json'
            path.write_text(literal)
            refused = self._launch('--recipe', str(path))
            self.assertEqual(refused.returncode, 2, literal)   # argparse parser.error exit
            self.assertNotIn('Command:', refused.stdout)       # nothing was assembled
            self.assertIn("'lr'", refused.stderr)              # key named
            self.assertIn('finite', refused.stderr)            # reason given, no value echoed
        # Finite float/int overrides keep flowing through unchanged.
        finite = self.data_dir / 'finite.json'
        finite.write_text(json.dumps({'lr': 1e-4, 'epochs': 3}))
        accepted = self._launch('--recipe', str(finite))
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        command = command_from(accepted.stdout)
        self.assertEqual(command[command.index('--lr') + 1], '0.0001')
        self.assertEqual(command[command.index('--epochs') + 1], '3')


if __name__ == '__main__':
    unittest.main()
