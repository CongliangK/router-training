"""Issue #3: invented probability fixtures only; no models, data, or GPU.

Import the actual training helpers using the existing stubbing harness.
Expected values below are hand-computed, not snapshots of the implementation.
"""
import argparse
import ast
import copy
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_training_reproducibility import train_eval


def row(i=0,true='fast',pred='fast',confidence=.4,group=None):
    te=train_eval();answers={}
    for field,spec in te.QUESTIONS.items():
        labels=list(spec['criteria']);chosen=pred if field=='tier' else labels[0]
        answers[field]={'choice':chosen,'confidence':confidence,
                        'probabilities':{c:confidence if c==chosen else (1-confidence)/(len(labels)-1) for c in labels}}
    return {'id':f'synthetic-{i}','split_group':group if group is not None else f'group-{i}',
            'label':{'tier':true,'needs_vision':'no','task_type':'extraction'},
            'answers':answers,'context_truncated':False,'latency_ms':2}


class MetricTests(unittest.TestCase):
    def test_boundary_regression_and_neighbors_partition_exactly_once(self):
        te=train_eval()
        values=[0.,1.]
        for edge in [.2,.4,.6,.8]:values.extend([math.nextafter(edge,0),edge,math.nextafter(edge,1)])
        for value in values:
            with self.subTest(value=value):
                result=te.metrics([row(0,confidence=value),row(1,confidence=value)])['tier']
                self.assertAlmostEqual(result['ece_5_bins'],1-value)
                self.assertEqual(sum(b['count'] for b in result['ece_bins']),2)
                self.assertAlmostEqual(sum(b['weight'] for b in result['ece_bins']),1)
                self.assertTrue(0<=result['ece_5_bins']<=1)
                index=sum(value>=edge for edge in [.2,.4,.6,.8])
                self.assertEqual(result['ece_bins'][index]['count'],2)
        self.assertAlmostEqual(te.metrics([row()])['tier']['ece_5_bins'],.6)

    def test_hand_computed_probability_metrics(self):
        rows=[row(0,confidence=.8),row(1,true='balanced',confidence=.6)]
        result=train_eval().metrics(rows)
        self.assertEqual(result['tier']['accuracy'],.5)
        self.assertAlmostEqual(result['tier']['nll'],(-math.log(.8)-math.log(.2))/2)
        # Correct [.8,.1,.1] costs .06; wrong [.6,.2,.2] costs 1.04.
        self.assertAlmostEqual(result['tier']['brier'],.55)
        self.assertAlmostEqual(result['tier']['ece_5_bins'],.4)
        self.assertEqual(result['tier']['confusion']['balanced']['fast'],1)
        self.assertAlmostEqual(result['tier_per_class']['fast']['precision'],.5)
        self.assertAlmostEqual(result['tier_per_class']['fast']['recall'],1)
        self.assertAlmostEqual(result['tier']['macro_f1'],(2/3)/3)
        self.assertAlmostEqual(result['task_type']['macro_f1'],1/7)
        self.assertEqual(result['task_type_per_class']['extraction']['support'],2)
        self.assertEqual(result['task_type_per_class']['creative']['support'],0)
        json.dumps(result,allow_nan=False)

    def test_zero_one_probabilities_and_nll_floor(self):
        te=train_eval()
        correct=te.metrics([row(confidence=1)])['tier']
        self.assertEqual((correct['nll'],correct['brier'],correct['ece_5_bins']),(0,0,0))
        wrong=te.metrics([row(true='balanced',confidence=1)])['tier']
        self.assertAlmostEqual(wrong['nll'],-math.log(1e-12))
        self.assertEqual((wrong['brier'],wrong['ece_5_bins']),(2,1))

    def test_empty_and_unknown_labels_have_no_scores_or_intervals(self):
        te=train_eval();unknown=row();unknown['label']={k:'unknown' for k in unknown['label']}
        for rows in [[],[unknown]]:
            result=te.metrics(rows)
            for field in te.QUESTIONS:
                self.assertEqual(result[field]['evaluated_rows'],0)
                for key in ['accuracy','macro_f1','nll','brier','ece_5_bins']:self.assertIsNone(result[field][key])
                for label in te.QUESTIONS[field]['criteria']:
                    pc=result[field+'_per_class'][label]
                    self.assertEqual(pc['support'],0)
                    for key in ['precision','recall','f1']:self.assertIsNone(pc[key])
            for interval in result['bootstrap'].values():
                self.assertEqual(interval['status'],'unavailable')
                self.assertIsNone(interval['low']);self.assertIsNone(interval['high'])
            self.assertIsNone(result['routing_costs']['mean_cost'])
            json.dumps(result,allow_nan=False)
        self.assertIsNone(te.metrics([])['mean_latency_ms'])

    def test_fixed_macro_and_low_support_threshold(self):
        te=train_eval();rows=[row(i,confidence=1) for i in range(10)]
        pc=te.metrics(rows)['tier_per_class']
        self.assertEqual(pc['macro_f1'],1/3)
        self.assertEqual(pc['macro_f1_definition']['denominator'],3)
        self.assertEqual(pc['macro_f1_definition']['excluded_classes'],[])
        self.assertFalse(pc['fast']['low_support'])
        self.assertTrue(pc['balanced']['low_support'])
        self.assertIsNone(pc['balanced']['f1'])
        self.assertEqual(te.per_class(rows,'tier',11)['low_support_classes'],list(te.TIERS))
        for invalid in [0,-1,True,1.2]:
            with self.assertRaises(ValueError):te.metrics(rows,low_support_threshold=invalid)

    def test_routing_counts_and_custom_weighted_cost_matrix(self):
        te=train_eval();rows=[row(0,'flagship','fast'),row(1,'balanced','fast'),row(2,'fast','flagship'),row(3,'fast','balanced'),row(4)]
        cost=te.metrics(rows)['routing_costs']
        self.assertEqual(cost['downgrade'],{'count':2,'weighted_cost':3})
        self.assertEqual(cost['overroute'],{'count':2,'weighted_cost':3})
        self.assertEqual(cost['mean_cost'],6/5)
        matrix={a:{b:(0 if a==b else 10 if te.TIERS.index(a)>te.TIERS.index(b) else 2) for b in te.TIERS} for a in te.TIERS}
        cost=te.metrics(rows,cost_matrix=matrix)['routing_costs']
        self.assertEqual(cost['downgrade']['weighted_cost'],20)
        self.assertEqual(cost['overroute']['weighted_cost'],4)
        self.assertEqual(cost['total_weighted_cost'],24)
        self.assertEqual(sum(x['count'] for cells in cost['by_transition'].values() for x in cells.values()),5)
        for invalid in [{}, {**matrix,'bad':{}}, {**matrix,'fast':{'fast':1,'balanced':2,'flagship':2}}]:
            with self.assertRaises(ValueError):te.metrics(rows,cost_matrix=invalid)
        for value in [True,-1,float('nan'),float('inf')]:
            invalid=copy.deepcopy(matrix);invalid['fast']['balanced']=value
            with self.assertRaises(ValueError):te.metrics(rows,cost_matrix=invalid)

    def test_bad_probabilities_fail_closed(self):
        te=train_eval()
        for bad in [float('nan'),float('inf'),-.01,1.01,True]:
            r=row();r['answers']['tier']['confidence']=bad
            with self.assertRaises(ValueError):te.metrics([r])
        for mutation in [lambda a:a['probabilities'].pop('fast'),
                         lambda a:a['probabilities'].update(fast=.8),
                         lambda a:a.update(choice='unknown'),
                         lambda a:a.update(confidence=.5),
                         lambda a:a['probabilities'].update(fast=float('nan'))]:
            r=row();mutation(r['answers']['tier'])
            with self.assertRaises(ValueError):te.metrics([r])


class BootstrapTests(unittest.TestCase):
    def test_empty_single_group_missing_group_and_missing_denominator(self):
        te=train_eval()
        for rows in [[],[row()], [row(0,group='same'),row(1,group='same')]]:
            result=te.metrics(rows)['bootstrap']['tier_accuracy']
            self.assertEqual(result['status'],'unavailable');self.assertIsNone(result['low'])
            self.assertEqual(result['supporting_groups'],1 if rows else 0)
        for value in [None,'',1]:
            rows=[row(0),row(1)];rows[0]['split_group']=value
            self.assertEqual(te.metrics(rows)['bootstrap']['tier_accuracy']['status'],'unavailable')
        rows=[row(0),row(1)];rows[0].pop('split_group')
        self.assertEqual(te.metrics(rows)['bootstrap']['tier_accuracy']['status'],'unavailable')
        result=te.metrics([row(0),row(1)])['bootstrap']
        for name in ['flagship_precision','flagship_recall','vision_recall']:
            self.assertEqual(result[name]['status'],'unavailable');self.assertIsNone(result[name]['high'])

    def test_only_one_supporting_group_is_not_a_confident_interval(self):
        result=train_eval().metrics([row(0,'flagship','flagship'),row(1)])['bootstrap']
        self.assertEqual(result['tier_accuracy']['status'],'available')
        for name in ['flagship_recall','flagship_precision']:
            self.assertEqual(result[name]['status'],'unavailable')
            self.assertEqual(result[name]['supporting_groups'],1)

    def test_group_resampling_retains_cluster_composition_and_seed(self):
        te=train_eval();rows=[row(0,group='a'),row(1,group='a'),row(2,group='b')];seen=[]
        def statistic(draw):
            ids=[r['id'] for r in draw]
            self.assertEqual(ids.count('synthetic-0'),ids.count('synthetic-1'))
            seen.append(ids)
            return sum(r['split_group']=='a' for r in draw)/len(draw)
        a=te.bootstrap_ci(rows,'tier',statistic,samples=100,seed=7)
        b=te.bootstrap_ci(rows,'tier',statistic,samples=100,seed=7)
        self.assertEqual(a,b);self.assertEqual((a['low'],a['high']),(0,1))
        self.assertEqual(a['independent_groups'],2);self.assertEqual(a['valid_resamples'],100)
        self.assertTrue(any(len(draw)==4 for draw in seen))
        for n in [0,1,True,1.1]:
            with self.assertRaises(ValueError):te.bootstrap_ci(rows,'tier',statistic,samples=n)

    def test_ratio_bootstrap_samples_full_field_pool_and_discloses_undefined_draws(self):
        te=train_eval();rows=[row(0,'flagship','flagship'),row(1,'flagship','fast'),row(2)]
        result=te.metrics(rows)['bootstrap']['flagship_recall']
        self.assertEqual(result['evaluated_rows'],3);self.assertEqual(result['independent_groups'],3)
        self.assertEqual(result['supporting_groups'],2)
        # Independent reference: draws with only group 2 have zero denominator.
        rng=random.Random(0);undefined=0;values=[]
        for _ in range(1000):
            draw=[rng.randrange(3) for _ in range(3)]
            denominator=sum(i<2 for i in draw)
            if denominator:values.append(sum(i==0 for i in draw)/denominator)
            else:undefined+=1
        values.sort()
        self.assertGreater(undefined,0)
        self.assertEqual(result['undefined_resamples'],undefined)
        self.assertEqual(result['valid_resamples'],len(values))
        self.assertEqual(result['low'],values[int(.025*len(values))])
        self.assertEqual(result['high'],values[int(.975*len(values))])

    def test_predict_rows_preserves_source_group_without_inventing_one(self):
        te=train_eval();rows=[{**row(i),'prompt':'synthetic'} for i in range(2)];rows[1].pop('split_group')
        with mock.patch.object(te,'score',return_value={'answers':{}}):scored=te.predict_rows(None,rows)
        self.assertEqual(scored[0]['split_group'],'group-0')
        self.assertNotIn('split_group',scored[1])


class SelectionSupportTests(unittest.TestCase):
    def test_real_legacy_run_rejects_unsupported_selection_before_model_work(self):
        te=train_eval()
        for rows in [[],[row(0,true='unsupported'),row(1,true='unsupported')]]:
            with self.subTest(rows=len(rows)), tempfile.TemporaryDirectory() as directory:
                args=argparse.Namespace(legacy_calibration=True,seed=0,synth=False,smoke=False,run='synthetic-selection')
                with mock.patch.object(te,'ROOT',Path(directory)), \
                     mock.patch.object(te,'read_jsonl',return_value=rows), \
                     mock.patch.object(te,'load_engine') as load, \
                     mock.patch.object(te,'predict_rows') as predict:
                    with self.assertRaisesRegex(ValueError,'checkpoint selection requires'):
                        te.run(args)
                    load.assert_not_called();predict.assert_not_called()

    def test_supported_legacy_selection_keeps_original_preflight_behavior(self):
        te=train_eval();rows=[row(0),row(1)]
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(legacy_calibration=True,seed=0,synth=False,smoke=False,run='synthetic-selection')
            with mock.patch.object(te,'ROOT',Path(directory)), \
                 mock.patch.object(te,'read_jsonl',return_value=rows), \
                 mock.patch.object(te,'load_engine',side_effect=RuntimeError('model boundary sentinel')) as load:
                with self.assertRaisesRegex(RuntimeError,'model boundary sentinel'):te.run(args)
                load.assert_called_once()

    def test_actual_epoch_and_soup_score_assignments_reject_unavailable_metrics(self):
        # Execute the actual assignments inside run(), not a retyped copy of
        # their comparisons. This tests that both expensive consumers call
        # the guard before sorting/comparing/rounding any nullable metric.
        te=train_eval();tree=ast.parse((ROOT/'train_eval.py').read_text())
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        assignments={node.targets[0].id:node for node in ast.walk(run)
                     if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                     and node.targets[0].id in ['score','soup_f1']}
        self.assertEqual(set(assignments),{'score','soup_f1'})
        for rows in [[],[row(true='unsupported')]]:
            report=te.metrics(rows)
            for name,node in assignments.items():
                namespace={'validation_metrics':report,'soup_metrics':report,'require_selection_score':te.require_selection_score}
                with self.subTest(name=name,rows=len(rows)):
                    with self.assertRaisesRegex(ValueError,'macro-F1 is unavailable'):
                        exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual run assignment>','exec'),namespace)
        for value in [0,.5,1]:self.assertEqual(te.require_selection_score(value),value)
        for value in [None,True,float('nan'),float('inf'),-1,1.1]:
            with self.assertRaisesRegex(ValueError,'macro-F1 is unavailable'):te.require_selection_score(value)

    def test_swa_rejects_unavailable_history_before_snapshot(self):
        te=train_eval()
        for values in [[None],[.5,None],[None,.5]]:
            history=[{'epoch':i+1,'calibration_macro_f1':value} for i,value in enumerate(values)]
            snapshot=mock.Mock()
            with self.assertRaisesRegex(ValueError,'macro-F1 is unavailable'):te.top_swa_epochs(history)
            with self.assertRaisesRegex(ValueError,'macro-F1 is unavailable'):te.update_swa_pool([],history,snapshot)
            snapshot.assert_not_called()
        self.assertEqual(te.top_swa_epochs([{'epoch':1,'calibration_macro_f1':0},{'epoch':2,'calibration_macro_f1':0}]),[1,2])


class EceGateTests(unittest.TestCase):
    def test_ece_can_only_veto_never_qualify(self):
        te=train_eval();evaluation={'tier':{'ece_5_bins':.2}}
        for passed in [False,True]:
            gate={'passed':passed,'mode':'qualified' if passed else 'shadow','reasons':['original evidence']}
            before=copy.deepcopy(gate)
            within=te.apply_ece_veto(gate,evaluation,.25)
            self.assertEqual(within['passed'],passed)
            self.assertEqual(within['calibration_constraint']['status'],'within_limit')
            veto=te.apply_ece_veto(gate,evaluation,.1)
            self.assertFalse(veto['passed']);self.assertEqual(veto['mode'],'shadow')
            self.assertEqual(veto['reasons'][0],'original evidence');self.assertEqual(gate,before)
            default=te.apply_ece_veto(gate,evaluation)
            self.assertEqual(default['passed'],passed)
            self.assertEqual(default['calibration_constraint']['status'],'not_registered')
        self.assertEqual(te.apply_ece_veto({'passed':False},evaluation,.2)['calibration_constraint']['status'],'within_limit')

    def test_unavailable_or_invalid_ece_vetoes_and_threshold_validates(self):
        te=train_eval();gate={'passed':True,'mode':'qualified'}
        for value in [None,True,float('nan'),float('inf'),-1,2]:
            result=te.apply_ece_veto(gate,{'tier':{'ece_5_bins':value}},.25)
            self.assertFalse(result['passed']);self.assertEqual(result['calibration_constraint']['status'],'unavailable')
            json.dumps(result,allow_nan=False)
        self.assertFalse(te.apply_ece_veto(gate,None,.25)['passed'])
        for threshold in [True,'0.25',float('nan'),float('inf'),-1,1.1]:
            with self.assertRaises(ValueError):te.apply_ece_veto(gate,None,threshold)

    def test_launcher_forwards_registered_threshold_without_gpu(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--run','synthetic-metrics-dry-run','--legacy','--dry-run','--max-tier-ece','.25'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("'--max-tier-ece', '0.25'",result.stdout)
        self.assertFalse((ROOT/'artifacts/synthetic-metrics-dry-run').exists())
        for value in ['nan','inf','-1','1.1']:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--run','synthetic-metrics-dry-run','--legacy','--dry-run','--max-tier-ece',value],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)


if __name__=='__main__':unittest.main()
