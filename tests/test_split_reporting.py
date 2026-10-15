"""Invented CPU-only rows. No private corpus, model, tensor or GPU imports."""
import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from split_reporting import split_report
from model_utils import QUESTIONS


def row(i,group='big',true='flagship',pred='flagship',vision='yes',root='source-a'):
    labels={'tier':true,'needs_vision':vision,'task_type':'extraction'}
    answers={}
    for field,spec in QUESTIONS.items():
        choice=pred if field=='tier' else 'yes' if field=='vision' else 'extraction'
        answers[field]={'choice':choice,'confidence':1.,'probabilities':{k:float(k==choice) for k in spec['criteria']}}
    return {'id':f'private-id-{i}','prompt':'PRIVATE PROMPT SENTINEL','split_group':group,'group':root,'label':labels,'answers':answers}


def source(n=0):
    return {'splits':{'test':{'rows':n,'tiers':{'flagship':n},'vision':{'yes':n},'task_types':{'extraction':n},'groups':['PRIVATE ID']}}}


class SplitReportTests(unittest.TestCase):
    def test_aggregate_only_never_fabricates_groups_metrics_or_intervals(self):
        result=split_report(source(79))
        self.assertEqual(result['aggregate_input_counts']['test']['tier']['flagship'],79)
        for key in ['row_support','group_provenance','evaluation','group_deletion','deletion_metric_ranges']:
            self.assertEqual(result[key]['status'],'unavailable')
        self.assertNotIn('PRIVATE ID',json.dumps(result))
        self.assertIn('known skew',result['dataset_notice'])

    def test_large_group_deletion_exposes_dependency_and_support(self):
        rows=[row(i) for i in range(79)]+[row(80,'small','fast','balanced','no','source-b')]
        before=copy.deepcopy(rows);result=split_report(source(80),rows)
        self.assertEqual(rows,before)
        self.assertEqual(result['evaluation']['tier']['accuracy'],79/80)
        deletion=result['group_deletion']['results'][0]
        self.assertEqual(deletion['removed_rows'],79)
        self.assertEqual(deletion['remaining_support']['rows'],1)
        self.assertEqual(deletion['remaining_support']['classes']['tier']['flagship'],0)
        self.assertEqual(deletion['remaining_provenance']['observed_split_groups'],1)
        self.assertEqual(deletion['metrics']['tier']['accuracy'],0)
        self.assertIsNone(deletion['metrics']['vision_per_class']['yes']['recall'])
        self.assertEqual(deletion['change_from_full']['tier_accuracy'],-79/80)
        self.assertIsNone(deletion['change_from_full']['vision_recall'])
        self.assertEqual(deletion['metrics']['bootstrap']['tier_accuracy']['status'],'unavailable')
        self.assertEqual(result['deletion_metric_ranges']['vision_recall']['undefined_deletions'],1)
        self.assertEqual(result['evaluation']['bootstrap']['vision_recall']['supporting_groups'],1)
        self.assertIsNone(result['evaluation']['bootstrap']['vision_recall']['low'])
        text=json.dumps(result,allow_nan=False)
        for secret in ['PRIVATE PROMPT SENTINEL','private-id-','source-a','source-b','"big"','"small"']:
            self.assertNotIn(secret,text)

    def test_empty_and_single_group_are_explicit(self):
        empty=split_report(source(),[])
        self.assertIsNone(empty['evaluation']['tier']['accuracy'])
        self.assertEqual(empty['group_deletion']['status'],'unavailable')
        self.assertEqual(empty['group_provenance']['observed_split_groups'],0)
        one=split_report(source(2),[row(0),row(1)])
        deleted=one['group_deletion']['results'][0]
        self.assertEqual(deleted['remaining_support']['rows'],0)
        self.assertIsNone(deleted['metrics']['tier']['macro_f1'])
        self.assertIsNone(deleted['metrics']['vision_false_negative_rate'])
        self.assertEqual(one['evaluation']['bootstrap']['tier_accuracy']['supporting_groups'],1)
        json.dumps(one,allow_nan=False)

    def test_label_only_rows_group_support_without_scores(self):
        rows=[row(0),row(1,'other')]
        for r in rows:r.pop('answers')
        result=split_report(source(2),rows)
        self.assertEqual(result['group_provenance']['observed_split_groups'],2)
        self.assertEqual(result['evaluation']['status'],'unavailable')
        self.assertEqual(result['group_deletion']['status'],'unavailable')

    def test_missing_group_never_uses_root_or_id(self):
        for group in [None,'',1,'  ']:
            r=row(0);r['split_group']=group
            result=split_report(source(1),[r])
            self.assertEqual(result['group_provenance']['status'],'unavailable')
            self.assertEqual(result['evaluation']['status'],'unavailable')

    def test_root_and_split_group_are_distinct_and_aliases_stable(self):
        rows=[row(0,'a',root='root-z'),row(1,'a',root='root-y'),row(2,'b',root='root-z')]
        result=split_report(source(3),rows);p=result['group_provenance']
        self.assertEqual(p['observed_split_groups'],2)
        self.assertEqual(p['root_provenance']['observed_roots'],2)
        self.assertEqual(len(p['groups'][0]['roots']),2)
        kept=result['group_deletion']['results'][0]['remaining_provenance']['groups'][0]
        self.assertEqual(kept['group'],p['groups'][1]['group'])
        self.assertEqual(kept['roots'],p['groups'][1]['roots'])
        rows[0].pop('group')
        self.assertEqual(split_report(source(3),rows)['group_provenance']['root_provenance']['status'],'unavailable')

    def test_reproducible_and_probability_fail_closed(self):
        rows=[row(0,'a'),row(1,'b'),row(2,'c',pred='fast')]
        self.assertEqual(split_report(source(3),rows),split_report(source(3),rows))
        self.assertEqual(split_report(source(999),rows)['aggregate_row_count_matches'],False)
        rows[0]['answers']['tier']['confidence']=float('nan')
        with self.assertRaises(ValueError):split_report(source(3),rows)

    def test_cli_is_stdlib_only_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'split.json').write_text(json.dumps(source(2)))
            (p/'rows.jsonl').write_text('\n'.join(json.dumps(row(i,str(i))) for i in range(2)))
            command=[sys.executable,'-S',str(ROOT/'scripts/report_split.py'),'--split-report',str(p/'split.json'),'--predictions',str(p/'rows.jsonl'),'--output',str(p/'report.json')]
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            payload=(p/'report.json').read_bytes();result=json.loads(payload)
            self.assertEqual(len(result['input_digests']['prediction_input_sha256']),64)
            self.assertNotIn(temp,json.dumps(result))
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual((p/'report.json').read_bytes(),payload)

    def test_training_attaches_both_report_paths_after_selection(self):
        tree=ast.parse((ROOT/'train_eval.py').read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='split_report']
        self.assertEqual(sorted(len(node.args) for node in calls),[1,2,2])
        # No report call participates in selection/temperature fitting.
        self.assertTrue(all(node.lineno>next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='effective' for t in n.targets)) for node in calls))


if __name__=='__main__':unittest.main()
