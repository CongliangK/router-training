"""Synthetic-only combined contract regression; no private inputs or models."""
import copy, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_split_v2 import fixture, write_inputs
from task_label_schema import V2, LEGACY, metadata, validate_dataset_schema, validate_supervised_rows
from split_task_v2 import split_task_v2, prepare_task_split_v2, CONTRACT

ROOT=Path(__file__).resolve().parents[1]


def inputs(groups=12,size=2):
    tasks,labels=fixture(groups,size)
    # Programmatic synthetic fixtures exercise primary-row contracts; do not
    # present these test fixtures as supplemental or real-data support.
    for task in tasks:
        task.pop('synthetic');task['task_label_schema_version']=V2
    for i,label in enumerate(labels):
        label.update(task_label_schema_version=V2,label_status='resolved',
                     task_type=metadata(V2)['classes']['task_type'][i%6],needs_vision=('yes','no')[i%2])
    return tasks,labels


class CombinedTests(unittest.TestCase):
    def test_cli_bundle_existing_validators_and_input_immutability(self):
        tasks,labels=inputs()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'input';root.mkdir();write_inputs(root,tasks,labels)
            before={p.name:p.read_bytes() for p in root.iterdir()};out=Path(temp)/'new'
            result=subprocess.run([sys.executable,str(ROOT/'prepare.py'),'--data',str(root),'--output',str(out),
                '--label-schema',V2,'--split-version','split-v2'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(validate_dataset_schema(out,V2),V2)
            report=json.loads((out/'split-report.json').read_text())
            self.assertEqual(report['bundle_contract'],CONTRACT)
            self.assertEqual(report['label_schema'],metadata(V2))
            self.assertNotIn('uncertain',report['class_component_support']['needs_vision'])
            self.assertNotIn('visual',report['class_component_support']['task_type'])
            roots={};components={};seen=[]
            for name in ('train','calibration','test'):
                rows=[json.loads(line) for line in (out/f'{name}.jsonl').read_text().splitlines()]
                validate_supervised_rows(rows,V2)
                self.assertLessEqual(report['splits'][name]['max_group_share'],.35)
                for row in rows:
                    self.assertEqual(roots.setdefault(row['group'],name),name)
                    self.assertEqual(components.setdefault(row['split_group'],name),name)
                    seen.append(row['id'])
            self.assertEqual(sorted(seen),sorted(r['id'] for r in tasks))
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            with self.assertRaises(FileExistsError):prepare_task_split_v2(root,out,schema_version=V2)
            link=Path(temp)/'link';link.symlink_to(Path(temp)/'missing')
            with self.assertRaises(FileExistsError):prepare_task_split_v2(root,link,schema_version=V2)

    def test_determinism_and_no_mutation(self):
        tasks,labels=inputs();original=copy.deepcopy((tasks,labels))
        a=split_task_v2(tasks,labels,schema_version=V2,attempts=100)
        b=split_task_v2(tasks[::-1],labels[::-1],schema_version=V2,attempts=100)
        self.assertEqual(a,b);self.assertEqual((tasks,labels),original)

    def test_pending_root_bridges_short_equality_before_exclusion(self):
        tasks,labels=inputs(15,2)
        # Root 1 is entirely pending, linking roots 0 and 2 by opposite rows.
        for i,prompt in ((0,' A B '),(2,'ab'),(3,'CD'),(4,' c d ')):
            tasks[i]['prompt']=prompt
        for i in (2,3):labels[i].update(label_status='adjudication',adjudication_reasons=['review pending'],needs_vision=None)
        splits,report,pending=split_task_v2(tasks,labels,schema_version=V2,attempts=100)
        self.assertIsNotNone(splits);self.assertEqual(len(pending),2)
        rows={r['id']:r for values in splits.values() for r in values}
        self.assertEqual(rows[tasks[0]['id']]['split_group'],rows[tasks[4]['id']]['split_group'])
        self.assertNotIn(tasks[2]['id'],rows)
        self.assertEqual(report['grouping']['cross_root_near_duplicate_edges'],2)

    def test_different_short_prompts_do_not_link(self):
        tasks,labels=inputs();tasks[0]['prompt']='a';tasks[2]['prompt']='b'
        _,report,_=split_task_v2(tasks,labels,schema_version=V2,attempts=1)
        self.assertEqual(report['grouping']['observed_components'],12)

    def test_fail_closed_versions_supervision_and_provenance(self):
        mutations=[lambda t,l:t[0].pop('task_label_schema_version'),
            lambda t,l:l[0].update(task_label_schema_version=LEGACY),
            lambda t,l:t[0].update(partition='measured_shadow'),
            lambda t,l:l[0].update(partition='measured_shadow'),
            lambda t,l:l[0].update(label_source='measured_candidate_outcomes'),
            lambda t,l:l[0].update(label_status='unknown'),
            lambda t,l:l[0].update(needs_vision='uncertain'),
            lambda t,l:l[0].update(task_type='visual'),
            lambda t,l:l[0].update(label_status='excluded'),
            lambda t,l:t[0].update(synthetic=True,supplemental_only=True),
            lambda t,l:l[0].update(label_schema_version=2),
            lambda t,l:l[0].update(confidence=float('nan'))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                tasks,labels=inputs();mutate(tasks,labels)
                with self.assertRaises(ValueError):split_task_v2(tasks,labels,schema_version=V2)
        tasks,labels=inputs()
        for version in (LEGACY,None,'bad'):
            with self.assertRaises(ValueError):split_task_v2(tasks,labels,schema_version=version)

    def test_infeasible_and_all_excluded_write_no_supervision(self):
        for exclude in (False,True):
            tasks,labels=inputs(3,1)
            if exclude:
                for label in labels:label.update(label_status='excluded',adjudication_reasons=['insufficient evidence'])
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'input';root.mkdir();write_inputs(root,tasks,labels);out=Path(temp)/'out'
                report=prepare_task_split_v2(root,out,schema_version=V2)
                self.assertEqual(report['status'],'infeasible');self.assertTrue(report['feasibility']['proven_infeasible'])
                self.assertEqual(report['outputs']['splits'],{})
                self.assertFalse((out/'train.jsonl').exists())
                validate_dataset_schema(out,V2)

    def test_search_exhaustion_not_infeasibility(self):
        tasks,labels=inputs()
        with patch('split_v2._candidates',return_value=iter([(0,)*12])):
            splits,report,_=split_task_v2(tasks,labels,schema_version=V2,attempts=1)
        self.assertIsNone(splits);self.assertEqual(report['status'],'search_exhausted')
        self.assertFalse(report['feasibility']['proven_infeasible'])

    def test_cli_guards_remain(self):
        for flag in ('--unified','--preflight','--outcomes','--routes'):
            args=[sys.executable,str(ROOT/'prepare.py'),'--label-schema',V2,'--split-version','split-v2',flag]
            if flag in ('--outcomes','--routes'):args.append('unused.json')
            result=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(result.returncode,2);self.assertIn('does not support',result.stderr)

if __name__=='__main__':unittest.main()
