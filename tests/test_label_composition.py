"""Synthetic #8/#18/#24 boundaries only: no model, observations, or training."""
import contextlib, copy, io, itertools, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import label_schema as intake
import task_label_schema as semantic
import prepare
import split_v2
from test_label_schema import fixture, snapshot, write_inputs
from test_task_label_schema import v2_rows, write_fixture
import test_calibration_groups as calibration

ROOT=Path(__file__).resolve().parents[1]

class CompositionTests(unittest.TestCase):
    def test_distinct_public_apis_and_namespaces(self):
        self.assertTrue(callable(intake.unify));self.assertTrue(callable(semantic.questions_for))
        self.assertFalse(hasattr(intake,'questions_for'));self.assertFalse(hasattr(semantic,'unify'))
        self.assertEqual(semantic.metadata(semantic.V2)['task_label_schema_version'],semantic.V2)
        self.assertNotIn('label_schema_version',semantic.metadata(semantic.V2))

    def test_source_contract_integer_is_not_task_version(self):
        for selected in (semantic.LEGACY,semantic.V2):
            row={'label':{}}
            if selected==semantic.V2:row['task_label_schema_version']=selected
            row['label_schema_version']=1
            semantic.check_row_versions([row],selected)
            for invalid in (True,False,0,2,None,'router-label-v1','router-label-v2'):
                for nested in (False,True):
                    bad=copy.deepcopy(row);target=bad['label'] if nested else bad
                    target['label_schema_version']=invalid
                    with self.subTest(selected=selected,invalid=invalid,nested=nested),self.assertRaises(ValueError):
                        semantic.check_row_versions([bad],selected)

    def test_measured_shadow_never_becomes_supervision(self):
        tasks,experts,routes,envelope=fixture()
        _,shadow=intake.unify(tasks,experts,envelope,routes)
        for selected in (semantic.LEGACY,semantic.V2):
            with self.assertRaisesRegex(ValueError,'measured shadow'):
                semantic.validate_supervised_rows(shadow,selected)
            row={'label':envelope['labels'][0]}
            with self.assertRaisesRegex(ValueError,'measured shadow'):
                semantic.check_row_versions([row],selected)

    def test_v2_unsupported_prepare_flags_reject_before_input_reads(self):
        for key,value in [('unified',True),('preflight',True),('outcomes',{}),('routes',[])]:
            with patch.object(prepare,'load',side_effect=AssertionError('NO INPUT READ')) as read:
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'does not support'):
                    prepare.prepare(Path('missing'),schema_version=semantic.V2,output=Path('missing-output'),**{key:value})
                read.assert_not_called()

    def test_v2_cli_all_unsupported_combinations_reject_before_reads(self):
        groups=[['--unified'],['--outcomes','missing'],['--routes','missing'],['--preflight'],['--split-version','split-v2']]
        for count in range(1,len(groups)+1):
            for flags in itertools.combinations(groups,count):
                args=['--label-schema',semantic.V2,'--data','missing','--output','missing-output',*[v for flag in flags for v in flag]]
                with self.subTest(flags=flags),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                    prepare.main(args)
                self.assertEqual(error.exception.code,2)

    def test_v2_input_cannot_silently_enter_either_v1_splitter(self):
        tasks,experts,routes,envelope=fixture();experts[0]['task_label_schema_version']=semantic.V2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_inputs(root,tasks,experts,envelope,routes);before=snapshot(root)
            with self.assertRaises(ValueError):prepare.prepare(root,unified=True)
            with self.assertRaises(ValueError):split_v2.prepare_v2(root,root/'new')
            self.assertEqual(before,snapshot(root))

    def test_ambiguous_semantic_source_field_fails_v1_before_outputs(self):
        tasks,experts,routes,envelope=fixture();experts[0]['label_schema_version']=semantic.V2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_inputs(root,tasks,experts,envelope,routes);before=snapshot(root)
            with self.assertRaises(ValueError):prepare.prepare(root,unified=True)
            with self.assertRaises(ValueError):split_v2.prepare_v2(root,root/'new')
            self.assertEqual(before,snapshot(root))

    def test_v2_expert_confidence_and_nested_json_fail_before_output(self):
        tasks,experts,routes,envelope=fixture()
        for task in tasks:task.pop('synthetic',None)
        for label in experts:label.update(task_label_schema_version=semantic.V2,label_status='resolved')
        for mutation in ('bool-confidence','nan','utf8','ambiguous','task-mismatch'):
            t,e=copy.deepcopy(tasks),copy.deepcopy(experts)
            if mutation=='bool-confidence':e[0]['confidence']=True
            elif mutation=='nan':e[0]['audit']={'value':float('nan')}
            elif mutation=='utf8':t[0]['metadata']='\ud800'
            elif mutation=='ambiguous':e[0]['label_schema_version']=semantic.V2
            else:t[0]['task_label_schema_version']=semantic.LEGACY
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);write_inputs(root,t,e,envelope,routes);out=root/'new';before=snapshot(root)
                with self.assertRaises(ValueError):prepare.prepare(root,schema_version=semantic.V2,output=out)
                self.assertFalse(out.exists());self.assertEqual(before,snapshot(root))

    def test_preflight_rejects_shadow_in_train_no_model_no_test(self):
        from scripts.validate_calibration import run_validation
        for selected in (semantic.LEGACY,semantic.V2):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);_,_,roles=write_fixture(root,selected)
                rows=prepare.load(root/'train.jsonl');rows[0]['partition']='measured_shadow'
                calibration.write_jsonl(root/'train.jsonl',rows)
                ok,text=run_validation(root,roles,selected)
                self.assertFalse(ok);self.assertIn('measured shadow',text)
                self.assertEqual((root/'test.jsonl').read_text(),'DO NOT READ TEST: INVALID JSON')

    def test_old_checkpoint_bare_ambiguous_version_rejected(self):
        for value in (1,semantic.V2):
            with self.assertRaises(ValueError):semantic.resolve_run_schema({'label_schema_version':value})
        self.assertEqual(semantic.resolve_run_schema({}),semantic.LEGACY)

if __name__=='__main__':unittest.main()
