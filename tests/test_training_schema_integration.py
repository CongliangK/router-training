"""CPU-only #18/#8 + #19/#20 integration; no model load/forward/training.

Fixtures are invented, test files are canaries, and every launcher call is dry-run
or patched to forbid a subprocess. Numerical capacity/loss tests remain separate.
"""
import argparse, ast, contextlib, copy, importlib.util, io, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research_training as rt
import task_label_schema as ls
from calibration_groups import questions_from_source


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


f=load('integration_calibration_fixtures',ROOT/'tests/test_calibration_groups.py')
launcher=load('integration_train_launcher',ROOT/'scripts/train.py')


def config():return rt.load_config(ROOT/'configs/research-v1.json')


def source_rows():
    rows=f.make_train_rows(n=30)
    for row in rows:
        row.update(label_schema_version=1,task_label_schema_version=ls.LEGACY)
        row['label'].update(label_source='expert_provisional',confidence=.8,
                            label_schema_version=1,task_label_schema_version=ls.LEGACY)
    return rows


def fixture(directory):
    rows=source_rows();pool=f.make_pool(n_groups=16)
    f.write_jsonl(directory/'train.jsonl',rows)
    f.write_jsonl(directory/'calibration.jsonl',pool)
    (directory/'test.jsonl').write_text('INVALID TEST CANARY: never read')
    roles=directory/'roles.json';roles.write_text(json.dumps(f.make_roles(pool)))
    return rows,roles


class SchemaCompositionTests(unittest.TestCase):
    def test_research_recipe_semantic_mapping_never_uses_source_integer(self):
        self.assertEqual(config()['schema_version'],'legacy7-v1')
        for selected in (None,ls.LEGACY):
            self.assertEqual(rt.check_task_schema(selected),'legacy7-v1')
            rt.check_questions(ls.questions_for(selected,questions_from_source()))
        for selected in (ls.V2,1,True,'1','legacy7-v1'):
            with self.subTest(selected=selected),self.assertRaises(ValueError):rt.check_task_schema(selected)
        with self.assertRaisesRegex(ValueError,'unsupported'):
            rt.check_args(config(),argparse.Namespace(label_schema=ls.V2))

    def test_old_checkpoint_and_questions_keep_legacy_meaning(self):
        q=questions_from_source()
        self.assertEqual(ls.questions_for(None,q),q)
        self.assertEqual(tuple(q['task_type']['criteria']),rt.TYPES)
        self.assertEqual(len(ls.questions_for(ls.V2,q)['task_type']['criteria']),6)
        self.assertEqual(ls.resolve_run_schema({}),ls.LEGACY)
        with self.assertRaises(ValueError):ls.resolve_run_schema({},requested=ls.V2)
        meta=ls.metadata(ls.V2)
        self.assertIn('task_label_schema_version',meta)
        self.assertNotIn('label_schema_version',meta)
        corrupt=copy.deepcopy(meta);corrupt['question_sha256']='0'*64
        with self.assertRaises(ValueError):ls.resolve_run_schema({'label_schema':corrupt},corrupt,ls.V2)

    def test_source_fields_preserve_sampling_values_and_input(self):
        rows=source_rows();plain=copy.deepcopy(rows)
        for row in plain:
            for obj in (row,row['label']):
                obj.pop('label_schema_version');obj.pop('task_label_schema_version')
        before=copy.deepcopy(rows)
        self.assertEqual(rt.sample_epochs(rows,config(),7,2)[1],rt.sample_epochs(plain,config(),7,2)[1])
        self.assertEqual(rows,before)
        self.assertTrue(all(type(r['label_schema_version']) is int for r in rows))
        self.assertTrue(all(isinstance(r['task_label_schema_version'],str) for r in rows))

    def test_sampler_rejects_semantic_source_and_shadow_confusion(self):
        mutations=[lambda r:r.update(label_schema_version=ls.LEGACY),
                   lambda r:r['label'].update(label_schema_version=ls.V2),
                   lambda r:r.update(label_schema_version=True),
                   lambda r:r.update(task_label_schema_version=1),
                   lambda r:r['label'].update(task_label_schema_version=1),
                   lambda r:r.update(task_label_schema_version=ls.V2),
                   lambda r:r['label'].update(task_label_schema_version=ls.V2),
                   lambda r:r.update(partition='measured_shadow'),
                   lambda r:r['label'].update(label_source='measured_candidate_outcomes')]
        for change in mutations:
            rows=source_rows();change(rows[0])
            with self.subTest(row=rows[0]),self.assertRaises(ValueError):rt.sampler_plan(rows,config())
        measured=source_rows()[0];measured['label'].update(label_source='measured_candidate_outcomes',confidence=.95)
        self.assertEqual(rt.confidence_weight(measured),1.)  # coverage is NOT expert certainty


class LauncherCompositionTests(unittest.TestCase):
    def launch(self,flags):
        out=io.StringIO();err=io.StringIO()
        with mock.patch.object(sys,'argv',['train.py',*flags]),contextlib.redirect_stdout(out),contextlib.redirect_stderr(err),mock.patch.object(launcher.subprocess,'call',side_effect=AssertionError('NO LAUNCH')):
            result=launcher.main()
        command=ast.literal_eval(next(line for line in out.getvalue().splitlines() if line.startswith('Command:')).split('Command:',1)[1])
        return result,command,out.getvalue()

    def test_default_and_explicit_v1_research_forward_selected_schema_and_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);_,roles=fixture(directory)
            flags=['--run','integration-research-cpu','--roles',str(roles),'--data-dir',str(directory),
                   '--research-config',str(ROOT/'configs/research-v1.json'),'--dry-run']
            before={p.name:p.read_bytes() for p in directory.iterdir()}
            default=self.launch(flags);explicit=self.launch(flags+['--label-schema',ls.LEGACY])
            self.assertEqual(default,explicit);self.assertEqual(default[0],0)
            command=default[1]
            self.assertEqual(command[command.index('--label-schema')+1],ls.LEGACY)
            self.assertEqual(command[command.index('--research-config')+1],str(ROOT/'configs/research-v1.json'))
            self.assertNotIn('--data-dir',command)  # v1 is still preflight-only
            for key,value in config()['training'].items():
                flag='--'+key.replace('_','-')
                if type(value) is bool:self.assertEqual(flag in command,value)
                else:self.assertEqual(command[command.index(flag)+1],str(value))
            self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir()})
            self.assertFalse((ROOT/'artifacts/integration-research-cpu').exists())

    def test_default_and_explicit_v1_nonresearch_keep_legacy_command(self):
        default=self.launch(['--run','integration-legacy-cpu','--legacy','--dry-run'])
        explicit=self.launch(['--run','integration-legacy-cpu','--legacy','--dry-run','--label-schema',ls.LEGACY])
        self.assertEqual(default,explicit);self.assertEqual(default[0],0)
        self.assertNotIn('--label-schema',default[1]);self.assertNotIn('--research-config',default[1])

    def test_v2_research_refused_before_config_data_sampling_output_or_launch(self):
        flags=['train.py','--run','integration-v2-refused','--label-schema',ls.V2,
               '--research-config','/not-read.json','--dry-run']
        with mock.patch.object(sys,'argv',flags),mock.patch.object(rt,'load_config',side_effect=AssertionError('NO CONFIG')) as config_read,mock.patch.object(rt,'sample_epochs',side_effect=AssertionError('NO SAMPLER')) as sampler,mock.patch.object(Path,'read_text',side_effect=AssertionError('NO INPUT')),mock.patch.object(Path,'mkdir',side_effect=AssertionError('NO OUTPUT')),mock.patch.object(launcher.subprocess,'call',side_effect=AssertionError('NO LAUNCH')) as process,contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as exit:launcher.main()
            self.assertEqual(exit.exception.code,2)
            self.assertIn('router-label-v2 research is unsupported',err.getvalue())
            config_read.assert_not_called();sampler.assert_not_called();process.assert_not_called()


class RuntimeCompositionTests(unittest.TestCase):
    def test_v2_research_fails_before_all_work_with_minimal_args(self):
        te=f.import_train_eval_with_stubs()
        args=argparse.Namespace(label_schema=ls.V2,research_config='/not-read.json')
        with mock.patch.object(rt,'load_config',side_effect=AssertionError('NO CONFIG')) as config_read,mock.patch.object(rt,'sample_epochs',side_effect=AssertionError('NO SAMPLER')) as sampler,mock.patch.object(te,'read_jsonl',side_effect=AssertionError('NO DATA')) as reader,mock.patch.object(te,'load_engine',side_effect=AssertionError('NO MODEL')) as model,mock.patch.object(te,'ensure_optimizer_available',side_effect=AssertionError('NO OPTIMIZER')) as optimizer,mock.patch.object(Path,'mkdir',side_effect=AssertionError('NO OUTPUT')):
            with self.assertRaisesRegex(ValueError,'router-label-v2 research is unsupported'):te.run(args)
            for sentinel in (config_read,sampler,reader,model,optimizer):sentinel.assert_not_called()

    def test_default_and_explicit_v1_runtime_preflight_share_selected_questions(self):
        for selected in (None,ls.LEGACY):
            with self.subTest(selected=selected),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data=root/'data';data.mkdir();_,roles=fixture(data)
                reads=[];te=f.import_train_eval_with_stubs(reads)
                args=argparse.Namespace(**config()['training'],run='integration-no-artifacts',smoke=False,
                    legacy_calibration=False,structural_dry_run=True,calibration_roles=str(roles),
                    research_config=str(ROOT/'configs/research-v1.json'))
                if selected is not None:args.label_schema=selected
                with mock.patch.object(te,'ROOT',root),mock.patch.object(te,'load_engine',side_effect=AssertionError('NO MODEL')) as model,mock.patch.object(rt,'sample_epochs',wraps=rt.sample_epochs) as sampler,mock.patch.object(rt,'check_questions',wraps=rt.check_questions) as questions,contextlib.redirect_stdout(io.StringIO()):
                    te.run(args);model.assert_not_called()
                    self.assertEqual(sampler.call_count,1)
                    self.assertEqual(tuple(questions.call_args.args[0]['task_type']['criteria']),rt.TYPES)
                self.assertEqual({Path(p).name for p in reads},{'train.jsonl','calibration.jsonl'})
                self.assertFalse((root/'artifacts').exists())

    def test_v1_runtime_bad_rows_fail_before_sampler_model_or_output(self):
        for update in ({'task_label_schema_version':ls.V2},{'label_schema_version':ls.LEGACY},{'partition':'measured_shadow'}):
            with self.subTest(update=update),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data=root/'data';data.mkdir();rows,roles=fixture(data)
                rows[0].update(update);f.write_jsonl(data/'train.jsonl',rows)
                te=f.import_train_eval_with_stubs()
                args=argparse.Namespace(**config()['training'],run='integration-refused',smoke=False,label_schema=ls.LEGACY,
                    legacy_calibration=False,structural_dry_run=True,calibration_roles=str(roles),
                    research_config=str(ROOT/'configs/research-v1.json'))
                with mock.patch.object(te,'ROOT',root),mock.patch.object(te,'load_engine',side_effect=AssertionError('NO MODEL')) as model,mock.patch.object(rt,'sample_epochs',side_effect=AssertionError('NO SAMPLER')) as sampler,mock.patch.object(Path,'mkdir',side_effect=AssertionError('NO OUTPUT')):
                    with self.assertRaises(ValueError):te.run(args)
                    model.assert_not_called();sampler.assert_not_called()
                self.assertFalse((root/'artifacts').exists())


class RepairedIntegrationGuards(unittest.TestCase):
    def test_research_relative_config_is_resolved_once_before_validation_and_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'dataset';data.mkdir();_,roles=fixture(data)
            recipe=root/'research.json';recipe.write_text(json.dumps(config()))
            moved=root/'changed-cwd';moved.mkdir();previous=Path.cwd()
            original=rt.load_config;calls=[]
            def read_config(path):
                calls.append(path);result=original(path);os.chdir(moved);return result
            try:
                os.chdir(root)
                flags=['train.py','--run','integration-repaired-path','--roles','dataset/roles.json',
                       '--data-dir','dataset','--research-config','research.json','--dry-run']
                with mock.patch.object(sys,'argv',flags),mock.patch.object(rt,'load_config',side_effect=read_config),mock.patch.object(launcher.subprocess,'call',side_effect=AssertionError('NO LAUNCH')),contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(launcher.main(),0)
                command=ast.literal_eval(next(s.split('Command:',1)[1] for s in out.getvalue().splitlines() if s.startswith('Command:')))
                self.assertEqual(calls,[recipe.resolve()])
                self.assertEqual(command[command.index('--research-config')+1],str(recipe.resolve()))
                self.assertEqual(command[command.index('--calibration-roles')+1],str(roles.resolve()))
                self.assertEqual(command[command.index('--label-schema')+1],ls.LEGACY)
                self.assertFalse((ROOT/'artifacts/integration-repaired-path').exists())
            finally:os.chdir(previous)

    def test_v2_research_rejects_before_even_dataset_metadata_gate(self):
        te=f.import_train_eval_with_stubs()
        args=argparse.Namespace(label_schema=ls.V2,research_config='never-read.json')
        with mock.patch.object(te,'validate_dataset_schema',side_effect=AssertionError('NO METADATA')) as metadata,mock.patch.object(rt,'load_config',side_effect=AssertionError('NO CONFIG')) as reader,mock.patch.object(Path,'mkdir',side_effect=AssertionError('NO OUTPUT')):
            with self.assertRaisesRegex(ValueError,'research is unsupported'):te.run(args)
            metadata.assert_not_called();reader.assert_not_called()

    def test_nonresearch_v2_metadata_precedes_all_data_optimizer_sampler_model_output(self):
        for name in ('label-schema.json','split-report.json'):
            with self.subTest(document=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data=root/'data';data.mkdir();fixture(data)
                # One valid document cannot substitute for the other; no rows are read.
                meta=ls.metadata(ls.V2)
                (data/name).write_text(json.dumps(meta if name=='label-schema.json' else {'label_schema':meta}))
                te=f.import_train_eval_with_stubs()
                args=argparse.Namespace(label_schema=ls.V2,research_config=None,synth=False,
                    calibration_roles='never-read-roles.json',data_dir=data)
                with mock.patch.object(te,'read_jsonl',side_effect=AssertionError('NO DATA')) as reader,mock.patch.object(te,'ensure_optimizer_available',side_effect=AssertionError('NO OPTIMIZER')) as optimizer,mock.patch.object(rt,'sample_epochs',side_effect=AssertionError('NO RESEARCH SAMPLER')) as sampler,mock.patch.object(te,'oversample_epoch_rows',side_effect=AssertionError('NO LEGACY SAMPLER')) as legacy_sampler,mock.patch.object(te,'load_engine',side_effect=AssertionError('NO MODEL')) as model,mock.patch.object(Path,'mkdir',side_effect=AssertionError('NO OUTPUT')):
                    with self.assertRaisesRegex(ValueError,'both schema metadata'):te.run(args)
                    for sentinel in (reader,optimizer,sampler,legacy_sampler,model):sentinel.assert_not_called()
                self.assertFalse((root/'artifacts').exists())

    def test_prediction_score_cannot_replace_identity_source_or_train_only_provenance(self):
        te=f.import_train_eval_with_stubs()
        for selected in (ls.LEGACY,ls.V2):
            with self.subTest(selected=selected):
                row=source_rows()[0]
                row.update(prompt='INVENTED CPU-only provenance fixture',
                    task_label_schema_version=selected,synthetic=True,supplemental_only=True,
                    partition='synthetic_train_only',provenance={'invented':True,'origin':'CPU-only'})
                row['label'].update(task_label_schema_version=selected,label_status='resolved')
                before=copy.deepcopy(row);answers={}
                for head,question in ls.questions_for(selected,questions_from_source()).items():
                    labels=list(question['criteria']);choice=labels[0]
                    answers[head]={'choice':choice,'confidence':1.,'probabilities':{c:float(c==choice) for c in labels},'logits':[float(c==choice) for c in labels]}
                injected={'answers':answers,'task_label_schema_version':selected,'id':'REPLACED',
                    'label':{'tier':'flagship'},'split_group':'REPLACED','group':'REPLACED',
                    'label_schema_version':1,'synthetic':False,'supplemental_only':False,
                    'partition':'primary','provenance':{'invented':False}}
                with mock.patch.object(te,'score',return_value=injected) as scorer:
                    result=te.predict_rows(object(),[row],selected)[0]
                scorer.assert_called_once_with(mock.ANY,row['prompt'],schema_version=selected)
                self.assertEqual(row,before);self.assertNotIn('prompt',result)
                for key,value in row.items():
                    if key!='prompt':self.assertEqual(result[key],value)
                if selected==ls.V2:
                    from evaluation_metrics import metrics
                    with self.assertRaisesRegex(ValueError,'supplemental'):metrics([result],schema_version=selected)


if __name__=='__main__':unittest.main()
