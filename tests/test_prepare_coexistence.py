"""Synthetic #8/#24 CLI coexistence; no real evidence, weights, or torch imports."""
import contextlib, hashlib, io, itertools, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import prepare as module
import split_v2
from test_label_schema import fixture, snapshot, write_inputs

# Oracle: historical prepare() at reviewed main 6b6f1de, on fixture() below
# with UTF-8 metadata/rationale. No Git or archived/private files needed at runtime.
LEGACY_HASHES={
    '20261010':{
        'train.jsonl':'298a52d03f047a17f851d703466563607f1cb1cd292ebae25e89f45211947cf6',
        'calibration.jsonl':'9a9184738312b736ce22f67803ed33ab84c629bf6d827bd9df5648603b540e0f',
        'test.jsonl':'4142d179fb260e901664a744ff4b8ff418bfc230ca0c1f686e0f519c6d8693a9',
        'split-report.json':'646b90abfd6809d95bec28a0e6bc3ff3d8c5142e767ead34f87afa95bdc52f35',
        'expert-labels.jsonl':'0d20ee11b04d51c3cdf4cd7357a95c24dce74e97950b4380db23a8620ce81272'},
    '42':{
        'train.jsonl':'fd2b80f0dc2e08c3f748ac0561273c988eda03a62acfd5ef330860bf2e58f8d6',
        'calibration.jsonl':'0876c6ba5fbd3f8daeb99928c33d4c1cbf1eac3d97be29b7b92d801414fbda6d',
        'test.jsonl':'17f5549eb6e4f932a530e8ae10c14a1ea766fdd363a146152cc41c20bc756e7c',
        'split-report.json':'d57fd7501f5ca59bb83c5e33c7b2adf425fdfba18c200a2a5bf25d0da77e173c',
        'expert-labels.jsonl':'0d20ee11b04d51c3cdf4cd7357a95c24dce74e97950b4380db23a8620ce81272'}}

BLOCKED_CLI="""import runpy, sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','transformers','peft','model_utils'):
   raise RuntimeError('forbidden import '+name)
sys.meta_path.insert(0,Block())
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""


class PrepareCoexistenceTests(unittest.TestCase):
    def setUp(self):
        self.tasks,self.experts,self.routes,self.envelope=fixture()
        self.tasks[0]['metadata']={'note':'合成 UTF-8 only'}
        self.experts[0]['rationale']='合成专家注释；不是执行证据。'

    def inputs(self,root):write_inputs(root,self.tasks,self.experts,self.envelope,self.routes)

    def cli(self,*args):
        return subprocess.run([sys.executable,'-S','-c',BLOCKED_CLI,str(ROOT/'prepare.py'),*map(str,args)],
            cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void',
                          'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'},
            capture_output=True,text=True)

    def measured_flags(self,root):return ['--outcomes',root/'mock-outcomes.json','--routes',root/'mock-routes.json']

    def test_default_dispatch_and_split_v2_exports_stay_separate(self):
        self.assertIs(module.prepare_v2,split_v2.prepare_v2)
        self.assertIs(module.split_v2,split_v2.split_v2)
        for flags in ([],['--split-version','frozen-v1']):
            with patch.object(module,'prepare') as legacy,patch.object(module,'prepare_v2') as v2:
                self.assertEqual(module.main(flags),0)
                legacy.assert_called_once_with(ROOT/'data',20261010,outcomes=None,routes=None,unified=False,preflight=False)
                v2.assert_not_called()
        with patch.object(module,'prepare') as legacy,patch.object(module,'prepare_v2',return_value={'status':'feasible'}) as v2:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(['--split-version','split-v2','--data','synthetic-input','--output','new','--seed','42']),0)
            v2.assert_called_once_with(Path('synthetic-input'),Path('new'),seed=42)
            legacy.assert_not_called()

    def test_legacy_cli_bytes_match_historical_main_for_default_and_explicit_modes(self):
        for seed,expected in LEGACY_HASHES.items():
            for version in ([],['--split-version','frozen-v1']):
                with self.subTest(seed=seed,version=version),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);self.inputs(root)
                    flags=['--data',root,*version]
                    if seed!='20261010':flags+=['--seed',seed]
                    for _ in range(2):  # Historical five-file overwrite/rerun behavior.
                        result=self.cli(*flags)
                        self.assertEqual(result.returncode,0,result.stderr)
                        self.assertEqual(expected,{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in module.LEGACY_OUTPUTS})
                        self.assertEqual(json.loads(result.stdout),json.loads((root/'split-report.json').read_text()))
                        self.assertFalse(any((root/name).exists() for name in module.UNIFIED_OUTPUTS))

    def test_preflight_flag_matrix_is_stdlib_only_and_write_free(self):
        for version in ([],['--split-version','frozen-v1']):
            for mode in ('expert','unified','measured','both'):
                with self.subTest(version=version,mode=mode),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);self.inputs(root);before=snapshot(root)
                    flags=['--unified'] if mode in ('unified','both') else []
                    if mode in ('measured','both'):flags+=self.measured_flags(root)
                    result=self.cli('--data',root,*version,*flags,'--preflight')
                    self.assertEqual(result.returncode,0,result.stderr)
                    summary=json.loads(result.stdout)
                    self.assertEqual(summary,{'status':'structurally_valid','tasks':120,'expert_training_rows':120,
                        'measured_shadow_rows':int(mode in ('measured','both')),'models_loaded':False,'outputs_written':False})
                    self.assertEqual(snapshot(root),before)

    def test_explicit_frozen_unified_and_measured_cli_preserve_expert_training(self):
        for measured in (False,True):
            with self.subTest(measured=measured),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.inputs(root)
                result=self.cli('--data',root,'--split-version','frozen-v1','--unified',
                                *(self.measured_flags(root) if measured else []))
                self.assertEqual(result.returncode,0,result.stderr)
                for name in ('train.jsonl','calibration.jsonl','test.jsonl','expert-labels.jsonl'):
                    self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),LEGACY_HASHES['20261010'][name])
                rows=module.load(root/'unified-labels.jsonl');shadow=module.load(root/'measured-shadow.jsonl')
                self.assertEqual(len(rows),120);self.assertEqual(len(shadow),int(measured))
                self.assertTrue(all(r['cold_start_training_source']=='expert_provisional' for r in rows))
                if measured:
                    self.assertEqual(shadow[0]['label'],self.envelope['labels'][0])
                    self.assertEqual(shadow[0]['task_understanding'],self.experts[0])
                    self.assertEqual(shadow[0]['outcome_metadata']['preregistration'],self.envelope['preregistration'])
                    self.assertEqual(json.loads((root/'measured-evidence.json').read_text()),self.envelope)
                else:self.assertFalse((root/'measured-evidence.json').exists())
                before=snapshot(root)
                for flags in ([],['--preflight'],['--unified'],self.measured_flags(root)):
                    rejected=self.cli('--data',root,'--split-version','frozen-v1',*flags)
                    self.assertNotEqual(rejected.returncode,0)
                    self.assertIn('existing prepare output',rejected.stderr)
                    self.assertEqual(before,snapshot(root))

    def test_split_v2_rejects_every_nonempty_unified_flag_subset_before_reads_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'sentinel').write_bytes(b'preserve user file\n')
            output=root/'never-created';missing=root/'missing-input'
            flag_groups=[['--unified'],['--outcomes',root/'missing-outcomes.json'],
                         ['--routes',root/'missing-routes.json'],['--preflight']]
            before=snapshot(root)
            for count in range(1,5):
                for groups in itertools.combinations(flag_groups,count):
                    flags=[value for group in groups for value in group]
                    with self.subTest(flags=flags):
                        result=self.cli('--split-version','split-v2','--data',missing,'--output',output,*flags)
                        self.assertEqual(result.returncode,2,result.stderr)
                        self.assertIn('split-v2 does not support',result.stderr)
                        for group in groups:self.assertIn(group[0],result.stderr.split('error: ')[-1])
                        self.assertNotIn('Traceback',result.stderr)
                        self.assertFalse(output.exists());self.assertEqual(snapshot(root),before)

    def test_valid_measured_evidence_is_not_silently_ignored_by_split_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root);before=snapshot(root)
            output=root/'never-created'
            result=self.cli('--split-version','split-v2','--data',root,'--output',output,*self.measured_flags(root))
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('does not support --outcomes, --routes',result.stderr)
            self.assertFalse(output.exists());self.assertEqual(snapshot(root),before)

    def test_invalid_frozen_flag_combinations_fail_before_reads_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);before=snapshot(root);missing=root/'missing-input'
            for extras,message in [(['--outcomes',root/'missing-outcomes'],'--outcomes requires --routes'),
                                   (['--routes',root/'missing-routes'],'--routes requires --outcomes'),
                                   (['--output',root/'never-created'],'--output requires --split-version split-v2')]:
                for mode in ([],['--unified'],['--preflight'],['--unified','--preflight']):
                    with self.subTest(extras=extras,mode=mode):
                        result=self.cli('--split-version','frozen-v1','--data',missing,*extras,*mode)
                        self.assertEqual(result.returncode,2,result.stderr)
                        self.assertIn(message,result.stderr);self.assertNotIn('Traceback',result.stderr)
                        self.assertEqual(before,snapshot(root))

    def test_split_v2_can_read_experts_but_never_mutates_or_migrates_unified_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'inputs';root.mkdir();self.inputs(root)
            with contextlib.redirect_stdout(io.StringIO()):module.prepare(root,outcomes=self.envelope,routes=self.routes)
            before=snapshot(root);output=Path(directory)/'v2'
            result=self.cli('--split-version','split-v2','--data',root,'--output',output)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(before,snapshot(root))
            report=json.loads(result.stdout);self.assertEqual(report['targets']['max_group_share'],.35)
            self.assertEqual({p.name for p in output.iterdir()},
                {'train.split-v2.jsonl','calibration.split-v2.jsonl','test.split-v2.jsonl',
                 'expert-labels.split-v2.jsonl','split-report.split-v2.json'})
            for name,stats in report['splits'].items():
                self.assertTrue(stats['constraint_satisfied']);self.assertLessEqual(stats['max_group_share'],.35)
                self.assertTrue(all(r['label']['label_source']=='expert_provisional' for r in module.load(output/f'{name}.split-v2.jsonl')))
            written=snapshot(output)
            rejected=self.cli('--split-version','split-v2','--data',root,'--output',output)
            self.assertNotEqual(rejected.returncode,0);self.assertIn('already exists',rejected.stderr)
            self.assertEqual(written,snapshot(output));self.assertEqual(before,snapshot(root))

    def test_split_v2_python_apis_reject_unified_keywords_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root);before=snapshot(root);output=root/'never-created'
            for key,value in (('unified',True),('preflight',True),('outcomes',self.envelope),('routes',self.routes)):
                with self.subTest(key=key):
                    with self.assertRaisesRegex(TypeError,'unexpected keyword'):
                        module.prepare_v2(root,output,**{key:value})
                    with self.assertRaisesRegex(TypeError,'unexpected keyword'):
                        module.split_v2(self.tasks,self.experts,**{key:value})
                    self.assertFalse(output.exists());self.assertEqual(before,snapshot(root))


if __name__=='__main__':unittest.main()
