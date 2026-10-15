"""Synthetic contract tests only: no real executions, prompts or model assets."""
import contextlib, copy, io, json, os, random, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from label_schema import unify
from outcomes import import_outcomes
from prepare import LEGACY_OUTPUTS, UNIFIED_OUTPUTS, load, prepare
from synthetic_data import write_rows


def fixture():
    rng=random.Random(10)
    tasks=[{'id':f'schema-{i}','group':f'group-{i//10}',
            'prompt':''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=100)),'synthetic':True} for i in range(120)]
    experts=[{'id':t['id'],'tier':('fast','balanced','flagship')[i%3],
              'needs_vision':('yes','no')[i%2],'task_type':'implementation',
              'label_source':'expert_provisional','evidence':'prompt_only','confidence':.6,
              'rationale':'Synthetic expert annotation, not execution evidence.'} for i,t in enumerate(tasks)]
    routes=[{'provider':'synthetic-provider','model':'synthetic-model','tier':'flagship',
             'vision':True,'model_version':'synthetic-v1','reasoning_effort':'low'}]
    trials=[{'task_id':tasks[0]['id'],'provider':routes[0]['provider'],'model':routes[0]['model'],
             'trial':i,'passed':True,'evidence_kind':'acceptance_tests','acceptance_spec_sha256':'a'*64,
             'artifact_sha256':'b'*64,'execution_id':f'synthetic-{i}','context_sha256':'c'*64,
             'timeout':False,'wall_time':1.,'cost':1.,'latency_ms':2.,'input_tokens':3,'output_tokens':4,
             'model_version':'synthetic-v1','reasoning_effort':'low'} for i in range(29)]
    # Deliberately synthetic mock of the importer contract, NOT measured supply.
    envelope=import_outcomes(tasks,trials,routes)
    return tasks,experts,routes,envelope


def write_inputs(root,tasks,experts,envelope,routes):
    write_rows(root/'tasks.jsonl',tasks)
    for n in range(1,5):write_rows(root/f'expert-recovery-{n}.jsonl',experts[(n-1)*30:n*30])
    (root/'mock-outcomes.json').write_text(json.dumps(envelope))
    (root/'mock-routes.json').write_text(json.dumps(routes))


def snapshot(root):
    return {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()}


class LabelSchemaTests(unittest.TestCase):
    def setUp(self):self.tasks,self.experts,self.routes,self.envelope=fixture()

    def test_mixed_prepare_preserves_expert_splits_and_complete_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy_root=Path(directory)/'legacy';legacy_root.mkdir()
            root=Path(directory)/'mixed';root.mkdir()
            for path in (legacy_root,root):write_inputs(path,self.tasks,self.experts,self.envelope,self.routes)
            with contextlib.redirect_stdout(io.StringIO()):prepare(legacy_root)
            legacy={name:(legacy_root/f'{name}.jsonl').read_bytes() for name in ('train','calibration','test','expert-labels')}
            with contextlib.redirect_stdout(io.StringIO()):prepare(root,outcomes=self.envelope,routes=self.routes)
            self.assertEqual(legacy,{name:(root/f'{name}.jsonl').read_bytes() for name in legacy})
            rows=load(root/'unified-labels.jsonl');shadow=load(root/'measured-shadow.jsonl')
            self.assertEqual(len(rows),120);self.assertEqual(len(shadow),1)
            self.assertTrue(all(r['candidate_success_probabilities'] is None for r in rows))
            self.assertEqual(shadow[0]['label'],self.envelope['labels'][0])
            self.assertEqual(shadow[0]['task_understanding'],self.experts[0])
            self.assertEqual(shadow[0]['label']['tier'],'flagship')
            self.assertEqual(shadow[0]['task_understanding']['tier'],'fast')
            self.assertEqual(shadow[0]['candidate_metrics'],self.envelope['candidate_metrics'])
            self.assertEqual(shadow[0]['trial_details'],self.envelope['trial_details'])
            self.assertEqual(shadow[0]['outcome_metadata']['preregistration'],self.envelope['preregistration'])
            self.assertIn('not_sample_weight',shadow[0]['confidence_semantics'])
            self.assertEqual(json.loads((root/'measured-evidence.json').read_text()),self.envelope)
            for name in ('train','calibration','test'):
                self.assertTrue(all(r['label']['label_source']=='expert_provisional' for r in load(root/f'{name}.jsonl')))
            self.assertIn(shadow[0]['expert_split'],('train','calibration','test'))

    def test_fresh_expert_only_unified_bundle_has_no_stale_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_inputs(root,self.tasks,self.experts,self.envelope,self.routes)
            with contextlib.redirect_stdout(io.StringIO()):prepare(root,unified=True)
            self.assertEqual(load(root/'measured-shadow.jsonl'),[])
            self.assertFalse((root/'measured-evidence.json').exists())
            self.assertEqual(json.loads((root/'split-report.json').read_text())['measured_shadow_rows'],0)
            self.assertTrue(all(r['measured_candidate_outcomes'] is None for r in load(root/'unified-labels.jsonl')))

    def test_finite_arbitrary_envelope_metadata_roundtrips_without_mutation(self):
        envelope=copy.deepcopy(self.envelope);envelope['note']={'arbitrary':[None,False,1.25,'有限审计']}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_inputs(root,self.tasks,self.experts,envelope,self.routes)
            before=copy.deepcopy(envelope)
            with contextlib.redirect_stdout(io.StringIO()):prepare(root,outcomes=envelope,routes=self.routes)
            self.assertEqual(json.loads((root/'measured-evidence.json').read_text()),before)
            self.assertEqual(envelope,before)

    def test_existing_unified_bundle_refuses_all_reruns_without_touching_outputs(self):
        # F1: ordinary expert-only replacement used to leave unrelated shadow
        # tasks; --unified expert-only used to empty shadow but retain evidence.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_inputs(root,self.tasks,self.experts,self.envelope,self.routes)
            with contextlib.redirect_stdout(io.StringIO()):prepare(root,outcomes=self.envelope,routes=self.routes)
            for replacement in (False,True):
                if replacement:
                    tasks=copy.deepcopy(self.tasks);experts=copy.deepcopy(self.experts)
                    for task,expert in zip(tasks,experts):task['id']=expert['id']='replacement-'+task['id']
                    write_inputs(root,tasks,experts,self.envelope,self.routes)
                before=snapshot(root)
                modes=[{}, {'unified':True}]
                if not replacement:modes.append({'outcomes':self.envelope,'routes':self.routes})
                for kwargs in modes:
                    for preflight in (False,True):
                        with self.subTest(replacement=replacement,kwargs=tuple(kwargs),preflight=preflight):
                            with self.assertRaisesRegex(ValueError,'existing prepare output'):
                                prepare(root,preflight=preflight,**kwargs)
                            self.assertEqual(before,snapshot(root))

    def test_unified_refuses_each_existing_output_including_legacy_migration(self):
        # No ownership assumptions: even an unknown/partial single file blocks
        # a unified bundle. Orphaned unified files block legacy mode as well.
        for name in (*LEGACY_OUTPUTS,*UNIFIED_OUTPUTS):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);write_inputs(root,self.tasks,self.experts,self.envelope,self.routes)
                (root/name).write_bytes(b'unknown user data\n')
                before=snapshot(root)
                for preflight in (False,True):
                    for kwargs in ({'unified':True},{'outcomes':self.envelope,'routes':self.routes}):
                        with self.assertRaisesRegex(ValueError,'existing prepare output'):
                            prepare(root,preflight=preflight,**kwargs)
                        self.assertEqual(before,snapshot(root))
                    if name in UNIFIED_OUTPUTS:
                        with self.assertRaisesRegex(ValueError,'existing prepare output'):prepare(root,preflight=preflight)
                        self.assertEqual(before,snapshot(root))

    def test_output_symlinks_and_directories_are_refused_before_writes(self):
        for name in ('train.jsonl','measured-evidence.json'):
            for kind in ('directory','dangling_symlink'):
                with self.subTest(name=name,kind=kind),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);write_inputs(root,self.tasks,self.experts,self.envelope,self.routes)
                    path=root/name
                    if kind=='directory':path.mkdir()
                    else:path.symlink_to(root/'missing-user-file')
                    before={p.name:p.read_bytes() for p in root.iterdir() if p!=path}
                    for kwargs in ({},{'unified':True}):
                        with self.assertRaises(ValueError):prepare(root,**kwargs)
                        self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir() if p!=path})
                        self.assertFalse((root/'missing-user-file').exists())
                        self.assertTrue(path.is_dir() if kind=='directory' else path.is_symlink())

    def test_legacy_reruns_keep_historical_five_output_overwrite_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_inputs(root,self.tasks,self.experts,self.envelope,self.routes)
            with contextlib.redirect_stdout(io.StringIO()):prepare(root)
            expected={name:(root/name).read_bytes() for name in LEGACY_OUTPUTS}
            with contextlib.redirect_stdout(io.StringIO()):prepare(root)
            self.assertEqual(expected,{name:(root/name).read_bytes() for name in LEGACY_OUTPUTS})
            for name in UNIFIED_OUTPUTS:self.assertFalse((root/name).exists())
            # The legacy-only contract intentionally still allows replacement.
            experts=copy.deepcopy(self.experts);experts[0]['tier']='balanced'
            write_inputs(root,self.tasks,experts,self.envelope,self.routes)
            with contextlib.redirect_stdout(io.StringIO()):prepare(root)
            self.assertEqual(next(r for r in load(root/'expert-labels.jsonl') if r['id']==experts[0]['id'])['tier'],'balanced')

    def test_nonfinite_note_cli_rejects_preflight_and_prepare_with_existing_bundle_untouched(self):
        # Exact F2: decoder accepts NaN in an otherwise ignored envelope note.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_inputs(root,self.tasks,self.experts,self.envelope,self.routes)
            with contextlib.redirect_stdout(io.StringIO()):prepare(root,outcomes=self.envelope,routes=self.routes)
            experts=copy.deepcopy(self.experts);experts[0]['tier']='balanced'
            bad=copy.deepcopy(self.envelope);bad['note']=float('nan')
            write_inputs(root,self.tasks,experts,bad,self.routes)
            before=snapshot(root)
            for flags in (['--preflight'],[]):
                result=subprocess.run([sys.executable,'-S',str(ROOT/'prepare.py'),'--data',str(root),
                    '--outcomes',str(root/'mock-outcomes.json'),'--routes',str(root/'mock-routes.json'),*flags],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'},capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('measured evidence must be finite UTF-8 JSON',result.stderr)
                self.assertEqual(before,snapshot(root))

    def test_nested_invalid_metadata_rejected_for_all_sources_before_outputs(self):
        for source in ('task','expert','envelope','route'):
            for value in (float('nan'),float('inf'),-float('inf'),'\ud800'):
                with self.subTest(source=source,value=repr(value)),tempfile.TemporaryDirectory() as directory:
                    tasks,experts,routes,envelope=copy.deepcopy((self.tasks,self.experts,self.routes,self.envelope))
                    target={'task':tasks[0],'expert':experts[0],'envelope':envelope,'route':routes[0]}[source]
                    target['ignored_metadata']={'nested':[{'value':value}]}
                    root=Path(directory);write_inputs(root,tasks,experts,envelope,routes)
                    for existing in (False,True):
                        # Also prove refusal on a fresh directory (no conflicts)
                        # and no truncation/mtime changes on existing outputs.
                        if existing:
                            for name in LEGACY_OUTPUTS:(root/name).write_bytes(b'old corpus\n')
                        before=snapshot(root)
                        for preflight in (False,True):
                            for kwargs in ({'outcomes':envelope,'routes':routes},*([{}] if source in ('task','expert') else [])):
                                with self.assertRaisesRegex(ValueError,'finite UTF-8 JSON'):
                                    prepare(root,preflight=preflight,**kwargs)
                                self.assertEqual(before,snapshot(root))

    def test_unserializable_and_cyclic_api_metadata_rejected(self):
        cycle=[];cycle.append(cycle)
        for value in ({1,2},object(),complex(1,2),cycle):
            for source in ('task','expert','envelope','route'):
                tasks,experts,routes,envelope=copy.deepcopy((self.tasks,self.experts,self.routes,self.envelope))
                target={'task':tasks[0],'expert':experts[0],'envelope':envelope,'route':routes[0]}[source]
                target['ignored_metadata']={'nested':[value]}
                with self.subTest(source=source,value=type(value).__name__),self.assertRaisesRegex(ValueError,'finite UTF-8 JSON'):
                    unify(tasks,experts,envelope,routes)

    def test_preflight_cli_without_site_packages_or_torch_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_inputs(root,self.tasks,self.experts,self.envelope,self.routes)
            before={p.name:p.read_bytes() for p in root.iterdir()}
            result=subprocess.run([sys.executable,'-S',str(ROOT/'prepare.py'),'--data',str(root),
                '--outcomes',str(root/'mock-outcomes.json'),'--routes',str(root/'mock-routes.json'),'--preflight'],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'},text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['measured_shadow_rows'],1)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            # Stronger than absence of site-packages: explicitly prohibit imports.
            code="""import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','transformers','peft','model_utils'):raise RuntimeError('forbidden import '+name)
sys.meta_path.insert(0,Block())
from prepare import prepare
from pathlib import Path
import json
p=Path(sys.argv[1]);prepare(p,outcomes=json.loads((p/'mock-outcomes.json').read_text()),routes=json.loads((p/'mock-routes.json').read_text()),preflight=True)
"""
            blocked=subprocess.run([sys.executable,'-S','-c',code,str(root)],cwd=ROOT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'},capture_output=True,text=True)
            self.assertEqual(blocked.returncode,0,blocked.stderr)

    def test_invalid_prepare_inputs_fail_before_any_output_and_null_cli_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);envelope=copy.deepcopy(self.envelope);envelope.pop('preregistration')
            write_inputs(root,self.tasks,self.experts,envelope,self.routes)
            before={p.name:p.read_bytes() for p in root.iterdir()}
            with self.assertRaises(ValueError):prepare(root,outcomes=envelope,routes=self.routes)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            (root/'mock-outcomes.json').write_text('null')
            result=subprocess.run([sys.executable,'-S',str(ROOT/'prepare.py'),'--data',str(root),
                '--outcomes',str(root/'mock-outcomes.json'),'--routes',str(root/'mock-routes.json'),'--preflight'],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'},capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('full v2 audit object',result.stderr)
            self.assertEqual(set(before),{p.name for p in root.iterdir()})

    def test_expert_only_unified_has_empty_shadow_and_nullable_future_head(self):
        unified,shadow=unify(self.tasks,self.experts)
        self.assertFalse(shadow)
        self.assertTrue(all(r['measured_candidate_outcomes'] is None and r['candidate_success_probabilities'] is None for r in unified))

    def test_missing_and_conflicting_expert_sources(self):
        for change in ({'label_source':None},{'label_source':'measured_candidate_outcomes'},
                       {'evidence':'acceptance_tests'},{'provider':'leaked-candidate'}):
            experts=copy.deepcopy(self.experts);experts[0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):unify(self.tasks,experts)

    def test_missing_understanding_is_not_inferred_from_measured(self):
        for key in ('tier','task_type','needs_vision','rationale'):
            experts=copy.deepcopy(self.experts);del experts[0][key]
            with self.subTest(key=key),self.assertRaises(ValueError):unify(self.tasks,experts,self.envelope,self.routes)
        with self.assertRaises(ValueError):unify(self.tasks,self.experts[1:],self.envelope,self.routes)

    def test_duplicate_task_expert_and_measured_ids(self):
        for tasks,experts,envelope in ((self.tasks+[self.tasks[0]],self.experts,self.envelope),
                (self.tasks,self.experts+[self.experts[0]],self.envelope)):
            with self.assertRaisesRegex(ValueError,'duplicate'):unify(tasks,experts,envelope,self.routes)
        envelope=copy.deepcopy(self.envelope);envelope['labels']*=2
        with self.assertRaisesRegex(ValueError,'duplicate'):unify(self.tasks,self.experts,envelope,self.routes)

    def test_reject_ungated_missing_or_tampered_audit(self):
        changes=[lambda e:e.pop('preregistration'),lambda e:e.pop('trial_details'),
                 lambda e:e.update(trial_details=[]),lambda e:e['labels'][0].update(success_lb=1.),
                 lambda e:e['labels'][0].update(label_source=None),
                 lambda e:e['labels'][0].update(evidence='prompt_only'),
                 lambda e:e['labels'][0].update(needs_vision='yes'),
                 lambda e:e['labels'][0].update(route_fingerprint='0'*16),
                 lambda e:e['preregistration'].update(routes_fingerprint='0'*64),
                 lambda e:e['trial_details'][0].update(passed=False),
                 lambda e:e['candidate_metrics'][0].update(trials=1),
                 lambda e:e.update(simulated=True),lambda e:e.update(minimum_success_lb=.99)]
        for change in changes:
            envelope=copy.deepcopy(self.envelope);change(envelope)
            with self.subTest(change=change),self.assertRaises(ValueError):unify(self.tasks,self.experts,envelope,self.routes)
        with self.assertRaises(ValueError):unify(self.tasks,self.experts,self.envelope['labels'],self.routes)
        with self.assertRaises(ValueError):unify(self.tasks,self.experts,self.envelope,None)

    def test_invalid_expert_confidence_rejected(self):
        for confidence in (True,float('nan'),float('inf'),-1,1.1,None):
            experts=copy.deepcopy(self.experts);experts[0]['confidence']=confidence
            with self.subTest(confidence=confidence),self.assertRaises(ValueError):unify(self.tasks,experts)

    def test_gate_confidence_and_expert_confidence_are_separate_and_inputs_unchanged(self):
        before=copy.deepcopy((self.tasks,self.experts,self.envelope,self.routes))
        unified,shadow=unify(self.tasks,self.experts,self.envelope,self.routes)
        self.assertEqual(unified[0]['task_understanding']['confidence'],.6)
        self.assertEqual(shadow[0]['label']['confidence'],.95)
        self.assertEqual(before,(self.tasks,self.experts,self.envelope,self.routes))
        shadow[0]['trial_details'].clear()
        self.assertEqual(before,(self.tasks,self.experts,self.envelope,self.routes))


if __name__=='__main__':unittest.main()
