"""Synthetic CPU development tests; NEVER load/forward/train a backbone.

Torch tensor cases are optional; stdlib config/sampler/launcher tests always run.
"""
import argparse, copy, importlib.util, json, math, subprocess, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research_training as rt
import research_capacity as rc
from calibration_groups import questions_from_source

try:import torch
except ImportError:torch=None


def config():return rt.load_config(ROOT/'configs/research-v1.json')


def rows():
    return [{'id':f'toy-{i}','split_group':'large' if i<4 else f'small-{i}',
             'label':{'tier':rt.TIERS[i%3],'needs_vision':rt.VISION[i%2],
                      'task_type':'creative','label_source':'expert_provisional','confidence':.8}}
            for i in range(6)]


def fixtures():
    spec=importlib.util.spec_from_file_location('research_calibration_fixtures',ROOT/'tests/test_calibration_groups.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ConfigTests(unittest.TestCase):
    def test_imports_are_stdlib_only(self):
        code="import research_training,research_capacity,sys; assert 'torch' not in sys.modules; assert 'peft' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_config_hash_and_frozen_recipe_are_separate(self):
        c=config();self.assertEqual(len(rt.config_hash(c)),64)
        self.assertNotEqual(c['training']['rank'],json.loads((ROOT/'configs/v3c.json').read_text())['rank'])
        c['sampler']['epoch_rows']+=1
        self.assertNotEqual(rt.config_hash(c),rt.config_hash(config()))

    def test_bad_config_fails_closed(self):
        mutations=[lambda c:c.update(version='v2'),lambda c:c.update(schema_version='six-type'),
                   lambda c:c.update(unknown=1),lambda c:c['sampler'].update(epoch_rows=True),
                   lambda c:c['sampler'].update(prior_source='test'),lambda c:c['sampler'].update(prior_smoothing=-1),
                   lambda c:c['loss'].update(tier_tau=float('nan')),lambda c:c['loss'].update(auxiliary='independent-head'),
                   lambda c:c['loss']['cost_matrix'][2].__setitem__(0,0),lambda c:c['capacity'].update(interpolation=.65),
                   lambda c:c['training'].update(swa=True),lambda c:c['training'].update(synth=True),
                   lambda c:c['training'].update(rank=8),lambda c:c['training'].update(fullft=True)]
        for change in mutations:
            c=config();change(c)
            with self.subTest(c=c),self.assertRaises(ValueError):rt.validate_config(c)

    def test_staged8bit_config_is_distinct_control(self):
        c=config();c['capacity']['mode']='staged8bit'
        with self.assertRaises(ValueError):rt.validate_config(c)
        c['training'].update(fullft=True,optimizer='adamw8bit',epochs=2)
        rt.validate_config(c)
        c['training']['epochs']=3
        with self.assertRaises(ValueError):rt.validate_config(c)

    def test_cli_mismatch_legacy_and_smoke_fail_before_model(self):
        c=config();args=argparse.Namespace(**c['training'],legacy_calibration=False,smoke=False)
        rt.check_args(c,args)
        for key,value in [('epochs',3),('legacy_calibration',True),('smoke',True)]:
            bad=copy.copy(args);setattr(bad,key,value)
            with self.assertRaises(ValueError):rt.check_args(c,bad)
        te=fixtures().import_train_eval_with_stubs()
        with mock.patch.object(te,'load_engine',side_effect=AssertionError('must not load')):
            args.research_config=str(ROOT/'configs/research-v1.json');args.smoke=True
            with self.assertRaises(ValueError):te.run(args)

    def test_explicit_schema_boundary(self):
        q=questions_from_source();rt.check_questions(q)
        q['task_type']['criteria'].pop('visual')
        with self.assertRaises(ValueError):rt.check_questions(q)

    def test_auxiliary_coefficients_are_explicit_and_independent_is_unavailable(self):
        c=config();self.assertEqual(rt.head_weights(c),{'tier':1.,'vision':.5,'task_type':.15})
        for mode,vision,task in [('tier-only',0,0),('task-type-zero',.5,0),('current-03',.5,.3)]:
            c['loss']['auxiliary']=mode
            self.assertEqual(rt.head_weights(c),{'tier':1.,'vision':vision,'task_type':task})


class SamplingTests(unittest.TestCase):
    def test_sqrt_group_distribution_is_not_tier_balancing(self):
        plan=rt.sampler_plan(rows(),config())
        self.assertEqual(plan['natural_row_probabilities'],[1/6]*6)
        self.assertEqual(plan['target_row_probabilities'],[.125]*4+[.25]*2)
        self.assertEqual(plan['group_count'],3)

    def test_synthetic_exposure_is_probability_only_and_prior_excluded(self):
        c=config();r=rows();r.append({'id':'synthetic','split_group':'synth','synthetic':True,
            'label':dict(r[1]['label'],tier='balanced',needs_vision='yes')})
        p=rt.sampler_plan(r,c)
        self.assertAlmostEqual(p['target_row_probabilities'][-1]/p['target_row_probabilities'][-2],1.5)
        self.assertEqual(p['priors'],rt.sampler_plan(rows(),c)['priors'])
        self.assertEqual(p['prior_provenance']['unique_nonsynthetic_expert_support']['tier'],dict.fromkeys(rt.TIERS,2))

    def test_expected_sampler_prior_reports_changed_distribution(self):
        c=config();c['sampler'].update(prior_source='expected-sampler',prior_smoothing=0.)
        r=rows();r[0]['label']['tier']='flagship'
        p=rt.sampler_plan(r,c)
        for tier in rt.TIERS:
            expected=sum(v for v,row in zip(p['target_row_probabilities'],r) if row['label']['tier']==tier)
            self.assertAlmostEqual(p['priors']['tier'][tier],expected)
        self.assertTrue(p['prior_provenance']['expected_sampler_includes_synthetic'])

    def test_seed_budget_and_empirical_probability_accounting(self):
        c=config();c['sampler']['epoch_rows']=12000
        streams,p=rt.sample_epochs(rows(),c,7,2)
        self.assertEqual([len(s) for s in streams],[12000,12000])
        self.assertEqual(streams,rt.sample_epochs(rows(),c,7,2)[0])
        self.assertNotEqual(streams[0],streams[1])
        for observed,target in zip(p['actual_epoch_row_frequencies'][0],p['target_row_probabilities']):
            self.assertLess(abs(observed-target),.015)
        self.assertAlmostEqual(sum(p['actual_epoch_row_frequencies'][0]),1.)

    def test_sampling_refuses_missing_group_duplicate_measured_and_unsupported_prior(self):
        mutations=[lambda r:r[0].pop('split_group'),lambda r:r.append(r[0]),
                   lambda r:r[0]['label'].update(label_source='measured'),
                   lambda r:r[0]['label'].update(needs_vision='uncertain'),
                   lambda r:[x['label'].update(tier='balanced') for x in r],
                   lambda r:r[0].update(synthetic='false')]
        for change in mutations:
            r=rows();change(r)
            with self.assertRaises(ValueError):rt.sampler_plan(r,config())

    def test_effective_scheduler_uses_sample_budget_and_tail_ceil(self):
        c=config();c['sampler']['epoch_rows']=17
        streams,_=rt.sample_epochs(rows(),c,7,2);te=fixtures().import_train_eval_with_stubs()
        self.assertEqual(sum(te.optimizer_updates(len(s),8) for s in streams),6)
        # Same loop rule as integrated training, never raw unique-row length.
        self.assertEqual(sum(sum(i%8==0 or i==len(s) for i in range(1,len(s)+1)) for s in streams),6)


class NumericTests(unittest.TestCase):
    def test_training_bias_is_plus_log_prior_not_inference_subtraction(self):
        bias=rt.additive_bias({'a':.8,'b':.2},['a','b'],1.)
        self.assertAlmostEqual(math.exp(bias[0])/(math.exp(bias[0])+math.exp(bias[1])),.8)
        self.assertGreater(bias[0],bias[1])
        with self.assertRaises(ValueError):rt.additive_bias({'a':1.,'b':0.},['a','b'],1.)

    def test_cost_matrix_orientation_and_complete_expected_cost(self):
        self.assertEqual(rt.COST,((0.,1.,3.),(1.,0.,1.5),(5.,5.,0.)))
        self.assertAlmostEqual(rt.expected_cost([.2,.3,.5],0),1.8)
        self.assertAlmostEqual(rt.expected_cost([.2,.3,.5],1),.95)
        self.assertAlmostEqual(rt.expected_cost([.2,.3,.5],2),2.5)
        for p in ([.5,.5],[-1.,1.,1.],[float('nan'),0.,1.]):
            with self.assertRaises(ValueError):rt.expected_cost(p,0)

    def test_expert_confidence_only_and_floor(self):
        r=rows()[0];self.assertAlmostEqual(rt.confidence_weight(r),.64)
        r['label']['confidence']=.1;self.assertEqual(rt.confidence_weight(r),.25)
        r['label'].update(label_source='measured',confidence=.95)
        self.assertEqual(rt.confidence_weight(r),1.)
        r['label']['label_source']='expert_provisional';r['label']['confidence']=float('nan')
        with self.assertRaises(ValueError):rt.confidence_weight(r)
        self.assertEqual(rt.confidence_weight(r,False),1.)

    def test_parameter_whitelist_covers_every_eligible_identity_once(self):
        def param(ndim,grad=True):return types.SimpleNamespace(ndim=ndim,requires_grad=grad)
        named=[(n,param(d,g)) for n,d,g in [('layer.weight',2,True),('layer.bias',2,True),('norm.weight',2,True),('A_log',2,True),('dt_bias',2,True),('one',1,True),('frozen.weight',2,False),('excluded.weight',2,False)]]
        eligible={id(p) for _,p in named[:-1]};named.append(('tied',named[0][1]))
        groups,report=rt.parameter_groups(named,eligible)
        self.assertEqual(report['decay_names'],['layer.weight','frozen.weight'])
        self.assertEqual(len(report['no_decay_names']),5)
        self.assertEqual(sum(len(g['params']) for g in groups),7)
        self.assertEqual([g['weight_decay'] for g in groups],[.01,0.])


class LauncherTests(unittest.TestCase):
    def test_runtime_research_preflight_uses_same_sampler_before_model_or_artifacts(self):
        f=fixtures();te=f.import_train_eval_with_stubs();c=config()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir();r=f.make_train_rows(n=30)
            for row in r:row['label'].update(label_source='expert_provisional',confidence=.8)
            f.write_jsonl(root/'data/train.jsonl',r);f.write_jsonl(root/'data/calibration.jsonl',f.make_pool(n_groups=16))
            (root/'data/test.jsonl').write_text('INVALID TEST CANARY')
            roles=root/'roles.json';roles.write_text(json.dumps(f.make_roles(f.make_pool(n_groups=16))))
            args=argparse.Namespace(**c['training'],run='no-artifacts',legacy_calibration=False,smoke=False,
                research_config=str(ROOT/'configs/research-v1.json'),calibration_roles=str(roles),structural_dry_run=True)
            with mock.patch.object(te,'ROOT',root),mock.patch.object(te,'load_engine',side_effect=AssertionError('forbidden load')),mock.patch.object(rt,'sample_epochs',wraps=rt.sample_epochs) as sampler:
                te.run(args)
                self.assertEqual(sampler.call_count,1)
            self.assertFalse((root/'artifacts').exists())

    def test_synthetic_preflight_echoes_exact_config_and_never_opens_test(self):
        f=fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);r=f.make_train_rows(n=30)
            for row in r:row['label'].update(label_source='expert_provisional',confidence=.8)
            f.write_jsonl(root/'train.jsonl',r);f.write_jsonl(root/'calibration.jsonl',f.make_pool(n_groups=16))
            (root/'test.jsonl').write_text('DO NOT OPEN INVALID JSON')
            (root/'roles.json').write_text(json.dumps(f.make_roles(f.make_pool(n_groups=16))))
            command=[sys.executable,str(ROOT/'scripts/train.py'),'--run','research-cpu-fixture','--roles',str(root/'roles.json'),'--data-dir',str(root),'--research-config',str(ROOT/'configs/research-v1.json'),'--dry-run']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('not backbone compatibility',result.stdout)
            self.assertIn('--research-config',result.stdout)
            self.assertIn("'--rank', '16'",result.stdout)
            self.assertNotIn("'--fullft'",result.stdout)
            self.assertNotIn('fixture-train-',result.stdout)
            self.assertFalse((ROOT/'artifacts/research-cpu-fixture').exists())
            for extra in ('--legacy','--smoke'):
                refused=subprocess.run(command+[extra],capture_output=True,text=True)
                self.assertNotEqual(refused.returncode,0)
                self.assertNotIn('Command:',refused.stdout)


@unittest.skipIf(torch is None,'optional torch CPU tensor tests; no weights required')
class TensorTests(unittest.TestCase):
    def test_row_loss_value_gradient_sign_and_cost_use_raw_logits(self):
        c=config();c['loss']['auxiliary']='tier-only';r=rows()[0];r['label']['confidence']=1.
        p=rt.sampler_plan(rows(),c);p['priors']['tier']={'fast':.8,'balanced':.15,'flagship':.05}
        z=torch.zeros(3,requires_grad=True);loss=rt.row_loss({'tier':z},r,questions_from_source(),c,p)
        expected=-math.log(.8)+.25*(4/3)
        self.assertAlmostEqual(loss.item(),expected,places=6)
        loss.backward()
        # CE gradient = [.8-1,.15,.05]; cost gradient = .25/3*(C-mean(C)).
        want=torch.tensor([-.2,.15,.05])+.25/3*(torch.tensor([0.,1.,3.])-4/3)
        torch.testing.assert_close(z.grad,want)
        self.assertEqual(z.device.type,'cpu')

    def test_zero_auxiliary_really_has_no_auxiliary_gradient(self):
        c=config();c['loss']['auxiliary']='tier-only';q=questions_from_source()
        z={k:torch.zeros(len(v['criteria']),requires_grad=True) for k,v in q.items()}
        loss=rt.row_loss(z,rows()[0],q,c,rt.sampler_plan(rows(),c));loss.backward()
        self.assertIsNone(z['vision'].grad);self.assertIsNone(z['task_type'].grad)

    def toy(self):
        # Container only; NEVER execute this as a backbone or load weights.
        m=torch.nn.Module();m.config=types.SimpleNamespace(_commit_hash=rc.PINNED_REVISION)
        m.language_model=torch.nn.Module();m.language_model.layers=torch.nn.ModuleList()
        for _ in range(6):
            layer=torch.nn.Module();layer.q_proj=torch.nn.Linear(16,20);m.language_model.layers.append(layer)
        m.language_model.norm=torch.nn.LayerNorm(20)
        m.visual=torch.nn.Linear(16,16)
        return m

    def test_dora_enumeration_and_support_shape_checks(self):
        m=self.toy();targets=rc.enumerate_dora_targets(m)
        self.assertEqual(len(targets),6);self.assertEqual(targets[0]['shape'],[20,16])
        m.config._commit_hash=None
        with self.assertRaises(ValueError):rc.enumerate_dora_targets(m)
        m.config._commit_hash=rc.PINNED_REVISION;m.language_model.layers[0].q_proj=torch.nn.Linear(3,3)
        with self.assertRaises(ValueError):rc.enumerate_dora_targets(m)

    def test_dora_rejects_custom_linear_unknown_projection_and_missing_peft_api(self):
        class Custom(torch.nn.Linear):pass
        m=self.toy();m.language_model.layers[0].q_proj=Custom(16,20)
        with self.assertRaises(ValueError):rc.enumerate_dora_targets(m)
        m=self.toy();m.language_model.layers[0].unknown=torch.nn.Linear(16,20)
        with self.assertRaises(ValueError):rc.enumerate_dora_targets(m)
        m=self.toy()
        with self.assertRaises(ValueError):rc.dora_config(m,lambda r:None)

    def test_mocked_peft_configuration_is_not_a_runtime_compatibility_claim(self):
        class Config:
            def __init__(self,r,lora_alpha,lora_dropout,target_modules,bias,use_dora):self.__dict__.update(locals());self.__dict__.pop('self')
        m=self.toy();c,report=rc.dora_config(m,Config)
        self.assertEqual((c.r,c.lora_alpha,c.lora_dropout,c.use_dora),(16,32,.05,True))
        with self.assertRaises(ValueError):rc.verify_dora_injection(m,report)
        for t in report['targets']:
            module=m.get_submodule(t['name'])
            module.lora_A={'default':types.SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(16,16)))}
            module.lora_B={'default':types.SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(20,16)))}
            module.lora_magnitude_vector={'default':types.SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(20)))}
            module.use_dora={'default':True}
        rc.verify_dora_injection(m,report)
        self.assertIn('NOT established',report['runtime_status'])
        m.language_model.layers[0].q_proj.lora_B['default'].weight=torch.zeros(2,2)
        with self.assertRaises(ValueError):rc.verify_dora_injection(m,report)

    def test_staged_optimizer_keeps_initially_frozen_parameters_and_no_early_state(self):
        m=self.toy();eligible,top=rc.staged_parameters(m)
        for p in m.parameters():p.requires_grad=False
        report=rc.apply_stage(eligible,top,0,20)
        self.assertEqual(report['initial_updates'],3);self.assertEqual(top,{2,3,4,5})
        bottom=m.language_model.layers[0].q_proj.weight
        self.assertFalse(bottom.requires_grad)
        groups,_=rt.parameter_groups(m.named_parameters(),{id(p) for _,p in eligible})
        opt=torch.optim.AdamW(groups,lr=.01)
        before=bottom.detach().clone()
        # Arithmetic optimizer unit check, assigned toy gradients only; no loss
        # loop, data, real forward, training run, GPU or backbone exists.
        for _,p in eligible:
            if p.requires_grad:p.grad=torch.ones_like(p)
        opt.step();opt.zero_grad(set_to_none=True)
        torch.testing.assert_close(bottom,before);self.assertNotIn(bottom,opt.state)
        rc.apply_stage(eligible,top,2,20);self.assertFalse(bottom.requires_grad)
        report=rc.apply_stage(eligible,top,3,20);self.assertTrue(bottom.requires_grad)
        self.assertEqual(report['phase'],'full-language');self.assertFalse(m.visual.weight.requires_grad)
        bottom.grad=torch.ones_like(bottom);opt.step()
        self.assertIn(bottom,opt.state)

    def test_neftune_is_bounded_train_only_and_removable(self):
        embedding=torch.nn.Embedding(4,6);m=types.SimpleNamespace(get_input_embeddings=lambda:embedding)
        ids=torch.tensor([[0,1]])
        clean=embedding(ids).detach();handle=rc.install_neftune(m,5.)
        noisy=embedding(ids).detach();bound=5/math.sqrt(12)
        self.assertTrue(torch.any(noisy!=clean));self.assertLessEqual((noisy-clean).abs().max().item(),bound+1e-6)
        with self.assertRaises(ValueError):rc.install_neftune(m,5.)
        embedding.eval();torch.testing.assert_close(embedding(ids),clean)
        handle.remove();embedding.train();torch.testing.assert_close(embedding(ids),clean)


if __name__=='__main__':unittest.main()
