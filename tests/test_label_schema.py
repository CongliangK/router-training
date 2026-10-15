"""Issue #18: invented CPU fixtures only; never weights, private labels or training."""
import argparse, asyncio, contextlib, copy, io, json, math, subprocess, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import label_schema as ls
import model_utils as mu
import calibration_groups as cg
from evaluation_metrics import metrics
import test_calibration_groups as fixtures
import test_server_contract as sc

EXAMPLES=ROOT/'data/synthetic-examples/labels-v2-creative-vision.jsonl'


def v2_rows(rows):
    """Invented fixture conversion, NOT a migration of observed labels."""
    rows=copy.deepcopy(rows)
    for row in rows:
        row.pop('synthetic',None)
        row['label_schema_version']=ls.V2
        row['label']['label_status']='resolved'
        if row['label']['task_type']=='visual':row['label']['task_type']='extraction'
    return rows


def predicted(row, selected=ls.V2):
    row=copy.deepcopy(row);row.update(answers={},context_truncated=False,latency_ms=1.)
    for head,spec in ls.questions_for(selected).items():
        classes=list(spec['criteria']);expected=row['label'][{'vision':'needs_vision'}.get(head,head)]
        row['answers'][head]={'choice':expected,'confidence':1.,
            'probabilities':{c:float(c==expected) for c in classes},'logits':[float(c==expected) for c in classes]}
    return row


def write_fixture(directory):
    train=v2_rows(fixtures.make_train_rows());pool=v2_rows(fixtures.make_pool())
    fixtures.write_jsonl(directory/'train.jsonl',train)
    fixtures.write_jsonl(directory/'calibration.jsonl',pool)
    (directory/'test.jsonl').write_text('DO NOT READ TEST: INVALID JSON')
    roles=directory/'roles.json';roles.write_text(json.dumps(fixtures.make_roles(pool)))
    return train,pool,roles


class VersionTests(unittest.TestCase):
    def test_default_questions_are_unchanged_seven_types(self):
        before=copy.deepcopy(mu.QUESTIONS)
        self.assertEqual(ls.version(),ls.LEGACY)
        self.assertEqual(ls.questions_for(),before)
        self.assertEqual(list(before['task_type']['criteria']),fixtures.SEVEN)
        self.assertEqual(mu.TYPES,fixtures.SEVEN)
        self.assertEqual(list(ls.questions_for(ls.V2)['task_type']['criteria']),list(ls.SIX_TYPES))
        changed=ls.questions_for(ls.V2);changed['tier']['criteria'].clear()
        self.assertEqual(mu.QUESTIONS,before)

    def test_unknown_and_mixed_versions_are_rejected(self):
        for value in ['v2',2,True,'router-label-v3']:
            with self.assertRaises(ValueError):ls.version(value)
        row=v2_rows(fixtures.make_train_rows(1))[0]
        for selected in [ls.LEGACY]:
            with self.assertRaises(ValueError):ls.validate_supervised_rows([row],selected)
        row['label']['label_schema_version']=ls.LEGACY
        with self.assertRaises(ValueError):ls.validate_supervised_rows([row],ls.V2)

    def test_missing_v2_declaration_and_bad_labels_fail_even_optional_task_type(self):
        row=v2_rows(fixtures.make_train_rows(1))[0]
        for key,value in [('needs_vision','uncertain'),('task_type','visual'),('tier','other'),('label_status','adjudication')]:
            broken=copy.deepcopy(row);broken['label'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ls.validate_supervised_rows([broken],ls.V2)
        row.pop('label_schema_version')
        with self.assertRaises(ValueError):ls.validate_supervised_rows([row],ls.V2)

    def test_legacy_uncertain_preserved_by_prepare_but_not_calibration(self):
        label={'needs_vision':'uncertain','task_type':'visual','tier':'balanced'}
        self.assertEqual(ls.partition_labels([label]),([label],[]))
        row={'id':'synthetic-boundary','split_group':'synthetic-source','label':label}
        with self.assertRaises(cg.PlanError):cg.validate_rows([row],cg.default_business_schema())

    def test_all_synthetic_examples_are_supplemental_not_real_support(self):
        rows=ls.supplemental_rows([EXAMPLES],ls.V2)
        self.assertEqual(len(rows),15)
        self.assertEqual(len({r['id'] for r in rows}),15)
        self.assertTrue(all(r['synthetic'] is True and r['supplemental_only'] is True for r in rows))
        self.assertTrue(all(r['label']['task_type']=='creative' and r['label']['needs_vision']=='yes' for r in rows))
        self.assertEqual(sum(r['label']['tier']=='flagship' for r in rows),4)
        self.assertEqual(len({r['group'] for r in rows}),1)
        self.assertTrue(all('split' not in r and r['prompt'].startswith('SYNTHETIC EXAMPLE ONLY.') for r in rows))
        with self.assertRaises(ValueError):ls.supplemental_rows([EXAMPLES],ls.LEGACY)
        with self.assertRaises(ValueError):ls.validate_supervised_rows(rows,ls.V2)
        with self.assertRaises(cg.PlanError):cg.validate_rows(rows,cg.default_business_schema(schema_version=ls.V2))


class RubricTests(unittest.TestCase):
    def test_executable_primary_wins_and_nonprogramming_creation_is_creative(self):
        # Reviewer facts, not a word-matching classifier: a creative CSS page is executable.
        self.assertEqual(ls.reviewed_task_type('write_executable'),'implementation')
        self.assertEqual(ls.reviewed_task_type('nonprogramming_creative'),'creative')
        old={'tier':'balanced','needs_vision':'yes','task_type':'creative'}
        new=ls.review_legacy_label(old,primary_action='write_executable')
        self.assertEqual(new['task_type'],'implementation');self.assertEqual(old['task_type'],'creative')

    def test_visual_maps_by_explicit_action_evidence_only(self):
        for action,expected in [('visual_extract','extraction'),('visual_review','review')]:
            self.assertEqual(ls.reviewed_task_type(action,'synthetic reviewer: the deliverable names this action'),expected)
            with self.assertRaises(ValueError):ls.reviewed_task_type(action)
        with self.assertRaises(ValueError):ls.reviewed_task_type('analyze image')
        old={'tier':'balanced','needs_vision':'yes','task_type':'visual'}
        for kwargs in [{},{'primary_action':'visual_review'},{'primary_action':'ambiguous'}]:
            new=ls.review_legacy_label(old,**kwargs)
            self.assertEqual(new['label_status'],'adjudication');self.assertIsNone(new['task_type'])

    def test_uncertain_requires_binary_decision_and_evidence(self):
        old={'id':'invented-review','tier':'balanced','needs_vision':'uncertain','task_type':'visual'}
        before=copy.deepcopy(old)
        for kwargs in [{},{'vision_decision':'yes'},{'vision_evidence':'image supplied'},{'vision_decision':'no','vision_evidence':' '}]:
            new=ls.review_legacy_label(old,**kwargs)
            self.assertEqual(new['label_status'],'adjudication');self.assertIsNone(new['needs_vision'])
        new=ls.review_legacy_label(old,vision_decision='yes',vision_evidence='invented review: actual screenshot needed',
                                  primary_action='visual_extract',action_evidence='invented review: transcribe visible labels')
        self.assertEqual((new['needs_vision'],new['task_type'],new['label_status']),('yes','extraction','resolved'))
        self.assertEqual(old,before)
        active,pending=ls.partition_labels([new,ls.review_legacy_label({**old,'id':'pending'})],ls.V2)
        self.assertEqual((len(active),len(pending)),(1,1))


class CheckpointTests(unittest.TestCase):
    def test_unversioned_calibration_is_legacy_only(self):
        self.assertEqual(ls.resolve_run_schema({'temperatures':{}}),ls.LEGACY)
        with self.assertRaises(ValueError):ls.resolve_run_schema({},requested=ls.V2)
        with self.assertRaises(ValueError):ls.resolve_run_schema({'label_schema_version':ls.V2})

    def test_v2_needs_both_matching_complete_metadata_and_explicit_optin(self):
        meta=ls.metadata(ls.V2);cal={'label_schema':meta}
        self.assertEqual(ls.resolve_run_schema(cal,meta,ls.V2),ls.V2)
        for args in [(cal,meta,ls.LEGACY),(cal,None,ls.V2),({},meta,ls.V2)]:
            with self.assertRaises(ValueError):ls.resolve_run_schema(*args)
        for key,value in [('question_sha256','0'*64),('label_schema_version','router-label-v99'),('classes',{})]:
            bad=copy.deepcopy(meta);bad[key]=value
            with self.assertRaises(ValueError):ls.resolve_run_schema({'label_schema':bad},bad,ls.V2)
        bad=copy.deepcopy(meta);bad['classes']['task_type'].reverse()
        with self.assertRaises(ValueError):ls.resolve_run_schema({'label_schema':bad},bad,ls.V2)
        bad=copy.deepcopy(meta);bad['extra']='not ignored'
        with self.assertRaises(ValueError):ls.resolve_run_schema(cal,bad,ls.V2)

    def test_model_loader_checks_boundary_before_reference_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(mu,'reference_module',side_effect=AssertionError('NO MODEL')) as ref:
            root=Path(tmp);(root/'calibration.json').write_text('{}')
            with self.assertRaises(ValueError):mu.load_engine(full_checkpoint=root/'full-checkpoint',schema_version=ls.V2)
            ref.assert_not_called()
            meta=ls.metadata(ls.V2);(root/'calibration.json').write_text(json.dumps({'label_schema':meta}))
            (root/'label-schema.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):mu.load_engine(adapter=root/'adapter')
            ref.assert_not_called()

    def test_requests_share_truncation_and_bound_schema_without_model(self):
        tokenizer=types.SimpleNamespace(encode=lambda s,**k:list(s),decode=lambda ids:''.join(ids))
        legacy=types.SimpleNamespace(tokenizer=tokenizer)
        new=types.SimpleNamespace(tokenizer=tokenizer,label_schema_version=ls.V2)
        old_request,old_cut=mu.request_for(legacy,'<decision>'+('abcdef'*10),12)
        new_request,new_cut=mu.request_for(new,'<decision>'+('abcdef'*10),12)
        self.assertTrue(old_cut and new_cut)
        self.assertEqual(old_request['state'],new_request['state'])
        self.assertNotIn('<decision>',new_request['state']['delegated_task'])
        self.assertEqual(len(old_request['questions']['task_type']['criteria']),7)
        self.assertEqual(len(new_request['questions']['task_type']['criteria']),6)
        with self.assertRaises(ValueError):mu.request_for(legacy,'x',schema_version=ls.V2)
        with self.assertRaises(ValueError):mu.request_for(new,'x',schema_version=ls.LEGACY)


class ScoringTests(unittest.TestCase):
    def test_actual_score_and_calibration_use_six_options_without_torch(self):
        class Scalar(float):
            def item(self):return float(self)
        class Tensor:
            def __init__(self,values):self.values=list(values)
            def __len__(self):return len(self.values)
            def __getitem__(self,index):return Scalar(self.values[index])
            def __setitem__(self,index,value):self.values[index]=float(value)
            def __truediv__(self,value):return Tensor([x/value for x in self.values])
            def clone(self):return Tensor(self.values)
            def cpu(self):return self
            def tolist(self):return list(self.values)
        def softmax(tensor,dim=-1):
            values=[math.exp(x-max(tensor.values)) for x in tensor.values]
            return Tensor([x/sum(values) for x in values])
        torch=types.SimpleNamespace(tensor=Tensor,softmax=softmax,inference_mode=contextlib.nullcontext)
        engine=types.SimpleNamespace(label_schema_version=ls.V2,
            tokenizer=types.SimpleNamespace(encode=lambda text,**k:list(text),decode=lambda ids:''.join(ids)),
            backend=types.SimpleNamespace(model=types.SimpleNamespace(eval=lambda:None)))
        logits={head:Tensor([0.]*(len(spec['criteria'])-1)+[2.]) for head,spec in ls.questions_for(ls.V2).items()}
        with patch.dict(sys.modules,{'torch':torch}),patch.object(mu,'field_logits',return_value=(logits,10)):
            result=mu.score(engine,'Synthetic score contract.',schema_version=ls.V2)
            self.assertEqual(result['label_schema_version'],ls.V2)
            self.assertEqual(result['answers']['task_type']['choice'],'creative')
            self.assertEqual(list(result['answers']['task_type']['probabilities']),list(ls.SIX_TYPES))
            logits['task_type']=Tensor([0.]*7)
            with self.assertRaises(ValueError):mu.score(engine,'Synthetic wrong width.')
        row=v2_rows(fixtures.make_train_rows(1))[0];row.update(result)
        te=fixtures.import_train_eval_with_stubs()
        with patch.object(te,'torch',torch):
            recalibrated=te.calibrated([row],{'tier':2.,'vision':2.,'task_type':2.},schema_version=ls.V2)
            self.assertEqual(len(recalibrated[0]['answers']['task_type']['probabilities']),6)
            with self.assertRaises(ValueError):te.calibrated([row],{'tier':1.,'vision':1.,'task_type':1.})
            wrong=copy.deepcopy(row);wrong['answers']['task_type']['logits'].append(0.)
            with self.assertRaises(ValueError):te.fit_temperatures([wrong],schema_version=ls.V2)

    def test_v2_full_product_loader_binds_schema_with_mocked_reference_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta=ls.metadata(ls.V2)
            (root/'calibration.json').write_text(json.dumps({'label_schema':meta}))
            (root/'label-schema.json').write_text(json.dumps(meta))
            engine=types.SimpleNamespace(backend=types.SimpleNamespace(model=types.SimpleNamespace(eval=lambda:None)))
            reference=types.SimpleNamespace(DecisionEngine=lambda **kwargs:engine)
            with patch.object(mu,'reference_module',return_value=reference):
                loaded=mu.load_engine(full_checkpoint=root/'full-checkpoint',schema_version=ls.V2,device='cpu')
            self.assertIs(loaded,engine);self.assertEqual(engine.label_schema_version,ls.V2)


class MetricsTests(unittest.TestCase):
    def test_fixed_denominators_are_versioned_not_reinterpreted(self):
        row=v2_rows(fixtures.make_train_rows(1))[0];p=predicted(row)
        result=metrics([p],schema_version=ls.V2)
        self.assertEqual(result['label_schema_version'],ls.V2)
        self.assertEqual(result['task_type_per_class']['macro_f1_definition']['denominator'],6)
        self.assertAlmostEqual(result['task_type']['macro_f1'],1/6)
        self.assertNotIn('visual',result['task_type']['confusion'])
        with self.assertRaises(ValueError):metrics([p])
        old=fixtures.make_train_rows(1)[0];legacy=predicted(old,ls.LEGACY)
        self.assertEqual(metrics([legacy])['task_type_per_class']['macro_f1_definition']['denominator'],7)
        with self.assertRaises(ValueError):metrics([legacy],schema_version=ls.V2)

    def test_v2_rejects_seven_probabilities_and_logits(self):
        p=predicted(v2_rows(fixtures.make_train_rows(1))[0])
        for kind in ['probabilities','logits']:
            bad=copy.deepcopy(p)
            if kind=='probabilities':bad['answers']['task_type'][kind]['visual']=0.
            else:bad['answers']['task_type'][kind].append(0.)
            with self.subTest(kind=kind),self.assertRaises(ValueError):metrics([bad],schema_version=ls.V2)
        rows=ls.supplemental_rows([EXAMPLES],ls.V2)
        with self.assertRaises(ValueError):metrics([predicted(rows[0])],schema_version=ls.V2)


class PipelineTests(unittest.TestCase):
    def test_ast_preflight_runtime_schema_parity_and_unchanged_gates(self):
        te=fixtures.import_train_eval_with_stubs()
        for selected,n in [(ls.LEGACY,7),(ls.V2,6)]:
            schema=cg.default_business_schema(schema_version=selected)
            self.assertEqual(schema,te.grouped_schema(selected))
            self.assertEqual(len(schema['heads']['task_type']['classes']),n)
            self.assertTrue(schema['heads']['vision']['required'])
        pool=v2_rows(fixtures.make_pool());roles=fixtures.make_roles(pool)
        plan=te.build_grouped_plan(pool,roles,ls.V2)
        self.assertEqual(plan['plan_hash'],cg.preflight_roles(pool,roles,schema_version=ls.V2)['plan_hash'])
        self.assertEqual(plan['policy'],cg.DEFAULT_SUPPORT_POLICY)
        self.assertEqual(plan['optional_head_status']['task_type']['fixed_T'],1.)
        bad=copy.deepcopy(pool);bad[0]['label']['task_type']='visual'
        with self.assertRaises(cg.PlanError):te.build_grouped_plan(bad,roles,ls.V2)
        for row in pool:row['label']['needs_vision']='no'
        with self.assertRaises(cg.PlanError):te.build_grouped_plan(pool,roles,ls.V2)

    def test_calibration_adapter_never_relabels_legacy_predictions_as_v2(self):
        te=fixtures.import_train_eval_with_stubs();pool=v2_rows(fixtures.make_pool())
        plan=te.build_grouped_plan(pool,fixtures.make_roles(pool),ls.V2)
        scored=[predicted(row) for row in pool]
        with patch.object(cg,'select_tau_oof',return_value={'status':'synthetic-adapter-only'}) as select:
            te.run_grouped_calibration(plan,scored,{},plan['protocol'],'not-used-by-mock')
            self.assertTrue(all(r['label_schema_version']==ls.V2 for r in select.call_args.kwargs['rows']))
        for kind in ['unversioned','seven-logits']:
            bad=copy.deepcopy(scored)
            for row in bad:
                if kind=='unversioned':row.pop('label_schema_version')
                else:row['answers']['task_type']['logits'].append(0.)
            with patch.object(cg,'select_tau_oof') as select,self.assertRaises(ValueError):
                te.run_grouped_calibration(plan,bad,{},plan['protocol'],'not-used')
            select.assert_not_called()
        with self.assertRaises(ValueError):te.fit_temperatures(scored,schema_version=ls.V2)
        with self.assertRaises(cg.PlanError):cg.schema_from_questions(mu.QUESTIONS,ls.V2)

    def test_prior_excludes_supplements_and_refuses_missing_real_support(self):
        train=v2_rows(fixtures.make_train_rows());schema=cg.default_business_schema(schema_version=ls.V2)
        examples=ls.supplemental_rows([EXAMPLES],ls.V2)
        before=cg.compute_prior(train,'tier',schema)
        after=cg.compute_prior(train+examples,'tier',schema)
        self.assertEqual(before['probabilities'],after['probabilities'])
        self.assertEqual(after['excluded_synthetic'],15)
        with self.assertRaises(cg.PlanError):cg.compute_prior(examples,'tier',schema)
        with self.assertRaises(cg.PlanError):cg.compute_prior(fixtures.make_train_rows(),'tier',schema)

    def test_cli_v2_preflight_no_torch_no_test_and_launcher_flag_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);_,_,roles=write_fixture(directory)
            recipe=directory/'recipe.json';recipe.write_text('{"synth":false}')
            command=[sys.executable,str(ROOT/'scripts/train.py'),'--run','labels18-synthetic-structural',
                     '--label-schema',ls.V2,'--data-dir',str(directory),'--roles',str(roles),
                     '--recipe',str(recipe),'--supplemental',str(EXAMPLES),'--dry-run']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            printed=next(l for l in result.stdout.splitlines() if l.startswith('Command:'))
            import ast
            actual=ast.literal_eval(printed.split('Command:',1)[1].strip())
            for flag,value in [('--label-schema',ls.V2),('--data-dir',str(directory)),('--supplemental',str(EXAMPLES))]:
                self.assertEqual(actual[actual.index(flag)+1],value)
            self.assertNotIn('--synth',actual)
            self.assertFalse((ROOT/'artifacts/labels18-synthetic-structural').exists())
            legacy=subprocess.run([sys.executable,str(ROOT/'scripts/validate_calibration.py'),'--data-dir',str(directory),'--roles',str(roles)],capture_output=True,text=True)
            self.assertEqual(legacy.returncode,1)
            code="import sys; import label_schema, calibration_groups; calibration_groups.default_business_schema(schema_version=label_schema.V2); assert 'torch' not in sys.modules"
            pure=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(pure.returncode,0,pure.stderr)

    def test_supplemental_join_preflight_rejects_overlap_and_legacy_rows(self):
        rows=ls.supplemental_rows([EXAMPLES],ls.V2)
        with self.assertRaises(ValueError):ls.training_supplemental([EXAMPLES],[rows[0]],ls.V2)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);_,_,roles=write_fixture(directory)
            bad=directory/'bad-supplement.jsonl';fixtures.write_jsonl(bad,fixtures.make_train_rows(1))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/validate_calibration.py'),
                '--data-dir',str(directory),'--roles',str(roles),'--label-schema',ls.V2,
                '--supplemental',str(bad)],capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            self.assertNotIn('fixture-train-',result.stdout)

    def test_real_run_structural_adapter_never_loads_model_or_writes_artifacts(self):
        reads=[];te=fixtures.import_train_eval_with_stubs(reads)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir();_,_,roles=write_fixture(data)
            args=argparse.Namespace(label_schema=ls.V2,run='invented-run',seed=1,synth=False,smoke=False,
                calibration_roles=str(roles),legacy_calibration=False,structural_dry_run=True,data_dir=data)
            with patch.object(te,'ROOT',root),patch.object(te,'load_engine',side_effect=AssertionError('NO MODEL')) as load,contextlib.redirect_stdout(io.StringIO()):
                te.run(args)
                load.assert_not_called()
            self.assertFalse((root/'artifacts').exists())
            self.assertEqual({Path(p).name for p in reads},{'train.jsonl','calibration.jsonl'})

    def test_v2_refuses_legacy_calibration_and_legacy_augmentation_before_data(self):
        te=fixtures.import_train_eval_with_stubs()
        for legacy,synth in [(True,False),(False,True)]:
            args=argparse.Namespace(label_schema=ls.V2,legacy_calibration=legacy,synth=synth)
            with self.assertRaises(SystemExit):te.run(args)

    def test_prepare_pending_ledger_new_output_and_legacy_input_unchanged(self):
        import prepare
        from synthetic_data import synthetic_splits
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'input';data.mkdir()
            tasks,splits,_=synthetic_splits()
            fixtures.write_jsonl(data/'tasks.jsonl',tasks)
            labels=[r['label'] for rows in splits.values() for r in rows]
            for n in range(1,5):fixtures.write_jsonl(data/f'expert-recovery-{n}.jsonl',labels[(n-1)*30:n*30])
            for name,rows in splits.items():fixtures.write_jsonl(data/f'{name}.jsonl',rows)
            # Fixture generator writes clearly synthetic primary rows. Strip only its
            # synthetic flag for this structural test, NEVER claim real observations.
            tasks=prepare.load(data/'tasks.jsonl')
            for row in tasks:row.pop('synthetic',None)
            fixtures.write_jsonl(data/'tasks.jsonl',tasks)
            frozen={p.name:p.read_bytes() for p in data.iterdir() if p.is_file()}
            for n in range(1,5):
                path=data/f'expert-recovery-{n}.jsonl';labels=prepare.load(path)
                for label in labels:
                    label.update(label_schema_version=ls.V2,label_status='resolved')
                    if label['task_type']=='visual':label['task_type']='extraction'
                if n==1:
                    labels[0].update(label_status='adjudication',needs_vision=None,adjudication_reasons=['invented missing context'])
                fixtures.write_jsonl(path,labels)
            reviewed={p.name:p.read_bytes() for p in data.iterdir() if p.is_file()}
            with self.assertRaises(ValueError):prepare.prepare(data,schema_version=ls.V2)
            out=root/'new-v2'
            with contextlib.redirect_stdout(io.StringIO()):prepare.prepare(data,schema_version=ls.V2,output=out)
            rows=[r for name in ['train','calibration','test'] for r in prepare.load(out/f'{name}.jsonl')]
            ls.validate_supervised_rows(rows,ls.V2)
            self.assertEqual(len(rows),119);self.assertEqual(len(prepare.load(out/'adjudication.jsonl')),1)
            report=json.loads((out/'split-report.json').read_text())
            self.assertEqual(report['label_review']['adjudication_or_excluded'],1)
            self.assertEqual(report['label_schema'],ls.metadata(ls.V2))
            self.assertEqual(reviewed,{p.name:p.read_bytes() for p in data.iterdir() if p.is_file()})
            self.assertEqual(frozen['train.jsonl'],(data/'train.jsonl').read_bytes())
            with self.assertRaises(FileExistsError):prepare.prepare(data,schema_version=ls.V2,output=out)


class ServerTests(unittest.TestCase):
    def test_server_schema_mismatch_precedes_model_load(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);run_dir=root/'artifacts/synthetic';run_dir.mkdir(parents=True)
                (run_dir/'calibration.json').write_text('{}')
                with patch.object(sc.server,'ROOT',root),patch.object(sc.server,'load_engine',side_effect=AssertionError('NO MODEL')) as loader:
                    app=sc.server.create_app('synthetic',label_schema=ls.V2)
                    with self.assertRaises(ValueError):
                        async with app.router.lifespan_context(app):pass
                    loader.assert_not_called()
        asyncio.run(run())

    def test_server_v2_metadata_and_advisory_permissions(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);run_dir=root/'artifacts/synthetic';(run_dir/'full-checkpoint').mkdir(parents=True)
                meta=ls.metadata(ls.V2)
                (run_dir/'label-schema.json').write_text(json.dumps(meta))
                (run_dir/'calibration.json').write_text(json.dumps({'temperatures':{},'label_schema':meta}))
                with patch.object(sc.server,'ROOT',root),patch.object(sc.server,'load_engine',return_value=object()) as loader,patch.object(sc.server,'score',return_value=sc.prediction()),patch.object(sc.server,'record_observation'):
                    app=sc.server.create_app('synthetic',label_schema=ls.V2)
                    async with app.router.lifespan_context(app):
                        status,health=await sc.request(app,'/health')
                        self.assertEqual(status,200);self.assertEqual(health['label_schema_version'],ls.V2)
                        status,result=await sc.request(app,'/route',{'prompt':'Invented schema test.'})
                        self.assertEqual(status,200);self.assertEqual(result['label_schema_version'],ls.V2)
                        self.assertIs(result['execution_allowed'],False)
                    self.assertEqual(loader.call_args.kwargs['schema_version'],ls.V2)
                    app=sc.server.create_app('synthetic')
                    with self.assertRaises(ValueError):
                        async with app.router.lifespan_context(app):pass
        asyncio.run(run())


if __name__=='__main__':unittest.main()
