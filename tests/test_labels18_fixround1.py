"""Five independent red-team regressions: invented CPU data and pure mocks only.

No reference/model/weights, GPU, training, network, or actual training spawn.
The grouped adapter and selector run for real on raw 16-group plans.
"""
import argparse, ast, contextlib, copy, hashlib, importlib.util, io, json, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import calibration_groups as cg
import task_label_schema as schema
from evaluation_metrics import metrics
from scripts.validate_calibration import run_validation
from split_reporting import split_report
from test_calibration_groups import import_train_eval_with_stubs


def pool(selected):
    rows=[]
    for g in range(16):
        for i in range(6):
            rows.append({'id':f'invented-{g}-{i}','prompt':f'INVENTED CPU fixture {g}/{i}',
                'group':f'root-{g}','split_group':f'group-{g:02}',
                'task_label_schema_version':selected,
                'label':{'tier':('fast','balanced','flagship')[i%3],
                         'needs_vision':('no','yes')[i%2],
                         'task_type':schema.SIX_TYPES[i],'label_status':'resolved'}})
    return rows


def prediction(row,selected):
    result=copy.deepcopy(row);result.update(answers={},context_truncated=False,input_tokens=1,latency_ms=0.)
    for head,spec in schema.questions_for(selected).items():
        labels=list(spec['criteria']);expected=row['label'][{'vision':'needs_vision'}.get(head,head)]
        result['answers'][head]={'choice':expected,'confidence':1.,
            'probabilities':{c:float(c==expected) for c in labels},'logits':[2.*(c==expected) for c in labels]}
    return result


def score_payload(row,selected):
    result=prediction(row,selected)
    return {k:result[k] for k in ('answers','task_label_schema_version','context_truncated','input_tokens')}


def roles(rows):
    groups=sorted({r['split_group'] for r in rows})
    return {'role_groups':{'A':groups[:5],'B':groups[5:11],'C':groups[11:]}}


def digest_ref():
    # Internally consistent invented digest METADATA, not weight bytes or model files.
    files={'invented-toy-array.txt':hashlib.sha256(b'INVENTED CPU ARRAYS ONLY').hexdigest()}
    return json.dumps({'algorithm':'sha256','file_count':1,'file_digests':files,
        'digest':hashlib.sha256(''.join(f'{k}:{v}\n' for k,v in sorted(files.items())).encode()).hexdigest()})


def write_rows(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def dataset(directory,selected=schema.V2):
    rows=pool(selected);train=copy.deepcopy(rows[:6])
    for r in train:r['id']='train-'+r['id'];r['group']='train-'+r['group'];r['split_group']='train-'+r['split_group']
    write_rows(directory/'train.jsonl',train);write_rows(directory/'calibration.jsonl',rows)
    (directory/'test.jsonl').write_text('INVALID JSON: TEST MUST NEVER BE READ')
    rp=directory/'roles.json';rp.write_text(json.dumps(roles(rows)))
    if selected==schema.V2:
        meta=schema.metadata(selected)
        (directory/'label-schema.json').write_text(json.dumps(meta))
        (directory/'split-report.json').write_text(json.dumps({'label_schema':meta,'splits':{}}))
    return rows,rp


def load_script(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class FreshScoreJoinTests(unittest.TestCase):
    def test_real_grouped_adapter_raw_16_groups_both_versions_and_stale_cache_independence(self):
        te=import_train_eval_with_stubs();prior={'tier':dict.fromkeys(('fast','balanced','flagship'),1/3)}
        for selected in schema.VERSIONS:
            with self.subTest(version=selected):
                rows=pool(selected);plan=te.build_grouped_plan(rows,roles(rows),selected)
                before=copy.deepcopy(plan);scored=[prediction(r,selected) for r in plan['rows_by_role']['B']]
                calls=[]
                def fit(logits,*args):
                    calls.append(copy.deepcopy(logits));return cg.fit_temperature(logits,*args)
                proto={**plan['protocol'],'fit':fit}
                first=te.run_grouped_calibration(plan,scored,prior,proto,digest_ref())
                self.assertEqual(first['status'],'selected');self.assertEqual(plan,before)
                # First tau is zero: observed fitted input must be exactly submitted scores.
                wanted=[r['answers']['tier']['logits'] for r in scored if r['split_group'] not in set(plan['folds'][0])]
                self.assertEqual(calls[0],wanted)
                changed=copy.deepcopy(scored)
                for r in changed:
                    for answer in r['answers'].values():answer['logits']=[-100*x for x in answer['logits']]
                calls.clear();second=te.run_grouped_calibration(plan,changed,prior,proto,digest_ref())
                self.assertEqual(calls[0],[[-100*x for x in values] for values in wanted])
                self.assertNotEqual(first['candidates'],second['candidates'])
                # Arbitrarily wrong planning caches cannot change either result.
                for role_rows in plan['rows_by_role'].values():
                    for r in role_rows:r['logits']={h:[999.]*len(s['criteria']) for h,s in schema.questions_for(selected).items()}
                self.assertEqual(te.run_grouped_calibration(plan,scored,prior,proto,digest_ref()),first)
                self.assertEqual(te.run_grouped_calibration(plan,changed,prior,proto,digest_ref()),second)

    def test_missing_duplicate_mismatched_group_root_label_and_version_refuse_before_fit(self):
        te=import_train_eval_with_stubs()
        for selected in schema.VERSIONS:
            rows=pool(selected);plan=te.build_grouped_plan(rows,roles(rows),selected)
            good=[prediction(r,selected) for r in plan['rows_by_role']['B']]
            for mutation in ('missing','missing-id','duplicate','missing-group','group','root','missing-label','label','version','nested-version','source-version'):
                bad=copy.deepcopy(good)
                if mutation=='missing':bad.pop()
                elif mutation=='missing-id':bad[0].pop('id')
                elif mutation=='missing-label':bad[0].pop('label')
                elif mutation=='duplicate':bad.append(copy.deepcopy(bad[0]))
                elif mutation=='missing-group':bad[0].pop('split_group')
                elif mutation=='group':bad[0]['split_group']='group-00'  # role A, not B
                elif mutation=='root':bad[0]['group']='not-the-planned-root'
                elif mutation=='label':bad[0]['label']['tier']='balanced'
                elif mutation=='version':bad[0]['task_label_schema_version']=next(v for v in schema.VERSIONS if v!=selected)
                elif mutation=='nested-version':bad[0]['label']['task_label_schema_version']=next(v for v in schema.VERSIONS if v!=selected)
                else:bad[0]['label_schema_version']=True
                with self.subTest(version=selected,mutation=mutation),patch.object(cg,'fit_temperature') as fit:
                    with self.assertRaises(ValueError):te.run_grouped_calibration(plan,bad,{'tier':dict.fromkeys(('fast','balanced','flagship'),1/3)},plan['protocol'],digest_ref())
                    fit.assert_not_called()

    def test_selector_itself_checks_direct_scored_join_and_role_isolation(self):
        rows=pool(schema.V2);plan=cg.preflight_roles(rows,roles(rows),schema_version=schema.V2)
        good=[{**r,'logits':{h:a['logits'] for h,a in prediction(r,schema.V2)['answers'].items()}} for r in plan['rows_by_role']['B']]
        for mutation in ('extra-A','extra-C','missing-group','label','version'):
            bad=copy.deepcopy(good)
            if mutation.startswith('extra'):bad.append(copy.deepcopy(plan['rows_by_role'][mutation[-1]][0]))
            elif mutation=='missing-group':bad[0].pop('split_group')
            elif mutation=='label':bad[0]['label']['task_type']='creative'
            else:bad[0]['task_label_schema_version']=schema.LEGACY
            with self.subTest(mutation=mutation),patch.object(cg,'fit_temperature') as fit:
                with self.assertRaises(ValueError):cg.select_tau_oof(plan,bad,{'tier':dict.fromkeys(('fast','balanced','flagship'),1/3)},digest_ref())
                fit.assert_not_called()


class PredictionBoundaryTests(unittest.TestCase):
    def test_actual_15_supplements_keep_all_provenance_and_cannot_supply_metrics(self):
        te=import_train_eval_with_stubs()
        examples=schema.supplemental_rows([ROOT/'data/synthetic-examples/labels-v2-creative-vision.jsonl'],schema.V2)
        original=copy.deepcopy(examples);lookup={r['prompt']:r for r in examples}
        for r in examples:r['provenance']={'invented':True};r['partition']='synthetic_train_only';r['label_schema_version']=1
        with patch.object(te,'score',side_effect=lambda e,prompt,**kw:score_payload(lookup[prompt],schema.V2)):
            scored=te.predict_rows(object(),examples,schema.V2)
        self.assertEqual(len(scored),15)
        for source,result in zip(examples,scored):
            for key,value in source.items():
                if key!='prompt':self.assertEqual(result[key],value)
        with self.assertRaisesRegex(ValueError,'supplemental'):metrics(scored,schema_version=schema.V2)
        self.assertEqual(sum(r['synthetic'] is True for r in scored),15)
        self.assertEqual(sum(r['supplemental_only'] is True for r in scored),15)
        self.assertEqual([r['label'] for r in examples],[r['label'] for r in original])

    def test_conflicts_in_any_input_reject_before_first_score_both_versions(self):
        te=import_train_eval_with_stubs()
        for selected in schema.VERSIONS:
            for nested in (False,True):
                rows=pool(selected)[:2];other=next(v for v in schema.VERSIONS if v!=selected)
                (rows[1]['label'] if nested else rows[1])['task_label_schema_version']=other
                with self.subTest(version=selected,nested=nested),patch.object(te,'score') as scorer:
                    with self.assertRaises(ValueError):te.predict_rows(object(),rows,selected)
                    scorer.assert_not_called()

    def test_score_version_width_and_positive_unmarked_invented_primary_rows(self):
        te=import_train_eval_with_stubs()
        for selected in schema.VERSIONS:
            row=pool(selected)[0];good=score_payload(row,selected)
            with patch.object(te,'score',return_value=good):
                scored=te.predict_rows(object(),[row],selected)
            self.assertEqual(metrics(scored,schema_version=selected)['rows'],1)
            self.assertEqual(scored[0]['task_label_schema_version'],selected)
            for mutation in ('version','width','probabilities'):
                bad=copy.deepcopy(good)
                if mutation=='version':bad['task_label_schema_version']=next(v for v in schema.VERSIONS if v!=selected)
                elif mutation=='width':bad['answers']['task_type']['logits'].append(0.)
                else:bad['answers']['task_type']['probabilities']['other']=0.
                with self.subTest(version=selected,mutation=mutation),patch.object(te,'score',return_value=bad):
                    with self.assertRaises(ValueError):te.predict_rows(object(),[row],selected)
            self.assertEqual(row,pool(selected)[0])


class DatasetMetadataTests(unittest.TestCase):
    def test_both_preflights_reject_each_missing_malformed_or_conflicting_document_before_load_or_output(self):
        for target in ('label-schema.json','split-report.json'):
            for mutation in ('missing','json','null','missing-declaration','hash','order','version'):
                with self.subTest(document=target,mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);data=root/'dataset';data.mkdir();_,rp=dataset(data)
                    path=data/target
                    if mutation=='missing':path.unlink()
                    elif mutation=='json':path.write_text('INVALID JSON')
                    elif mutation=='null':path.write_text('null')
                    else:
                        doc=json.loads(path.read_text());meta=doc if target=='label-schema.json' else doc['label_schema']
                        if mutation=='missing-declaration':meta.pop('task_label_schema_version')
                        elif mutation=='hash':meta['question_sha256']='0'*64
                        elif mutation=='order':meta['classes']['task_type'].reverse()
                        else:meta['task_label_schema_version']=schema.LEGACY
                        path.write_text(json.dumps(doc))
                    ok,text=run_validation(data,rp,schema.V2);self.assertFalse(ok,text)
                    te=import_train_eval_with_stubs();args=argparse.Namespace(label_schema=schema.V2,run='never-created',seed=1,synth=False,smoke=False,
                        calibration_roles=rp,legacy_calibration=False,structural_dry_run=True,data_dir=data)
                    with patch.object(te,'ROOT',root),patch.object(te,'load_engine',side_effect=AssertionError('NO MODEL')) as loader,patch.object(te,'oversample_epoch_rows',side_effect=AssertionError('NO SAMPLER')) as sampler:
                        with self.assertRaises(ValueError):te.run(args)
                        loader.assert_not_called();sampler.assert_not_called()
                    self.assertFalse((root/'artifacts').exists())

    def test_valid_v2_and_absent_legacy_metadata_both_pass_without_test_or_model(self):
        for selected in schema.VERSIONS:
            with self.subTest(version=selected),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);_,rp=dataset(root,selected)
                ok,text=run_validation(root,rp,selected);self.assertTrue(ok,text)
                reads=[];te=import_train_eval_with_stubs(reads)
                args=argparse.Namespace(label_schema=selected,run='never-created',seed=1,synth=False,smoke=False,
                    calibration_roles=rp,legacy_calibration=False,structural_dry_run=True,data_dir=root)
                with patch.object(te,'ROOT',root),patch.object(te,'load_engine',side_effect=AssertionError('NO MODEL')) as loader,contextlib.redirect_stdout(io.StringIO()):
                    te.run(args);loader.assert_not_called()
                self.assertEqual({Path(p).name for p in reads},{'train.jsonl','calibration.jsonl'})
                self.assertFalse((root/'artifacts').exists())
                if selected==schema.LEGACY:
                    # #8 source integer is legal and distinct from task semantics.
                    (root/'split-report.json').write_text('{"label_schema_version":1}')
                    self.assertTrue(run_validation(root,rp,selected)[0])
                    for name in ('label-schema.json','split-report.json'):
                        meta=schema.metadata(schema.V2)
                        (root/name).write_text(json.dumps(meta if name=='label-schema.json' else {'label_schema':meta}))
                        self.assertFalse(run_validation(root,rp,selected)[0])
                        (root/name).unlink()


class CallerPathTests(unittest.TestCase):
    def test_external_caller_relative_data_supplement_roles_and_recipe_share_resolved_identity(self):
        launcher=load_script('labels18_launcher','train.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'dataset';data.mkdir();_,rp=dataset(data)
            supplement=root/'supplement.jsonl'
            supplement.write_bytes((ROOT/'data/synthetic-examples/labels-v2-creative-vision.jsonl').read_bytes())
            (root/'recipe.json').write_text('{"synth":false}')
            previous=Path.cwd()
            try:
                os.chdir(root)
                argv=['train.py','--run','invented-never-launched','--label-schema',schema.V2,'--data-dir','dataset',
                    '--supplemental','supplement.jsonl','--roles','dataset/roles.json','--recipe','recipe.json']
                with patch.object(sys,'argv',argv),patch.object(launcher.subprocess,'call',return_value=0) as spawn,contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.main(),0)
                spawn.assert_called_once();command=spawn.call_args.args[0]
                for flag,want in (('--data-dir',data),('--supplemental',supplement),('--calibration-roles',rp)):
                    actual=Path(command[command.index(flag)+1]);self.assertTrue(actual.is_absolute());self.assertEqual(actual,want.resolve())
                    self.assertEqual((Path(spawn.call_args.kwargs['cwd'])/actual).resolve(),want.resolve())
                self.assertNotEqual(Path(spawn.call_args.kwargs['cwd']),root)
                self.assertFalse((ROOT/'artifacts/invented-never-launched').exists())
            finally:os.chdir(previous)


class FrozenReportBoundaryTests(unittest.TestCase):
    def test_api_rejects_v2_aggregate_unscored_scored_missing_group_and_empty_paths(self):
        source={'splits':{}};v2source={**source,'label_schema':schema.metadata(schema.V2)}
        row=pool(schema.V2)[0]
        for rows in (None,[],[row],[{k:v for k,v in row.items() if k!='split_group'}],[prediction(row,schema.V2)]):
            for aggregate in (source,v2source):
                if aggregate==source and (rows is None or rows==[]):continue
                with self.subTest(metadata='label_schema' in aggregate,rows=bool(rows)),self.assertRaises(ValueError):split_report(aggregate,rows)
        for declaration in ({'task_label_schema_version':schema.V2},{'label_schema':None},{'label_schema':{'task_label_schema_version':schema.V2}}):
            with self.assertRaises(ValueError):split_report(declaration)
        legacy=copy.deepcopy(row);legacy['task_label_schema_version']=schema.LEGACY
        self.assertIn('visual',split_report(source,[legacy])['row_support']['classes']['task_type'])
        self.assertEqual(split_report(source),split_report({**source,'label_schema':schema.metadata(schema.LEGACY)}))

    def test_actual_report_cli_aggregate_unscored_and_scored_reject_before_output(self):
        cli=load_script('labels18_report_cli','report_split.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'split.json';rows=root/'rows.jsonl';output=root/'never-written.json'
            for mode in ('aggregate','unscored','scored'):
                source.write_text(json.dumps({'label_schema':schema.metadata(schema.V2),'splits':{}}))
                row=pool(schema.V2)[0];write_rows(rows,[prediction(row,schema.V2) if mode=='scored' else row])
                argv=['report_split.py','--split-report',str(source),'--output',str(output)]
                if mode!='aggregate':argv+=['--predictions',str(rows)]
                with self.subTest(mode=mode),patch.object(sys,'argv',argv),self.assertRaises(ValueError):cli.main()
                self.assertFalse(output.exists())

    def test_actual_v2_run_tail_pure_mock_never_calls_frozen_reader(self):
        # Execute the actual report tail with invented locals, not a training loop.
        # This isolates the production branch/serialization without model/weights.
        tree=ast.parse((ROOT/'train_eval.py').read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        grouped=next(n for n in run.body if isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp)
            and isinstance(n.test.operand,ast.Name) and n.test.operand.id=='legacy' and any(isinstance(x,ast.Return) for x in n.body))
        start=next(i for i,n in enumerate(grouped.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reasons' for t in n.targets))
        function=ast.FunctionDef(name='tail',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=grouped.body[start:],decorator_list=[])
        module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        te=import_train_eval_with_stubs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'dataset';data.mkdir();dataset(data)
            (root/'model-manifest.json').write_text('{}');out=root/'new-run';out.mkdir()
            env={'args':argparse.Namespace(run='INVENTED TAIL ONLY',seed=1,epochs=0,rank=8,accumulation=1),
                'ROOT':root,'data_dir':data,'output':out,'selected':schema.V2,'LEGACY':schema.LEGACY,'effective_lr':0.,'trainable':0,
                'updates':0,'start':0.,'history':[],'effective':{'label_schema':schema.metadata(schema.V2)},
                'calibration':{'status':'insufficient'},'weights':{},'vision_weights':{},'max_tier_ece':None,
                'json':json,'time':types.SimpleNamespace(perf_counter=lambda:0.),'apply_ece_veto':te.apply_ece_veto}
            with patch.object(te,'split_report',side_effect=AssertionError('NO FROZEN READER')) as frozen,contextlib.redirect_stdout(io.StringIO()):
                env['split_report']=frozen;exec(compile(module,'actual-run-report-tail','exec'),env);env['tail']();frozen.assert_not_called()
            report=json.loads((out/'report.json').read_text());diagnostic=report['frozen_test_diagnostics']
            self.assertEqual(diagnostic['status'],'unavailable');self.assertEqual(diagnostic['task_label_schema_version'],schema.V2)
            self.assertNotIn('dataset_notice',diagnostic);self.assertNotIn('row_support',diagnostic)
            self.assertEqual(metrics([prediction(pool(schema.V2)[0],schema.V2)],schema_version=schema.V2)['rows'],1)


if __name__=='__main__':unittest.main()
