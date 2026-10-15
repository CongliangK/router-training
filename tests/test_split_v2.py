"""Purely synthetic CPU acceptance for #24; never opens private data or weights."""
import contextlib, io, json, os, random, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import prepare, prepare_v2, split_v2
import split_v2 as module

ROOT=Path(__file__).resolve().parents[1]


def fixture(groups=12,size=4):
    rng=random.Random(24)
    tasks=[{'id':f't-{g:02}-{i:02}','group':f'root-{g:02}',
            'prompt':''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=120)), 'synthetic':True}
           for g in range(groups) for i in range(size)]
    labels=[{'id':r['id'],'tier':('fast','balanced','flagship')[i%3],
             'needs_vision':('yes','no','uncertain')[i%3],
             'task_type':('creative','visual','implementation')[i%3],
             'label_source':'expert_provisional','evidence':'prompt_only','confidence':1.,
             'rationale':'Entirely synthetic test label.'} for i,r in enumerate(tasks)]
    return tasks,labels


def write_inputs(root,tasks,labels):
    def write(name,rows):
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    write('tasks.jsonl',tasks)
    for i in range(4):write(f'expert-recovery-{i+1}.jsonl',labels[i::4])


class SplitV2Tests(unittest.TestCase):
    def assert_atomic(self,tasks,splits):
        rows=[r for values in splits.values() for r in values]
        self.assertEqual({r['id'] for r in tasks},{r['id'] for r in rows})
        self.assertEqual(len(rows),len(tasks))
        roots={};components={}
        for name,values in splits.items():
            for row in values:
                self.assertEqual(roots.setdefault(row['group'],name),name)
                self.assertEqual(components.setdefault(row['split_group'],name),name)

    def test_deterministic_even_with_reordered_inputs(self):
        tasks,labels=fixture()
        first=split_v2(tasks,labels,attempts=150)
        second=split_v2(list(reversed(tasks)),list(reversed(labels)),attempts=150)
        self.assertEqual(first,second)
        self.assertEqual(first[1]['status'],'feasible');self.assert_atomic(tasks,first[0])
        json.dumps(first,allow_nan=False)

    def test_every_selected_split_meets_fixed_cap_and_reports_deviations(self):
        tasks,labels=fixture()
        splits,report=split_v2(tasks,labels,attempts=300)
        self.assertEqual(report['targets']['max_group_share'],.35)
        for name,stats in report['splits'].items():
            self.assertLessEqual(stats['max_group_share'],.35)
            self.assertTrue(stats['constraint_satisfied']);self.assertEqual(stats['max_group_share_excess'],0)
            sizes=stats['group_sizes'];n=len(splits[name])
            self.assertAlmostEqual(stats['kish_group_size_concentration'],n*n/sum(s*s for s in sizes))
            self.assertAlmostEqual(stats['deviation_from_target']['row_fraction'],n/len(tasks)-report['targets']['row_fractions'][name])
            for field in module.CLASSES:
                for label,actual in stats['proportions'][field].items():
                    self.assertAlmostEqual(stats['deviation_from_target'][field][label],actual-report['targets']['class_proportions'][field][label])
        self.assertIn('vision_yes',report['objective']['weighted_terms'])
        self.assertIn('task_type',report['objective']['weighted_terms'])
        self.assertAlmostEqual(report['objective']['total'],sum(report['objective']['weighted_terms'].values()))
        self.assertTrue(any('not independent effective sample size' in s for s in report['limitations']))

    def test_cap_boundary_is_exact_and_not_rounded_down(self):
        tasks,labels=fixture(groups=9,size=1)
        targets={k:{c:sum(r[k]==c for r in labels)/len(labels) for c in values} for k,values in module.CLASSES.items()}
        def vector(n):
            return {'rows':n,**{k:{c:n if c==values[0] else 0 for c in values} for k,values in module.CLASSES.items()}}
        assignment=(0,0,0,1,1,1,2,2,2)
        vectors=[vector(n) for n in (7,7,6,7,7,6,7,7,6)]
        valid,_,stats,_=module._summaries(vectors,assignment,60,targets)
        self.assertTrue(valid);self.assertEqual(stats['train']['max_group_share'],.35)
        vectors[0]=vector(8);vectors[2]=vector(5)
        valid,_,stats,_=module._summaries(vectors,assignment,60,targets)
        self.assertFalse(valid);self.assertGreater(stats['train']['max_group_share_excess'],0)

    def test_proportion_objective_not_boolean_presence(self):
        targets={'tier':{'fast':1.,'balanced':0.,'flagship':0.},
                 'needs_vision':{'yes':.5,'no':.5,'uncertain':0.},
                 'task_type':{'creative':.5,'visual':.5,**{c:0. for c in module.CLASSES['task_type'] if c not in ('creative','visual')}}}
        def vector(yes,creative):
            return {'rows':10,'tier':{'fast':10,'balanced':0,'flagship':0},
                    'needs_vision':{'yes':yes,'no':10-yes,'uncertain':0},
                    'task_type':{c:creative if c=='creative' else 10-creative if c=='visual' else 0 for c in module.CLASSES['task_type']}}
        assignment=tuple(i//3 for i in range(9))
        balanced=[vector(5,5) for _ in range(9)]
        skewed=[vector(1,1) for _ in range(3)]+[vector(5,5) for _ in range(3)]+[vector(9,9) for _ in range(3)]
        _,a,_,terms_a=module._summaries(balanced,assignment,90,targets)
        _,b,_,terms_b=module._summaries(skewed,assignment,90,targets)
        self.assertGreater(b,a);self.assertGreater(terms_b['vision_yes'],terms_a['vision_yes'])
        self.assertGreater(terms_b['task_type'],terms_a['task_type'])

    def test_transitive_cross_root_links_and_whole_root_closure(self):
        tasks,labels=fixture(groups=15,size=2)
        # Change opposite ends in two steps: each edge passes .88, A--C does not.
        text=''.join(random.Random(246).choices('abcdefghijklmnopqrstuvwxyz',k=120))
        a=text;b='01234'+text[5:];c='01234'+text[5:-5]+'56789'
        tasks[0]['prompt']=a;tasks[2]['prompt']=b;tasks[4]['prompt']=c
        def jac(x,y):
            gx={x[i:i+5] for i in range(len(x)-4)};gy={y[i:i+5] for i in range(len(y)-4)}
            return len(gx&gy)/len(gx|gy)
        threshold=.88
        self.assertGreaterEqual(jac(a,b),threshold);self.assertGreaterEqual(jac(b,c),threshold)
        self.assertLess(jac(a,c),threshold)
        splits,report=split_v2(tasks,labels,similarity=threshold,attempts=500)
        self.assertIsNotNone(splits);self.assert_atomic(tasks,splits)
        rows={r['id']:r for values in splits.values() for r in values}
        self.assertEqual(len({rows[r['id']]['split_group'] for r in tasks[:6]}),1)
        self.assertEqual(report['grouping']['observed_components'],13)
        self.assertEqual(report['grouping']['cross_root_near_duplicate_edges'],2)

    def test_short_exact_duplicates_link_but_unrelated_short_prompts_do_not(self):
        tasks,labels=fixture(groups=12,size=1)
        tasks[0]['prompt']=' Ab ';tasks[1]['prompt']='aB';tasks[2]['prompt']='xy'
        splits,report=split_v2(tasks,labels,attempts=100)
        self.assertEqual(report['grouping']['observed_components'],11)
        self.assertEqual(report['grouping']['cross_root_near_duplicate_edges'],1)
        self.assertIsNotNone(splits);self.assert_atomic(tasks,splits)

    def test_single_group_is_honestly_infeasible(self):
        tasks,labels=fixture(groups=1,size=5)
        splits,report=split_v2(tasks,labels)
        self.assertIsNone(splits);self.assertEqual(report['status'],'infeasible')
        self.assertTrue(report['feasibility']['proven_infeasible'])
        self.assertEqual(report['diagnostic_candidate']['splits']['train']['kish_group_size_concentration'],1.)
        self.assertIsNone(report['diagnostic_candidate']['splits']['test']['kish_group_size_concentration'])
        self.assertEqual(report['splits'],{})

    def test_atomic_boulder_cannot_be_split_or_cap_relaxed(self):
        tasks,labels=fixture(groups=12,size=2)
        for row in tasks[:12]:row['group']='boulder'
        splits,report=split_v2(tasks,labels)
        self.assertIsNone(splits);self.assertEqual(report['status'],'infeasible')
        self.assertIn('ENTIRE corpus',report['feasibility']['reason'])
        self.assertEqual(report['targets']['max_group_share'],.35)
        self.assertGreater(report['diagnostic_candidate']['splits']['train']['max_group_share'],.35)

    def test_exhaustive_detects_impossible_without_simple_lower_bound(self):
        tasks,labels=fixture(groups=9,size=1)
        # 8 singleton groups plus a group of size 2: no partition into 3 triples
        # can place the size-2 group below .35, though its corpus share is .2.
        extra={**tasks[0],'id':'extra','prompt':'a completely distinct synthetic string'}
        tasks.append(extra);labels.append({**labels[0],'id':'extra'})
        splits,report=split_v2(tasks,labels)
        self.assertIsNone(splits);self.assertEqual(report['status'],'infeasible')
        self.assertEqual(report['search']['evaluated'],3**9)
        self.assertTrue(report['feasibility']['proven_infeasible'])

    def test_search_exhaustion_never_claims_proven_infeasibility(self):
        tasks,labels=fixture()
        with patch.object(module,'_candidates',return_value=iter([(0,)*12])):
            splits,report=split_v2(tasks,labels,attempts=1)
        self.assertIsNone(splits);self.assertEqual(report['status'],'search_exhausted')
        self.assertFalse(report['feasibility']['proven_infeasible'])
        self.assertIn('NOT proven',report['feasibility']['reason'])
        self.assertEqual(report['targets']['max_group_share'],.35)

    def test_rare_absent_classes_do_not_manufacture_support(self):
        tasks,labels=fixture(groups=12,size=2)
        for label in labels:label.update(tier='balanced',task_type='implementation',needs_vision='no')
        labels[0].update(tier='fast',task_type='creative',needs_vision='yes')
        splits,report=split_v2(tasks,labels,attempts=100)
        self.assertIsNotNone(splits);self.assert_atomic(tasks,splits)
        support=report['class_component_support']
        self.assertEqual(support['tier']['fast']['components'],1)
        self.assertTrue(support['tier']['fast']['fewer_than_three_components'])
        self.assertEqual(support['tier']['flagship']['rows'],0)
        self.assertEqual(sum(s['counts']['needs_vision']['yes']>0 for s in report['splits'].values()),1)

    def test_validation_rejects_duplicates_missing_labels_and_invalid_options(self):
        tasks,labels=fixture()
        for ts,ls in [(tasks+[tasks[0]],labels),(tasks,labels+[labels[0]]),(tasks,labels[:-1])]:
            with self.assertRaises(ValueError):split_v2(ts,ls)
        for options in [{'attempts':0},{'seed':True},{'similarity':float('nan')},{'similarity':0}]:
            with self.assertRaises(ValueError):split_v2(tasks,labels,**options)
        labels[0]['confidence']=float('nan')
        with self.assertRaises(ValueError):split_v2(tasks,labels)

    def test_new_bundle_never_overwrites_any_existing_output(self):
        tasks,labels=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_inputs(root,tasks,labels)
            historical={name:b'historical sentinel\n' for name in ('train.jsonl','calibration.jsonl','test.jsonl','split-report.json','expert-labels.jsonl')}
            for name,body in historical.items():(root/name).write_bytes(body)
            report=prepare_v2(root,root/'new-bundle',attempts=100)
            self.assertEqual(report['status'],'feasible')
            self.assertTrue((root/'new-bundle'/'train.split-v2.jsonl').is_file())
            before={p.name:p.read_bytes() for p in (root/'new-bundle').iterdir()}
            with self.assertRaises(FileExistsError):prepare_v2(root,root/'new-bundle')
            self.assertEqual(before,{p.name:p.read_bytes() for p in (root/'new-bundle').iterdir()})
            for name,body in historical.items():self.assertEqual((root/name).read_bytes(),body)
            for target in [root,root/'train.jsonl']:
                with self.assertRaises(FileExistsError):prepare_v2(root,target)
            (root/'dangling').symlink_to(root/'missing')
            with self.assertRaises(FileExistsError):prepare_v2(root,root/'dangling')
            self.assertFalse((root/'missing').exists())

    def test_versioned_bundle_bytes_are_reproducible(self):
        tasks,labels=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_inputs(root,tasks,labels)
            prepare_v2(root,root/'one',attempts=150)
            write_inputs(root,list(reversed(tasks)),list(reversed(labels)))
            prepare_v2(root,root/'two',attempts=150)
            first={p.name:p.read_bytes() for p in (root/'one').iterdir()}
            second={p.name:p.read_bytes() for p in (root/'two').iterdir()}
            self.assertEqual(first,second)

    def test_infeasible_writes_only_new_versioned_diagnostic(self):
        tasks,labels=fixture(groups=1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_inputs(root,tasks,labels)
            report=prepare_v2(root,root/'diagnostic')
            self.assertEqual(report['status'],'infeasible')
            self.assertEqual([p.name for p in (root/'diagnostic').iterdir()],['split-report.split-v2.json'])
            self.assertEqual(report['outputs']['splits'],{})

    def test_prepare_cli_requires_explicit_opt_in_paths_and_returns_status(self):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'}
        def run(*args):return subprocess.run([sys.executable,str(ROOT/'prepare.py'),*args],env=env,capture_output=True,text=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tasks,labels=fixture(groups=1);write_inputs(root,tasks,labels)
            result=run('--split-version','split-v2')
            self.assertNotEqual(result.returncode,0);self.assertIn('explicit --data and --output',result.stderr)
            result=run('--output',str(root/'wrong'))
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'wrong').exists())
            result=run('--split-version','split-v2','--data',str(root),'--output',str(root/'new'))
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'],'infeasible')
            result=run('--split-version','split-v2','--data',str(root),'--output',str(root/'new'))
            self.assertNotEqual(result.returncode,0);self.assertIn('already exists',result.stderr)
            tasks,labels=fixture(groups=12,size=1);write_inputs(root,tasks,labels)
            result=run('--split-version','split-v2','--data',str(root),'--output',str(root/'feasible'))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'],'feasible')
            self.assertTrue((root/'feasible'/'test.split-v2.jsonl').is_file())

    def test_synthetic_old_new_comparison_uses_unchanged_legacy(self):
        # 12 groups x 10 permits the original >=20 rows/split rule. This is not
        # a reconstruction of private topology, nor a promised vision improvement.
        tasks,labels=fixture(groups=12,size=10)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_inputs(root,tasks,labels)
            with contextlib.redirect_stdout(io.StringIO()):prepare(root)
            old={name:module._vector([json.loads(line) for line in (root/f'{name}.jsonl').read_text().splitlines()]) for name in module.NAMES}
            old_bytes={name:(root/f'{name}.jsonl').read_bytes() for name in module.NAMES}
            report=prepare_v2(root,root/'v2',attempts=300)
            self.assertEqual(report['status'],'feasible')
            self.assertEqual(sum(v['rows'] for v in old.values()),sum(v['rows'] for v in report['splits'].values()))
            for name in module.NAMES:
                self.assertEqual(old_bytes[name],(root/f'{name}.jsonl').read_bytes())
                self.assertLessEqual(report['splits'][name]['max_group_share'],.35)
            self.assertTrue(any(old[name]['rows']!=report['splits'][name]['rows'] for name in module.NAMES))


if __name__=='__main__':unittest.main()
