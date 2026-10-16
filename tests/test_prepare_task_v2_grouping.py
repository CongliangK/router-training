"""Issue #18 grouping regressions; all prompts/labels are invented CPU fixtures."""
import contextlib, io, json, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import prepare_task_v2 as prepare
from task_label_schema import V2


class TaskV2GroupingTests(unittest.TestCase):
    def prepare_rows(self,examples,pending=()):
        # Three unrelated 20-row roots make every output split feasible. These
        # invented primary rows exercise grouping, not real support or labels.
        tasks=[{'id':f'padding-{g}-{i}','group':f'padding-{g}',
                'prompt':char*12+str(i)}
               for g,char in enumerate('xyz') for i in range(20)]
        tasks += [{'id':identity,'group':group,'prompt':prompt}
                  for identity,group,prompt in examples]
        labels=[]
        for i,task in enumerate(tasks):
            label={'id':task['id'],'tier':('fast','balanced','flagship')[i%3],
                   'needs_vision':('no','yes')[i%2],'task_type':'extraction',
                   'label_source':'expert_provisional','evidence':'prompt_only',
                   'confidence':.8,'rationale':'Invented grouping fixture only.',
                   'task_label_schema_version':V2,'label_status':'resolved'}
            if task['id'] in pending:
                label.update(label_status='adjudication',needs_vision=None,
                             adjudication_reasons=['Invented missing context.'])
            labels.append(label)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input';source.mkdir();output=root/'output'
            def write(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write(source/'tasks.jsonl',tasks)
            for n in range(1,5):write(source/f'expert-recovery-{n}.jsonl',labels[n-1::4])
            before={p.name:p.read_bytes() for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.prepare_task_v2(source,schema_version=V2,output=output)
            rows={};owners={}
            for split in ('train','calibration','test'):
                for row in prepare.load(output/f'{split}.jsonl'):
                    self.assertNotIn(row['id'],rows)
                    rows[row['id']]=row;owners[row['id']]=split
            self.assertEqual(set(rows),{r['id'] for r in tasks}-set(pending))
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            report=json.loads((output/'split-report.json').read_text())
            self.assertTrue(report['group_disjoint'])
            self.assertEqual({r['id'] for r in prepare.load(output/'adjudication.jsonl')},set(pending))
            return rows,owners,report

    def assert_linked(self,rows,owners,*identities):
        self.assertEqual(len({rows[i]['split_group'] for i in identities}),1)
        self.assertEqual(len({owners[i] for i in identities}),1)

    def test_identical_short_prompts_link_across_roots(self):
        for prompt in ('q','ok','yes','abcd'):
            with self.subTest(prompt=prompt):
                self.assertEqual(prepare.grams(prompt),set())
                rows,owners,report=self.prepare_rows([('a','root-a',prompt),('b','root-b',prompt)])
                self.assert_linked(rows,owners,'a','b')
                self.assertEqual(report['cross_root_near_duplicate_edges'],1)
                self.assertEqual(report['independent_groups'],4)

    def test_different_short_prompts_do_not_link_empty_grams(self):
        rows,_,report=self.prepare_rows([('a','root-a','ok'),('b','root-b','no'),
                                         ('c','root-c','abcd'),('d','root-d','abce')])
        self.assertEqual(len({rows[i]['split_group'] for i in 'abcd'}),4)
        self.assertEqual(report['cross_root_near_duplicate_edges'],0)
        self.assertEqual(report['independent_groups'],7)

    def test_case_and_whitespace_normalization_precedes_short_comparison(self):
        rows,owners,report=self.prepare_rows([('a','root-a',' A\u2003B\tC\nD '),
                                              ('b','root-b','a b c d')])
        self.assert_linked(rows,owners,'a','b')
        self.assertEqual(report['cross_root_near_duplicate_edges'],1)

    def test_long_prompt_jaccard_and_short_long_separation_are_preserved(self):
        prompt='abcdefghijklmnopqrstuvwx'
        rows,owners,report=self.prepare_rows([('a','root-a',prompt),('b','root-b',prompt+'y'),
                                              ('c','root-c','abc'),('d','root-d','abcde')])
        self.assert_linked(rows,owners,'a','b')
        self.assertNotEqual(rows['c']['split_group'],rows['d']['split_group'])
        self.assertEqual(report['cross_root_near_duplicate_edges'],1)

    def test_short_edges_merge_transitively_through_shared_root(self):
        rows,owners,report=self.prepare_rows([('a','root-a','ok'),('b1','root-b','ok'),
                                              ('b2','root-b','yes'),('c','root-c','yes')])
        self.assert_linked(rows,owners,'a','b1','b2','c')
        self.assertEqual(report['cross_root_near_duplicate_edges'],2)
        self.assertEqual(report['independent_groups'],4)

    def test_pending_short_rows_still_bridge_active_roots(self):
        rows,owners,report=self.prepare_rows([('a','root-a','ok'),('b1','root-b','OK'),
                                              ('b2','root-b','YES'),('c','root-c','yes')],
                                             pending=('b1','b2'))
        self.assert_linked(rows,owners,'a','c')
        self.assertEqual(report['cross_root_near_duplicate_edges'],2)
        self.assertEqual(report['independent_groups'],4)
        self.assertEqual(report['label_review']['adjudication_or_excluded'],2)


if __name__=='__main__':unittest.main()
