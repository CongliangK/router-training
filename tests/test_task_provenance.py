"""Invented fixtures only: v2 primary supervision checks both provenance boundaries."""
import copy, itertools, json, tempfile, unittest
from pathlib import Path
from test_task_split_v2 import inputs
from test_split_v2 import write_inputs
from task_label_schema import V2, LEGACY, partition_labels, validate_supervised_rows, supplemental_rows
from prepare_task_v2 import prepare_task_v2
from split_task_v2 import split_task_v2, prepare_task_split_v2

FLAGS=('synthetic','supplemental_only')


def joined(task,label):
    return {**task,'label':label}


class ProvenanceTests(unittest.TestCase):
    def test_every_true_location_combination_rejects_primary_without_mutation(self):
        tasks,labels=inputs();original=joined(tasks[0],labels[0])
        # Includes individual flags, both, conflicting false/true declarations,
        # and flags split between task and label. False must never mask true.
        for values in itertools.product((None,False,True),repeat=4):
            if True not in values:continue
            with self.subTest(values=values):
                row=copy.deepcopy(original)
                for (obj,key),value in zip(((row,FLAGS[0]),(row,FLAGS[1]),
                                            (row['label'],FLAGS[0]),(row['label'],FLAGS[1])),values):
                    if value is not None:obj[key]=value
                before=copy.deepcopy(row)
                with self.assertRaisesRegex(ValueError,'primary supervision'):
                    validate_supervised_rows([row],V2)
                self.assertEqual(row,before)

    def test_absent_false_and_uninterpreted_audit_metadata_stay_valid(self):
        tasks,labels=inputs()
        for values in itertools.product((None,False),repeat=4):
            row=joined(copy.deepcopy(tasks[0]),copy.deepcopy(labels[0]))
            for (obj,key),value in zip(((row,FLAGS[0]),(row,FLAGS[1]),
                                       (row['label'],FLAGS[0]),(row['label'],FLAGS[1])),values):
                if value is not None:obj[key]=value
            row['label']['provenance']={'kind':'synthetic','purpose':'opaque issuance metadata'}
            validate_supervised_rows([row],V2)
            self.assertEqual(partition_labels([row['label']],V2),([row['label']],[]))

    def test_malformed_supervision_flags_reject_at_each_boundary(self):
        tasks,labels=inputs()
        for location,key,value in itertools.product(('task','label'),FLAGS,(None,0,1,'false','true',[],{})):
            with self.subTest(location=location,key=key,value=value):
                row=joined(copy.deepcopy(tasks[0]),copy.deepcopy(labels[0]))
                (row if location=='task' else row['label'])[key]=value
                for allow in (False,True):
                    with self.assertRaisesRegex(ValueError,'must be boolean'):
                        validate_supervised_rows([row],V2,allow_supplemental=allow)

    def test_partition_rejects_active_but_retains_reasoned_pending_flags(self):
        _,labels=inputs()
        for flags in ({'synthetic':True},{'supplemental_only':True},dict.fromkeys(FLAGS,True)):
            label={**labels[0],**flags}
            with self.assertRaises(ValueError):partition_labels([label],V2)
            for status in ('adjudication','excluded'):
                pending={**label,'label_status':status,'adjudication_reasons':['invented review pending']}
                before=copy.deepcopy(pending)
                self.assertEqual(partition_labels([pending],V2),([],[pending]))
                self.assertEqual(pending,before)
                with self.assertRaises(ValueError):validate_supervised_rows([{'task_label_schema_version':V2,'label':pending}],V2)

    def test_both_preparers_reject_before_output_and_preserve_inputs(self):
        for preparer,location,flags in itertools.product((prepare_task_v2,prepare_task_split_v2),
                ('task','label','mixed'),({'synthetic':True},{'supplemental_only':True},dict.fromkeys(FLAGS,True))):
            with self.subTest(preparer=preparer.__name__,location=location,flags=flags),tempfile.TemporaryDirectory() as temp:
                tasks,labels=inputs()
                if location=='mixed':
                    tasks[0]['synthetic']=True;labels[0]['supplemental_only']=True
                else:(tasks[0] if location=='task' else labels[0]).update(flags)
                root=Path(temp)/'input';root.mkdir();write_inputs(root,tasks,labels)
                before={p.name:p.read_bytes() for p in root.iterdir()};out=Path(temp)/'new'
                with self.assertRaises(ValueError):preparer(root,output=out,schema_version=V2)
                self.assertFalse(out.exists())
                self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_pending_flagged_bridge_preserves_closure_without_support(self):
        for status in ('adjudication','excluded'):
            tasks,labels=inputs(15,2)
            for i,prompt in ((0,' A B '),(2,'ab'),(3,'CD'),(4,' c d ')):tasks[i]['prompt']=prompt
            for i in (2,3):
                tasks[i].update(synthetic=True,supplemental_only=True)
                labels[i].update(label_status=status,adjudication_reasons=['invented pending'],
                                 synthetic=True,supplemental_only=True,needs_vision=None)
            splits,report,pending=split_task_v2(tasks,labels,schema_version=V2,attempts=100)
            self.assertIsNotNone(splits);self.assertEqual(pending,labels[2:4])
            rows={r['id']:r for values in splits.values() for r in values}
            self.assertEqual(len(rows),len(tasks)-2)
            self.assertEqual(rows[tasks[0]['id']]['split_group'],rows[tasks[4]['id']]['split_group'])
            self.assertNotIn(tasks[2]['id'],rows);self.assertNotIn(tasks[3]['id'],rows)
            self.assertEqual(report['grouping']['cross_root_near_duplicate_edges'],2)
            for values in splits.values():validate_supervised_rows(values,V2)

    def test_explicit_supplemental_loading_preserves_flags(self):
        path=Path(__file__).resolve().parents[1]/'data/synthetic-examples/labels-v2-creative-vision.jsonl'
        rows=supplemental_rows([path],V2)
        rows[0]['label'].update(synthetic=True,supplemental_only=True)
        before=copy.deepcopy(rows)
        validate_supervised_rows(rows,V2,allow_supplemental=True)
        self.assertEqual(rows,before)
        with self.assertRaises(ValueError):validate_supervised_rows(rows,V2)
        rows[0].pop('supplemental_only')
        with self.assertRaises(ValueError):validate_supervised_rows(rows,V2,allow_supplemental=True)

    def test_supplemental_optin_and_prior_exclusion_at_every_boundary(self):
        import calibration_groups as cg
        tasks,labels=inputs();schema=cg.default_business_schema(schema_version=V2)
        primary=[joined(copy.deepcopy(task),copy.deepcopy(label)) for task,label in zip(tasks[:3],labels[:3])]
        for row,tier in zip(primary,('fast','balanced','flagship')):row['label']['tier']=tier
        expected=cg.compute_prior(primary,'tier',schema)
        for values in itertools.product((None,False,True),repeat=4):
            if True not in values:continue
            with self.subTest(values=values):
                supplements=copy.deepcopy(primary)
                for row in supplements:
                    row['id']='supplement-'+row['id']
                    for (obj,key),value in zip(((row,FLAGS[0]),(row,FLAGS[1]),
                                               (row['label'],FLAGS[0]),(row['label'],FLAGS[1])),values):
                        if value is not None:obj[key]=value
                before=copy.deepcopy(supplements)
                if values[:2]==(True,True):
                    validate_supervised_rows(supplements,V2,allow_supplemental=True)
                    actual=cg.compute_prior(primary+supplements,'tier',schema)
                    self.assertEqual(actual['counts'],expected['counts'])
                    self.assertEqual(actual['probabilities'],expected['probabilities'])
                    self.assertEqual(actual['excluded_synthetic'],3)
                    # Supplements cannot repair missing primary class support,
                    # including when some other primary classes remain.
                    for real in ([],primary[:2]):
                        with self.assertRaisesRegex(cg.PlanError,'without unique real train support'):
                            cg.compute_prior(real+supplements,'tier',schema)
                else:
                    with self.assertRaisesRegex(ValueError,'top-level synthetic and supplemental_only'):
                        validate_supervised_rows(supplements,V2,allow_supplemental=True)
                    for real in ([],primary[:2],primary):
                        with self.assertRaises(cg.PlanError):cg.compute_prior(real+supplements,'tier',schema)
                self.assertEqual(supplements,before)

    def test_legacy_provenance_handling_is_unchanged(self):
        # Legacy readers own legality; the v2 boundary must not reinterpret it.
        for value in (True,False,None,1,'false'):
            row={'synthetic':value,'supplemental_only':value,
                 'label':{'synthetic':value,'supplemental_only':value}}
            validate_supervised_rows([row],LEGACY)
            self.assertEqual(partition_labels([row['label']],LEGACY),([row['label']],[]))


if __name__=='__main__':unittest.main()
