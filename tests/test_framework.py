import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from extract import redact
from policy import select,DEFAULT_ROUTES
from outcomes import import_outcomes
from synthetic_data import synthetic_splits, SPLIT_NAMES


def prediction(tier='fast',vision='no',confidence=.99,truncated=False):
    return {'answers':{'tier':{'choice':tier,'confidence':confidence},'vision':{'choice':vision,'confidence':.99}},'context_truncated':truncated}


def fixtures():
    return [{'task_id':'task','provider':route['provider'],'model':route['model'],'trial':trial,'passed':True,'evidence_kind':'acceptance_tests','acceptance_spec_sha256':'a'*64,'artifact_sha256':'b'*64,'cost':i+1.,'latency_ms':10.,'input_tokens':20,'output_tokens':10} for i,route in enumerate(DEFAULT_ROUTES) for trial in range(3)]

class FrameworkTests(unittest.TestCase):
    def test_redaction_removes_credentials_and_coordination_suffix(self):
        text='token sk-abcdefghijk123456789\nYour parent agent id is "SECRET". Before you finish...'
        clean=redact(text);self.assertNotIn('abcdefghijk',clean);self.assertNotIn('SECRET',clean)
    def test_fast_text_recommends_fast(self):
        result=select(prediction(),DEFAULT_ROUTES);self.assertEqual(result['recommendation']['model'],'glm-5.3-flash')
    def test_actual_image_signal_cannot_be_overridden_by_classifier(self):
        result=select(prediction(),DEFAULT_ROUTES,has_image=True);self.assertTrue(result['needs_vision']);self.assertEqual(result['recommendation']['model'],'glm-5.3-flash')
    def test_hard_visual_uses_vision_capable_flagship(self):
        result=select(prediction('flagship','yes'),DEFAULT_ROUTES);self.assertEqual(result['recommendation']['model'],'gpt-6.1-sol')
    def test_absent_flagship_vision_still_refuses_unsafe_downgrade(self):
        routes=[{**route,'vision':False} if route['tier']=='flagship' else route for route in DEFAULT_ROUTES]
        result=select(prediction('flagship','yes'),routes);self.assertIsNone(result['recommendation'])
    def test_uncertain_visual_uses_vision_capable_route(self):
        result=select(prediction('fast','yes',confidence=.5),DEFAULT_ROUTES);self.assertEqual(result['recommendation']['model'],'glm-5.3-flash')
        routes=[{**route,'vision':False} if route['tier']=='fast' else route for route in DEFAULT_ROUTES]
        result=select(prediction('fast','yes',confidence=.5),routes);self.assertEqual(result['recommendation']['model'],'gpt-6.1-sol')
    def test_low_confidence_keeps_argmax_tier(self):
        result=select(prediction(confidence=.5),DEFAULT_ROUTES)
        self.assertEqual(result['required_tier'],'fast');self.assertEqual(result['recommendation']['model'],'glm-5.3-flash')
        self.assertIn('low-tier-confidence-effort-compensation-only',result['reasons'])
    def test_truncation_alone_escalates_to_flagship(self):
        result=select(prediction(truncated=True),DEFAULT_ROUTES);self.assertEqual(result['required_tier'],'flagship');self.assertEqual(result['recommendation']['model'],'gpt-6.1-sol')
        self.assertIn('context-truncated-flagship',result['reasons'])
    def test_measured_label_requires_complete_new_trials(self):
        result=import_outcomes([{'id':'task'}],fixtures());self.assertEqual(result['labels'][0]['tier'],'fast')
        result=import_outcomes([{'id':'task'}],fixtures()[:-1]);self.assertEqual(result['labels'],[]);self.assertEqual(result['pending'][0]['reason'],'incomplete-candidate-trials')
    def test_duplicate_trial_or_simulated_outcome_is_rejected(self):
        records=fixtures()
        with self.assertRaises(ValueError):import_outcomes([{'id':'task'}],records+[records[0]])
        records[0]['simulated']=True
        with self.assertRaises(ValueError):import_outcomes([{'id':'task'}],records)
    def test_measured_effort_is_preserved_and_mixed_trials_are_rejected(self):
        records=fixtures()
        for row in records:row['reasoning_effort']='low'
        result=import_outcomes([{'id':'task'}],records)
        self.assertEqual(result['labels'][0]['reasoning_effort'],'low')
        self.assertTrue(all(row['reasoning_effort']=='low' for row in result['candidate_metrics']))
        records[1]['reasoning_effort']='high'
        with self.assertRaisesRegex(ValueError,'Reasoning effort changed'):import_outcomes([{'id':'task'}],records)
        records[1]['reasoning_effort']=None
        with self.assertRaisesRegex(ValueError,'Reasoning effort changed'):import_outcomes([{'id':'task'}],records)
        records[1]['reasoning_effort']=' '
        with self.assertRaisesRegex(ValueError,'Invalid measured'):import_outcomes([{'id':'task'}],records)
    def test_changed_acceptance_criteria_is_rejected(self):
        records=fixtures();records[1]['acceptance_spec_sha256']='c'*64
        with self.assertRaises(ValueError):import_outcomes([{'id':'task'}],records)
    def test_generated_split_groups_and_ids_are_disjoint(self):
        tasks,splits,report=synthetic_splits()
        for a,b in [('train','calibration'),('train','test'),('calibration','test')]:
            self.assertFalse({r['id'] for r in splits[a]}&{r['id'] for r in splits[b]})
            self.assertFalse({r['split_group'] for r in splits[a]}&{r['split_group'] for r in splits[b]})
        self.assertEqual({name:len(rows) for name,rows in splits.items()}, {'train':80,'calibration':20,'test':20})
        self.assertEqual(sum(map(len,splits.values())),120)
        rows=[r for split in splits.values() for r in split]
        self.assertEqual({r['id'] for r in rows},{r['id'] for r in tasks})
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(report['independent_groups'],12)
        self.assertTrue(report['group_disjoint'])
        self.assertEqual(report['cross_root_near_duplicate_edges'],0)
        for name in SPLIT_NAMES:
            self.assertEqual(report['splits'][name]['rows'],len(splits[name]))
            self.assertEqual(set(report['splits'][name]['groups']),{r['split_group'] for r in splits[name]})
        self.assertTrue(all(r['synthetic'] for r in rows))

    @unittest.skipUnless(all((Path(__file__).resolve().parents[1]/'data'/f'{name}.jsonl').is_file() for name in SPLIT_NAMES),
                         'Optional private archive validation: real split files are not distributed')
    def test_actual_split_groups_and_ids_are_disjoint(self):
        root=Path(__file__).resolve().parents[1]/'data';splits={name:[json.loads(s) for s in (root/f'{name}.jsonl').read_text().splitlines()] for name in ['train','calibration','test']}
        for a,b in [('train','calibration'),('train','test'),('calibration','test')]:
            self.assertFalse({r['id'] for r in splits[a]}&{r['id'] for r in splits[b]})
            self.assertFalse({r['split_group'] for r in splits[a]}&{r['split_group'] for r in splits[b]})
        self.assertEqual({name:len(rows) for name,rows in splits.items()}, {'train':469,'calibration':95,'test':104})
        self.assertEqual(sum(map(len,splits.values())),668)
    def test_quality_gate_does_not_pass_from_expert_imitation(self):
        gate=json.loads((Path(__file__).resolve().parents[1]/'artifacts/real-history-v3c/quality-gate.json').read_text());self.assertFalse(gate['passed']);self.assertEqual(gate['mode'],'shadow')

if __name__=='__main__':unittest.main()
