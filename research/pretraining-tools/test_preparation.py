"""CPU-only checks; failures never display private rows, IDs, prompts or roots."""
import importlib.util
import json
from pathlib import Path
import random
import unittest

HERE = Path(__file__).resolve().parent

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

audit = module('audit_checks', 'audit_local.py')
build = module('build_checks', 'build_preparation.py')
readiness = module('readiness_checks', 'check_readiness.py')

class ToyOnly(unittest.TestCase):
    def test_unknown_values_never_escape_aggregate(self):
        marker = 'INVENTED_PRIVATE_SENTINEL'
        result = audit.summary([{'id':marker,'group':marker,'split_group':marker,'prompt':marker,
                                 'label':{'tier':marker,'needs_vision':marker,'task_type':marker}}])
        self.assertTrue(marker not in json.dumps(result), 'unknown values escaped counts-only output')
        self.assertTrue(result['labels']['tier'].get('other') == 1)
    def test_test_split_path_refused(self):
        with self.assertRaises(ValueError): audit.safe_path('data/test.jsonl')
    def test_external_path_refused(self):
        with self.assertRaises(ValueError): audit.safe_path('../../outside.jsonl')
    def test_source_missing_components_not_independence(self):
        result = audit.summary([{'id':'toy','group':'toy-root','label':{'tier':'fast'}}])
        self.assertTrue(result['missing_component_rows'] == 1)
    def test_spread_is_seeded_and_root_balanced(self):
        tasks = {f'toy-{g}-{i}':{'group':f'root-{g}'} for g in range(4) for i in range(10)}
        a = build.spread(set(tasks), tasks, 16, random.Random(12))
        b = build.spread(set(tasks), tasks, 16, random.Random(12))
        self.assertTrue(a == b and len(set(a)) == 16)
        counts = [sum(tasks[key]['group'] == f'root-{g}' for key in a) for g in range(4)]
        self.assertTrue(counts == [4,4,4,4])
    def test_five_positive_roots_cannot_supply_three_disjoint_two_root_roles(self):
        import itertools
        roots = set(range(5))
        possible = any(not (set(a)&set(b) or set(a)&set(c) or set(b)&set(c))
                       for a in itertools.combinations(roots,2)
                       for b in itertools.combinations(roots,2)
                       for c in itertools.combinations(roots,2))
        self.assertFalse(possible)

class RepairRegressions(unittest.TestCase):
    def resolved(self, identifier):
        return {'review_id':identifier, 'label_status':'resolved', 'tier':'fast',
                'needs_vision':'no', 'task_type':'review', 'reviewer':'toy-reviewer',
                'vision_evidence':'toy evidence', 'action_evidence':'toy evidence'}
    def test_duplicates_do_not_complete_expected_cases(self):
        valid, completed = readiness.decision_progress({'a','b'}, [self.resolved('a')]*3)
        self.assertFalse(valid)
        self.assertTrue(completed == 0)
    def test_foreign_ids_do_not_complete_expected_cases(self):
        valid, completed = readiness.decision_progress({'a','b'}, [self.resolved('x'),self.resolved('y')])
        self.assertFalse(valid)
        self.assertTrue(completed == 0)
    def test_unique_expected_cases_complete_once(self):
        valid, completed = readiness.decision_progress({'a','b'}, [self.resolved('a'),self.resolved('b')])
        self.assertTrue(valid and completed == 2)
    def test_malformed_identity_is_not_counted(self):
        valid, completed = readiness.decision_progress({'a','b'}, [self.resolved(['a']), None, self.resolved('b')])
        self.assertFalse(valid)
        self.assertTrue(completed == 1)
    def test_sidecar_traversal_refused_before_any_byte_read(self):
        from unittest.mock import patch
        with patch.object(Path,'read_bytes',side_effect=AssertionError('no original reads permitted')):
            with self.assertRaises(ValueError):
                readiness.original_inputs_unchanged({'../outside.json':'0'*64})
    def test_sidecar_test_path_refused_before_any_byte_read(self):
        from unittest.mock import patch
        with patch.object(Path,'read_bytes',side_effect=AssertionError('no original reads permitted')):
            with self.assertRaises(ValueError):
                readiness.original_inputs_unchanged({'data/test.jsonl':'0'*64})
    def test_recovery_synthetic_flag_survives_join(self):
        joined = audit.recovery_rows([{'id':'toy','group':'root'}], [{'id':'toy','needs_vision':'yes','synthetic':True}])
        result = audit.summary(joined)
        self.assertTrue(result['synthetic_true_rows'] == 1)
        self.assertTrue(result['non_synthetic_support_not_authentication']['needs_vision']['yes']['rows'] == 0)
    def test_task_synthetic_flag_survives_join(self):
        joined = audit.recovery_rows([{'id':'toy','group':'root','synthetic':True}], [{'id':'toy','needs_vision':'yes'}])
        self.assertTrue(audit.summary(joined)['synthetic_true_rows'] == 1)
    def test_nested_synthetic_flag_excludes_support(self):
        result = audit.summary([{'id':'toy','group':'root','label':{'needs_vision':'yes','synthetic':True}}])
        self.assertTrue(result['non_synthetic_support_not_authentication']['needs_vision']['yes']['roots'] == 0)

class ActualMaterialIntegrity(unittest.TestCase):
    def test_source_and_frozen_inputs_unchanged(self):
        status = readiness.assess()
        self.assertTrue(status['source_128_files_exact'], 'reviewed public blob binding mismatch')
        self.assertTrue(status['original_input_and_frozen_manifest_bytes_unchanged'], 'original bytes changed')
    def test_blind_wrapper_and_review_ids(self):
        status = readiness.assess()
        self.assertTrue(status['blind_wrapper_and_identity_integrity'], 'review wrapper or opaque ID mismatch')
        self.assertTrue(status['development_review_total'] == 36)
        self.assertTrue(status['development_reviews_pending'] == 36)
    def test_nonlaunchable_proposal_is_explicit(self):
        proposal = json.loads((HERE/'prepared/protocol-proposal.BLOCKED.json').read_text())
        self.assertTrue(proposal['status'] == 'DRAFT_BLOCKED_NOT_A_TRAINING_CONFIG')
        self.assertTrue(proposal['training_authorized'] is False)
        self.assertTrue(proposal['matched_update_budget_approved'] is None)
    def test_readiness_stays_blocked_despite_intact_materials(self):
        status = readiness.assess()
        self.assertTrue(status['formal_training_ready'] is False and status['status'] == 'BLOCKED')
        self.assertTrue(len(status['blockers']) >= 6)

if __name__ == '__main__':
    unittest.main()
