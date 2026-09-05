"""Independent isolation audit must never turn unavailable evidence into zero."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_isolation.py'
    assert path.is_file(), 'isolation auditor has not been implemented'
    spec = importlib.util.spec_from_file_location('isolation_audit', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_source_mismatch_is_fail_with_both_digests():
    result = module().compare_contract({'pair': 'old'}, {'pair': 'new'})
    assert result == {'status': 'FAIL', 'differences': [{'field': 'pair', 'expected': 'old', 'actual': 'new'}]}


def test_missing_contract_field_is_not_ignored():
    assert module().compare_contract({'pair': 'old'}, {})['status'] == 'FAIL'


def test_matching_contract_passes():
    assert module().compare_contract({'pair': 'same'}, {'pair': 'same'})['status'] == 'PASS'


def test_overlap_combines_both_endpoints_and_code_roles():
    pair = lambda h, c, x, y: SimpleNamespace(human_origin_id=h, candidate_origin_id=c,
        human_code_sha256=x, candidate_code_sha256=y)
    result = module().overlap_counts([pair('a', 'b', 'x', 'y')], [pair('b', 'c', 'y', 'z')])
    assert result == {'endpoint_overlap_count': 1, 'exact_code_overlap_count': 1}


def test_language_scoping_does_not_hide_exact_code_overlap():
    pair = SimpleNamespace(human_origin_id='same', candidate_origin_id='same',
        human_code_sha256='same-code', candidate_code_sha256='same-code')
    result = module().overlap_counts([pair], [pair], train_language='c', test_language='java')
    assert result == {'endpoint_overlap_count': 0, 'exact_code_overlap_count': 1}


def test_blocked_metrics_are_null_and_not_pass():
    result = module().blocked_reconstruction('source mismatch', 480)
    assert result['status'] == 'BLOCKED'
    assert result['endpoint_overlap_count'] is None
    assert result['exact_code_overlap_count'] is None
    assert result['matched_records'] is None
    assert result['expected_records'] == 480


def test_raw_inventory_hashes_code_from_content(tmp_path):
    import hashlib
    import json
    path = tmp_path / 'raw.jsonl'
    path.write_text(json.dumps({'label': 1, 'file_name': 'a', 'human_src': 'H', 'llm_src': 'L'}) + '\n')
    result = module().raw_inventory(path)
    assert result['rows'] == 1
    assert result['human_code_hashes_sha256'] == module().digest([hashlib.sha256(b'H').hexdigest()])
    assert result['candidate_code_hashes_sha256'] == module().digest([hashlib.sha256(b'L').hexdigest()])


def test_record_binding_detects_changed_metric():
    audit = module()
    row = {'f1': .9}
    row['record_sha256'] = audit.digest(row)
    assert audit.record_bindings([row])['status'] == 'PASS'
    row['f1'] = .99
    assert audit.record_bindings([row])['status'] == 'FAIL'


def test_empty_ledger_is_fail():
    assert module().record_bindings([])['status'] == 'FAIL'


def test_failed_check_dominates_blocked():
    assert module().overall_status(['PASS', 'BLOCKED', 'FAIL']) == 'FAIL'
    assert module().overall_status(['PASS', 'BLOCKED']) == 'BLOCKED'


def test_compare_reconstruction_rejects_digest_mismatch_for_every_method():
    rows = [{'method': method, 'train_index_sha256': 'train', 'test_index_sha256': test}
            for method, test in [('a', 'test'), ('b', 'changed')]]
    result = module().compare_reconstruction(rows, {'train_index_sha256': 'train', 'test_index_sha256': 'test'},
        {'endpoint_overlap_count': 0, 'exact_code_overlap_count': 0})
    assert result['status'] == 'FAIL'
    assert result['matched_records'] == 1
    assert result['mismatches'][0]['record_index'] == 1


def test_compare_reconstruction_overlap_is_failure_even_when_digests_match():
    result = module().compare_reconstruction([{'train_index_sha256': 'x'}], {'train_index_sha256': 'x'},
        {'endpoint_overlap_count': 1, 'exact_code_overlap_count': 0})
    assert result['status'] == 'FAIL'


def test_preflight_attests_frozen_source_without_rewriting():
    audit = module()
    before = audit.file_hash(audit.ROOT / 'results/01_transition_test_strict_origins/config.json')
    report, configs, ledgers = audit.preflight(audit.ROOT)
    assert set(configs) == set('ABCD')
    assert len(ledgers['A']) == 480
    assert report['gates']['A']['implementation_contract']['status'] == 'FAIL'
    assert report['gates']['A']['record_bindings']['status'] == 'PASS'
    assert audit.file_hash(audit.ROOT / 'results/01_transition_test_strict_origins/config.json') == before


def test_current_reconstruction_uses_real_pair_builders_and_detects_ledger_tampering():
    audit = module()
    spec = importlib.util.spec_from_file_location('t5_fixture', audit.ROOT / 'repro/2502.17749/v1/tests/test_t5.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    from lpcode_v1 import t1_strict, t3
    cache = fixture._memory_cache()
    split = t3.build_t1_pair_splits(cache, 'c', 5, 42)[0]
    row = {'language': 'c', 'seed': 42, 'fold': 0, **t1_strict._split_metadata(split)}
    result = audit.reconstruct_with_caches({'c': cache}, {'A': {'seeds': [42], 'n_splits': 5}}, {'A': [row]})
    assert result['A']['status'] == 'PASS'
    row['test_index_sha256'] = 'changed'
    assert audit.reconstruct_with_caches({'c': cache}, {'A': {'seeds': [42], 'n_splits': 5}}, {'A': [row]})['A']['status'] == 'FAIL'


def test_cli_preflight_reports_nonzero_exit_and_blocked_reconstruction(tmp_path):
    import json
    import subprocess
    import sys
    audit = module()
    output = tmp_path / 'audit.json'
    completed = subprocess.run([sys.executable, str(audit.ROOT / 'scripts/audit_isolation.py'),
        '--preflight-only', '--output', str(output)], capture_output=True, text=True)
    assert output.is_file(), completed.stderr
    result = json.loads(output.read_text())
    assert completed.returncode == 1
    assert result['status'] == 'BLOCKED'
    assert result['gates']['B']['reconstruction']['matched_records'] is None
    assert all(item['status'] == 'PASS' for item in result['raw_source_bindings'].values())
    assert result['upstream_tracked_tree']['status'] == 'PASS'
    assert result['attack_contract']['status'] == 'PASS'
    assert 'reuse' in result['cache_policy']


def test_ledger_coverage_rejects_duplicate_and_missing_cells():
    audit = module()
    config = {'languages': ['c'], 'seeds': [42], 'n_splits': 1, 'models': ['xgb'], 'representations': ['concat']}
    row = {'language': 'c', 'seed': 42, 'fold': 0, 'model': 'xgb', 'representation': 'concat'}
    assert audit.ledger_coverage('A', config, [row])['status'] == 'PASS'
    assert audit.ledger_coverage('A', config, [row, row])['status'] == 'FAIL'
    assert audit.ledger_coverage('A', config, [])['status'] == 'FAIL'


def test_historical_archive_must_match_exact_declared_bytes(tmp_path):
    audit = module()
    source = tmp_path / 'historic.py'
    source.write_text('old bytes')
    expected = audit.file_hash(source)
    assert audit.historical_coverage({'pair': expected}, {'pair': 'changed'}, [source])['status'] == 'PASS'
    source.write_text('different bytes')
    assert audit.historical_coverage({'pair': expected}, {'pair': 'changed'}, [source])['status'] == 'FAIL'


def test_transformed_overlap_checks_successful_outputs_against_both_train_roles():
    audit = module()
    spec = importlib.util.spec_from_file_location('t4_fixture', audit.ROOT / 'repro/2502.17749/v1/tests/test_t4.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    from lpcode_v1 import t3, t4
    clean = fixture._memory_cache(10)
    attack = fixture._attack_cache(clean)
    split = t3.build_t1_pair_splits(clean, 'c', 5, 42)[0]
    from dataclasses import replace
    attack = replace(attack, output_sha256=attack.output_sha256.astype('U64'))
    attack.output_sha256[:] = split.train_pairs[0].human_code_sha256
    result = audit.attack_split_metadata(clean, attack, split, 'combined')
    assert result['transformed_exact_code_overlap_count'] == 1
    assert result['metadata']['attack_success_set_sha256'] == t3._digest_json([pair.pair_sha256 for pair in split.test_pairs])


def test_attack_language_audit_rejects_output_binding_change():
    audit = module()
    spec = importlib.util.spec_from_file_location('t4_fixture', audit.ROOT / 'repro/2502.17749/v1/tests/test_t4.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    from lpcode_v1 import t3
    clean = fixture._memory_cache(10)
    attack = fixture._attack_cache(clean)
    rows = [{'language': 'c', 'seed': 42, 'fold': split.fold, 'condition': 'combined', 'method': 'mstf',
             **audit.attack_split_metadata(clean, attack, split, 'combined')['metadata']}
            for split in t3.build_t1_pair_splits(clean, 'c', 5, 42)]
    config = {'seeds': [42], 'n_splits': 5, 'conditions': ['combined'], 'methods': ['mstf']}
    assert audit.reconstruct_attack_language(clean, attack, config, rows)['status'] == 'PASS'
    assert audit.reconstruct_attack_language(clean, attack, config, rows[:-1])['status'] == 'FAIL'
    rows[0]['attack_output_set_sha256'] = 'changed'
    assert audit.reconstruct_attack_language(clean, attack, config, rows)['status'] == 'FAIL'


def test_ablation_bindings_verify_all_variants_and_detect_changed_test_digest():
    import json
    audit = module()
    root = audit.ROOT
    formal = {gate: [json.loads(line) for line in (root / relative / 'folds.jsonl').read_text().splitlines()]
              for gate, relative in audit.GATES.items() if gate in 'AB'}
    rows = [json.loads(line) for line in (root / 'results/05_mechanism_analysis/folds.jsonl').read_text().splitlines()]
    assert audit.ablation_bindings(rows, formal)['matched_records'] == 1800
    assert audit.ablation_bindings(rows, formal)['status'] == 'PASS'
    rows[0]['test_index_sha256'] = 'changed'
    assert audit.ablation_bindings(rows, formal)['status'] == 'FAIL'


def test_late_attack_error_preserves_verified_pair_evidence():
    audit = module()
    report = {'gates': {'A': {'reconstruction': {'status': 'PASS', 'matched_records': 480}}},
              'attack_reconstruction': {'status': 'BLOCKED', 'matched_records': None}}
    audit.record_stage_error(report, 'attack_reconstruction', ValueError('bad cache'))
    assert report['gates']['A']['reconstruction']['matched_records'] == 480
    assert report['attack_reconstruction']['reason'] == 'ValueError: bad cache'
    assert report['reconstruction_error']['stage'] == 'attack_reconstruction'


def test_cli_read_failure_replaces_prior_pass_report(tmp_path):
    import json
    import subprocess
    import sys
    audit = module()
    output = tmp_path / 'audit.json'
    output.write_text('{"status": "PASS"}')
    completed = subprocess.run([sys.executable, str(audit.ROOT / 'scripts/audit_isolation.py'),
        '--root', str(tmp_path / 'missing'), '--output', str(output)], capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert completed.returncode == 1
    assert result['status'] == 'FAIL'
    assert result['error']['type'] == 'FileNotFoundError'
    assert result['matched_records'] is None
