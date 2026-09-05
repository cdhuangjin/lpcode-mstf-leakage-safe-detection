"""Read-only raw reconstruction audit. Never trains or edits frozen evidence.

Current-source reconstruction is separate from historical source attestation.
Gate D endpoint identities are language-scoped, not global semantic identities.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'repro/2502.17749/v1'))


def digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def compare_contract(expected, actual):
    differences = [{'field': key, 'expected': value, 'actual': actual.get(key)}
                   for key, value in expected.items() if actual.get(key) != value]
    return {'status': 'FAIL' if differences else 'PASS', 'differences': differences}


def historical_coverage(expected, actual, archives):
    archived = {file_hash(path): str(path) for path in archives if path.is_file()}
    rows = [{'field': key, 'expected': value, 'current': actual.get(key),
             'archive': archived.get(value),
             'status': 'PASS' if value == actual.get(key) or value in archived else 'FAIL'}
            for key, value in expected.items()]
    return {'status': overall_status([row['status'] for row in rows]), 'fields': rows}


def overlap_counts(train, test, train_language=None, test_language=None):
    def endpoints(pairs, language):
        return {(language, value) for pair in pairs for value in (pair.human_origin_id, pair.candidate_origin_id)}
    def codes(pairs):
        return {value for pair in pairs for value in (pair.human_code_sha256, pair.candidate_code_sha256)}
    return {'endpoint_overlap_count': len(endpoints(train, train_language) & endpoints(test, test_language)),
            'exact_code_overlap_count': len(codes(train) & codes(test))}


def blocked_reconstruction(reason, expected_records):
    return {'status': 'BLOCKED', 'reason': reason, 'expected_records': expected_records,
            'matched_records': None, 'endpoint_overlap_count': None, 'exact_code_overlap_count': None}


def raw_inventory(path):
    from lpcode_v1.data import load_jsonl
    rows = load_jsonl(path, 'task1')
    human = [hashlib.sha256(str(row['human_src']).encode()).hexdigest() for row in rows]
    candidate = [hashlib.sha256(str(row['llm_src']).encode()).hexdigest() for row in rows]
    return {'rows': len(rows), 'source_sha256': file_hash(path),
            'human_code_hashes_sha256': digest(human), 'candidate_code_hashes_sha256': digest(candidate),
            'unique_human_code_hashes': len(set(human)), 'unique_candidate_code_hashes': len(set(candidate)),
            'positive_rows': sum(int(row['label']) == 1 for row in rows)}


def record_bindings(rows):
    invalid = [i for i, row in enumerate(rows) if row.get('record_sha256') != digest({k: v for k, v in row.items() if k != 'record_sha256'})]
    return {'status': 'FAIL' if invalid or not rows else 'PASS', 'records': len(rows), 'invalid_record_indices': invalid}


def overall_status(statuses):
    return 'FAIL' if 'FAIL' in statuses else ('BLOCKED' if 'BLOCKED' in statuses else 'PASS')


def compare_reconstruction(rows, metadata, overlaps):
    mismatches = []
    for index, row in enumerate(rows):
        differences = compare_contract(metadata, row)['differences']
        if differences:
            mismatches.append({'record_index': index, 'differences': differences})
    return {'status': 'FAIL' if mismatches or any(overlaps.values()) or not rows else 'PASS',
            'matched_records': len(rows) - len(mismatches), 'expected_records': len(rows),
            'mismatches': mismatches, **overlaps}


GATES = {'A': 'results/01_transition_test_strict_origins', 'B': 'results/02_unseen_llm',
         'C': 'results/03_style_attack', 'D': 'repro/2502.17749/v1/results/04_cross_language'}


def ledger_coverage(gate, config, rows):
    axes = {'language': config['languages'], 'seed': config['seeds']}
    if gate == 'D':
        axes = {'heldout_language': config['heldout_languages'], 'seed': config['seeds'], 'method': config['methods']}
    else:
        axes['fold'] = list(range(config['n_splits']))
        if gate == 'A':
            axes.update(model=config['models'], representation=config['representations'])
        else:
            axes['method'] = config['methods']
            axes['heldout_llm' if gate == 'B' else 'condition'] = config['heldout_llms' if gate == 'B' else 'conditions']
    expected = set(itertools.product(*axes.values()))
    actual = [tuple(row.get(key) for key in axes) for row in rows]
    return {'status': 'PASS' if set(actual) == expected and len(actual) == len(expected) else 'FAIL',
            'expected_cells': len(expected), 'actual_records': len(actual),
            'duplicate_cells': len(actual) - len(set(actual)), 'missing_cells': len(expected - set(actual)),
            'unexpected_cells': len(set(actual) - expected)}


def ablation_bindings(rows, formal):
    from lpcode_v1.t3 import LLM_SOURCES
    fields = ('train_index_sha256', 'test_index_sha256', 'train_rows', 'test_rows')
    reports = []
    for environment, gate in [('clean', 'A'), ('unseen', 'B')]:
        keys = ('language', 'seed', 'fold') + (('heldout_llm',) if gate == 'B' else ())
        axes = [('c', 'cpp', 'java', 'py'), (42, 123, 2024), range(5)] + ([LLM_SOURCES] if gate == 'B' else [])
        expected = set(itertools.product(*axes))
        bindings = {}
        inconsistent = False
        for row in formal[gate]:
            key = tuple(row[name] for name in keys)
            value = {name: row[name] for name in fields}
            inconsistent |= key in bindings and bindings[key] != value
            bindings[key] = value
        for variant in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5'):
            selected = [row for row in rows if row['environment'] == environment and row['method'] == variant]
            observed = [tuple(row[name] for name in keys) for row in selected]
            matched = sum(key in bindings and all(row.get(name) == bindings[key][name] for name in fields)
                          for key, row in zip(observed, selected))
            okay = not inconsistent and set(bindings) == expected and set(observed) == expected and len(observed) == len(expected) and matched == len(expected)
            reports.append({'environment': environment, 'variant': variant, 'formal_gate': gate,
                            'status': 'PASS' if okay else 'FAIL', 'matched_records': matched, 'expected_records': len(expected)})
    return {'status': overall_status([item['status'] for item in reports] + ['PASS' if len(rows) == 1800 else 'FAIL']),
            'matched_records': sum(item['matched_records'] for item in reports), 'variants': reports,
            'scope': 'A0-A5 exact pair/index hashes and row counts against every formal A/B method per cell; not model refits.'}


def preflight(root):
    from lpcode_v1 import t1_strict, t3
    registry_path = root / 'results/06_paper_assets/frozen_result_registry.json'
    registry = json.loads(registry_path.read_text())
    report = {'schema_version': 1, 'registry_sha256': file_hash(registry_path), 'gates': {},
              'scope': 'Gate A-D only; no model training; current-source reconstruction does not attest historical source bytes',
              'gate_d_origin_scope': 'Language-scoped endpoint identities; no assertion of global problem or semantic identity.',
              'attack_scope': 'Gate C clean pair membership plus raw rebuilt attack-cache, success/output sets and successful transformed-test-code intersections.'}
    configs, ledgers = {}, {}
    modules = root / 'repro/2502.17749/v1/lpcode_v1'
    source_map = {'experiment_source_sha256': 'experiment.py', 'representations_source_sha256': 'representations.py',
                  't4_source_sha256': 't4.py', 't3_source_sha256': 't3.py', 't5_source_sha256': 't5.py',
                  'pair_builder_source_sha256': 't3.py', 'official_features_source_sha256': 'features_official.py',
                  'enhanced_features_source_sha256': 'features_enhanced.py'}
    for gate, relative in GATES.items():
        bundle = root / relative
        config = json.loads((bundle / 'config.json').read_text())
        rows = [json.loads(line) for line in (bundle / 'folds.jsonl').read_text().splitlines() if line.strip()]
        configs[gate], ledgers[gate] = config, rows
        files = []
        for name, expected in registry['bundles']['gate_' + gate.lower()]['files'].items():
            path = bundle / name
            actual = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
            files.append({'file': str(path.relative_to(root)), **actual,
                          **compare_contract({k: expected[k] for k in actual}, actual)})
        manifest = json.loads((bundle / 'manifest.json').read_text())
        manifest_files = []
        for name, expected in manifest['files'].items():
            path = bundle / name
            actual = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
            manifest_files.append({'file': name, **compare_contract({k: expected[k] for k in actual}, actual)})
        if gate == 'A':
            actual_contract = t1_strict._implementation_contract()
        elif gate == 'B':
            actual_contract = t3._runner_implementation_contract()
        else:
            actual_contract = {key: file_hash(modules / source_map[key])
                               for key in config['implementation_contract'] if key in source_map and (modules / source_map[key]).exists()}
        report['gates'][gate] = {'registry_files': files, 'manifest_files': manifest_files,
            'record_bindings': record_bindings(rows),
            'ledger_coverage': ledger_coverage(gate, config, rows),
            'config_binding': compare_contract({'config_id': digest({k: v for k, v in config.items() if k != 'config_id'})}, config),
            'implementation_contract': compare_contract(config['implementation_contract'], actual_contract),
            'historical_source_coverage': historical_coverage(config['implementation_contract'], actual_contract,
                [root / 'audit/historical_sources/t3_gate_a.py', root / 'audit/historical_sources/t3_gate_b.py']),
            'axes': compare_contract({'seeds': [42, 123, 2024], 'languages': ['c', 'cpp', 'java', 'py'], 'limit_origins': None}, config),
            'reconstruction': blocked_reconstruction('Raw reconstruction has not run', len(rows))}
    return report, configs, ledgers


def reconstruct_with_caches(caches, configs, ledgers):
    """Reuse public pair builders, then independently count endpoint intersections.

    Calls neither training runners nor resume/config validators. A historical
    source mismatch remains a separate failure even if these digests match.
    """
    from lpcode_v1 import t1_strict, t3, t5
    results = {}
    for gate, rows in ledgers.items():
        config = configs[gate]
        cells = []
        if gate != 'D':
            for language, cache in caches.items():
                for seed in config['seeds']:
                    holdouts = config['heldout_llms'] if gate == 'B' else [None]
                    for heldout in holdouts:
                        splits = (t3.build_t3_splits(cache, language, heldout, config['n_splits'], seed)
                                  if gate == 'B' else t3.build_t1_pair_splits(cache, language, config['n_splits'], seed))
                        for split in splits:
                            selected = [row for row in rows if row['language'] == language and row['seed'] == seed
                                        and row['fold'] == split.fold and (gate != 'B' or row['heldout_llm'] == heldout)]
                            if not selected:
                                continue
                            metadata = t3._t3_split_metadata(cache, split) if gate == 'B' else t1_strict._split_metadata(split)
                            counts = overlap_counts(split.train_pairs, split.test_pairs)
                            cells.append({'language': language, 'seed': seed, 'fold': split.fold, 'heldout_llm': heldout,
                                          **compare_reconstruction(selected, metadata, counts)})
        else:
            for seed in config['seeds']:
                banks = {language: t5.build_language_pair_bank(cache, language, seed, config['n_pair_folds'])
                         for language, cache in caches.items()}
                for heldout in config['heldout_languages']:
                    train = {language: (caches[language], banks[language]) for language in caches if language != heldout}
                    metadata = t5._cross_language_split_metadata(train, caches[heldout], banks[heldout])
                    train_codes, test_codes, train_origins, test_origins = set(), set(), set(), set()
                    for language, bank in banks.items():
                        codes = test_codes if language == heldout else train_codes
                        origins = test_origins if language == heldout else train_origins
                        for pair in bank.pairs:
                            codes.update((pair.human_code_sha256, pair.candidate_code_sha256))
                            origins.update(((language, pair.human_origin_id), (language, pair.candidate_origin_id)))
                    counts = {'endpoint_overlap_count': len(train_origins & test_origins), 'exact_code_overlap_count': len(train_codes & test_codes)}
                    selected = [row for row in rows if row['heldout_language'] == heldout and row['seed'] == seed]
                    cells.append({'heldout_language': heldout, 'seed': seed, **compare_reconstruction(selected, metadata, counts)})
        matched = sum(cell['matched_records'] for cell in cells)
        results[gate] = {'status': overall_status([cell['status'] for cell in cells] + ['PASS' if matched == len(rows) else 'FAIL']),
                         'expected_records': len(rows), 'matched_records': matched, 'cells': cells,
                         'endpoint_overlap_count': sum(cell['endpoint_overlap_count'] for cell in cells),
                         'exact_code_overlap_count': sum(cell['exact_code_overlap_count'] for cell in cells)}
    return results


def attack_split_metadata(clean, attack_cache, split, condition):
    from lpcode_v1 import t3, t4
    if condition == 'clean':
        audit = t4._clean_audit(split)
        test_codes = {code for pair in split.test_pairs for code in (pair.human_code_sha256, pair.candidate_code_sha256)}
    else:
        row_hashes = clean.row_sha256[[pair.candidate_positive_row_idx for pair in split.test_pairs]]
        _, mask, audit = t4._attack_rows(attack_cache, row_hashes, condition)
        audit['success_set_sha256'] = t3._digest_json([pair.pair_sha256 for pair, ok in zip(split.test_pairs, mask.tolist()) if ok])
        lookup = {str(value): index for index, value in enumerate(attack_cache.row_sha256)}
        column = t4.ATTACKS.index(condition)
        test_codes = {pair.human_code_sha256 for pair, ok in zip(split.test_pairs, mask.tolist()) if ok}
        test_codes.update(str(attack_cache.output_sha256[lookup[str(row)], column]) for row, ok in zip(row_hashes, mask.tolist()) if ok)
    train_codes = {code for pair in split.train_pairs for code in (pair.human_code_sha256, pair.candidate_code_sha256)}
    return {'metadata': {'attack_' + key: value for key, value in audit.items()},
            'transformed_exact_code_overlap_count': len(train_codes & test_codes)}


def reconstruct_attack_language(clean, attack, config, rows):
    from lpcode_v1 import t3
    local_rows = [row for row in rows if row['language'] == clean.language]
    expected = set(itertools.product(config['seeds'], range(config['n_splits']), config['conditions'], config['methods']))
    observed = [(row['seed'], row['fold'], row['condition'], row['method']) for row in local_rows]
    coverage = 'PASS' if set(observed) == expected and len(observed) == len(expected) else 'FAIL'
    cells = []
    for seed in config['seeds']:
        for split in t3.build_t1_pair_splits(clean, clean.language, config['n_splits'], seed):
            for condition in config['conditions']:
                selected = [row for row in rows if row['language'] == clean.language and row['seed'] == seed
                            and row['fold'] == split.fold and row['condition'] == condition]
                if not selected:
                    continue
                actual = attack_split_metadata(clean, attack, split, condition)
                cells.append({'seed': seed, 'fold': split.fold, 'condition': condition,
                    **compare_reconstruction(selected, actual['metadata'],
                        {'transformed_exact_code_overlap_count': actual['transformed_exact_code_overlap_count']})})
    return {'status': overall_status([cell['status'] for cell in cells] + [coverage]),
            'coverage': {'status': coverage, 'expected_records': len(expected), 'actual_records': len(observed)},
            'matched_records': sum(cell['matched_records'] for cell in cells), 'cells': cells,
            'transformed_exact_code_overlap_count': sum(cell['transformed_exact_code_overlap_count'] for cell in cells)}


def record_stage_error(report, stage, exc):
    report['reconstruction_error'] = {'status': 'FAIL', 'stage': stage, 'type': type(exc).__name__, 'message': str(exc)}
    if stage in report and report[stage].get('status') == 'BLOCKED':
        report[stage]['reason'] = f'{type(exc).__name__}: {exc}'
    for gate in report['gates'].values():
        if gate['reconstruction']['status'] == 'BLOCKED':
            gate['reconstruction']['reason'] = f'{type(exc).__name__}: {exc}'


def run_audit(root, preflight_only=False):
    from lpcode_v1 import attacks as attack_source, t1_strict, t3, t4, t5
    report, configs, ledgers = preflight(root)
    report['ablation_bindings'] = ablation_bindings([json.loads(line) for line in
        (root / 'results/05_mechanism_analysis/folds.jsonl').read_text().splitlines() if line.strip()], ledgers)
    report['source_attestation_policy'] = 'Current whole-module equality is disclosed, not required after historical byte-exact coverage; acceptance also requires independent raw pair/cache reconstruction.'
    report['cache_policy'] = 'Validated audit/cache reuse is permitted. Per-language cache_files_present_before distinguishes existing cache validation from fresh construction; presence does not assert recomputation. No author result caches are used.'
    upstream = root / 'repro/2502.17749/code'
    tracked = subprocess.run(['git', '-C', str(upstream), 'ls-files'], capture_output=True, text=True, check=True)
    tracked_paths = set(tracked.stdout.splitlines())
    report['upstream_file_manifest'] = [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                                        'sha256': file_hash(path), 'tracked_by_upstream': path.relative_to(upstream).as_posix() in tracked_paths}
                                        for path in sorted(upstream.rglob('*'))
                                        if path.is_file() and '.git' not in path.relative_to(upstream).parts]
    changed = subprocess.run(['git', '-C', str(upstream), 'diff', '--name-only', 'HEAD'], capture_output=True, text=True, check=True)
    report['upstream_tracked_tree'] = {'status': 'PASS' if not changed.stdout.strip() else 'FAIL',
                                       'changed_tracked_paths': changed.stdout.splitlines(),
                                       'note': 'Untracked local baseline outputs and bytecode are separately marked in the file manifest.'}
    revision = subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    registry = json.loads((root / 'results/06_paper_assets/frozen_result_registry.json').read_text())
    report['upstream_commit'] = compare_contract({'commit': registry['official_commit_expected']}, {'commit': revision.stdout.strip()})
    paths = {language: upstream / f'experiment/task1/dataset/{language}.jsonl' for language in configs['A']['languages']}
    report['raw_inventory'] = {language: raw_inventory(path) for language, path in paths.items()}
    raw_hashes = {language: item['source_sha256'] for language, item in report['raw_inventory'].items()}
    report['raw_source_bindings'] = {gate: compare_contract(config['source_jsonl_sha256'], raw_hashes) for gate, config in configs.items()}
    report['feature_contracts'] = {
        'A': compare_contract(configs['A']['feature_contract'], t1_strict._feature_contract()),
        'B': compare_contract(configs['B']['feature_contract'], t3._runner_feature_contract())}
    report['attack_contract'] = compare_contract(configs['C']['attack_contract'], {
        'version': attack_source.ATTACK_VERSION, 'source_sha256': attack_source.attack_source_sha256(),
        'conditions': list(t4.ATTACKS), 'scope': 'test-candidate-endpoint-only',
        'combined_order': ['comment_removal', 'identifier_rename', 'format_normalization']})
    report['cache_reconstruction'] = {'status': 'BLOCKED', 'reason': 'Raw reconstruction has not run', 'languages': None}
    report['attack_reconstruction'] = {'status': 'BLOCKED', 'reason': 'Attack reconstruction has not run',
                                       'matched_records': None, 'transformed_exact_code_overlap_count': None}
    if not preflight_only and all(item['status'] == 'PASS' for item in report['raw_source_bindings'].values()):
        caches = {}
        stage = 'cache_reconstruction'
        try:
            cache_rows = {}
            for language, path in paths.items():
                present = all(path.is_file() for path in t3._cache_paths(language, root / 'audit/cache/enhanced'))
                print(f'Validating/building {language} features in audit/cache/ (cache present: {present})', flush=True)
                full = t3.load_or_build_enhanced_cache(language, path, root / 'audit/cache/enhanced', root / 'audit/cache/official')
                cache = t3._select_t3_positive_bank(full, None)
                caches[language] = cache
                actual_b, actual_cd = t3._cache_content_sha256(cache), t5._cache_digest(cache)
                cache_rows[language] = {
                    'cache_files_present_before': present,
                    'B': compare_contract({'sha256': configs['B']['cache_content_sha256'][language]}, {'sha256': actual_b}),
                    'C': compare_contract({'sha256': configs['C']['clean_cache_content_sha256'][language]}, {'sha256': actual_cd}),
                    'D': compare_contract({'sha256': configs['D']['cache_content_sha256'][language]}, {'sha256': actual_cd}),
                    'current_positive_bank_B_digest': actual_b, 'current_positive_bank_CD_digest': actual_cd,
                    'full_raw_rows': len(full.labels), 'positive_bank_rows': len(cache.labels)}
            report['cache_reconstruction'] = {'status': overall_status([item[gate]['status'] for item in cache_rows.values() for gate in 'BCD']),
                'languages': cache_rows, 'contract': 'B hashes every positive-bank array; C/D hash normalized row hashes plus feature matrices.'}
            print('Reconstructing Gate A-D splits and all method bindings; no training', flush=True)
            stage = 'pair_reconstruction'
            for gate, result in reconstruct_with_caches(caches, configs, ledgers).items():
                report['gates'][gate]['reconstruction'] = result
            attacks = {}
            stage = 'attack_reconstruction'
            for language, path in paths.items():
                present = all(path.is_file() for path in t4._attack_cache_paths(language, root / 'audit/cache/attack'))
                print(f'Validating/building {language} attack cache (cache present: {present}); no training', flush=True)
                attack = t4.load_or_build_attack_cache(language, path, root / 'audit/cache/attack')
                attacks[language] = {'cache_files_present_before': present, 'cache_binding': compare_contract(
                    {'sha256': configs['C']['attack_cache_content_sha256'][language]}, {'sha256': attack.semantic_content_sha256}),
                    'actual_semantic_content_sha256': attack.semantic_content_sha256,
                    **reconstruct_attack_language(caches[language], attack, configs['C'], ledgers['C'])}
            report['attack_reconstruction'] = {'status': overall_status([item['status'] for item in attacks.values()] +
                [item['cache_binding']['status'] for item in attacks.values()]), 'languages': attacks,
                'matched_records': sum(item['matched_records'] for item in attacks.values()),
                'transformed_exact_code_overlap_count': sum(item['transformed_exact_code_overlap_count'] for item in attacks.values())}
        except Exception as exc:
            record_stage_error(report, stage, exc)
    report['raw_sources_unchanged_after_audit'] = compare_contract(raw_hashes, {language: file_hash(path) for language, path in paths.items()})
    def statuses(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'status':
                    yield child
                elif key != 'implementation_contract':
                    yield from statuses(child)
        elif isinstance(value, list):
            for child in value:
                yield from statuses(child)
    report['status'] = overall_status(list(statuses(report)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'audit/isolation_audit.json')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    try:
        report = run_audit(args.root.resolve(), args.preflight_only)
    except Exception as exc:
        report = {'schema_version': 1, 'status': 'FAIL', 'root': str(args.root.resolve()),
                  'error': {'stage': 'audit_execution', 'type': type(exc).__name__, 'message': str(exc)},
                  'matched_records': None, 'endpoint_overlap_count': None, 'exact_code_overlap_count': None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f"{report['status']}: {args.output}")
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
