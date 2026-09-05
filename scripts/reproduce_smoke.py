"""Fit one strict C fold through the frozen feature/pair/model implementations."""
from dataclasses import asdict
import csv
import json
import math
import platform
import subprocess
import sys
import time
from audit_isolation import file_hash, raw_inventory
from reproduce import write_json, artifact_hashes


def run(root, output):
    from lpcode_v1 import experiment, t1_strict, t3
    started = time.perf_counter()
    source = root / 'repro/2502.17749/code/experiment/task1/dataset/c.jsonl'
    upstream = root / 'repro/2502.17749/code'
    registry = json.loads((root / 'results/06_paper_assets/frozen_result_registry.json').read_text())
    revision = subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if revision != registry['official_commit_expected']:
        raise ValueError('upstream is not at pinned commit')
    expected = json.loads((root / 'results/01_transition_test_strict_origins/config.json').read_text())['source_jsonl_sha256']['c']
    if file_hash(source) != expected: raise ValueError('raw C dataset hash mismatch')
    enhanced, official = root / 'audit/cache/enhanced', root / 'audit/cache/official'
    reused = all(p.is_file() for p in t3._cache_paths('c', enhanced))
    full = t3.load_or_build_enhanced_cache('c', source, enhanced, official)
    cache = t1_strict._select_positive_bank(full, None)
    split = t3.build_t1_pair_splits(cache, language='c', n_splits=5, seed=42)[0]
    metadata = t1_strict._split_metadata(split)
    for key in ('leakage_count', 'endpoint_leakage_count', 'content_leakage_count', 'negative_component_violation_count'):
        if metadata[key] != 0: raise ValueError('smoke isolation failed: ' + key)
    config = {'language': 'c', 'seed': 42, 'fold': 0, 'n_splits': 5, 'positive_rows': len(cache.labels),
              'methods': {'A0': {'representation': 'concat', 'dimensions': 20}, 'A1': {'representation': 'concat_delta', 'dimensions': 30}},
              'model': 'fixed XGBoost from experiment.build_model', 'feature_contract': t1_strict._feature_contract(),
              'implementation_contract': t1_strict._implementation_contract(), 'official_commit': revision,
              'cache_reused': reused, 'cache_policy': 'Existing Phase3 raw-built cache validated by existing source/feature cache contracts; otherwise built from pinned raw JSONL.',
              'scaling': 'StandardScaler fits training matrix only inside experiment.evaluate_fold Pipeline.'}
    parameters = experiment.build_model('xgb', 42).named_steps['model'].get_params()
    config['model_parameters'] = {k: 'NaN (missing-value sentinel)' if isinstance(v, float) and math.isnan(v) else v for k, v in parameters.items()}
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', {'python': sys.version, 'platform': platform.platform(), 'packages': t1_strict._bound_package_versions()})
    write_json(output / 'dataset_manifest.json', {'source': source.relative_to(root).as_posix(), **raw_inventory(source), 'positive_bank_content_sha256': t3._cache_content_sha256(cache)})
    write_json(output / 'pair_manifest.json', {**metadata, 'train_pairs': [asdict(p) for p in split.train_pairs], 'test_pairs': [asdict(p) for p in split.test_pairs]})
    pair_rows = [{'split': side, **asdict(pair)} for side, pairs in [('train', split.train_pairs), ('test', split.test_pairs)] for pair in pairs]
    with (output / 'pair_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)
    metrics = []
    for method, spec in config['methods'].items():
        x_train, y_train = t1_strict._pair_matrices(cache, split.train_pairs, spec['representation'])
        x_test, y_test = t1_strict._pair_matrices(cache, split.test_pairs, spec['representation'])
        result = experiment.evaluate_fold(x_train, y_train, x_test, y_test, 'xgb', 42)
        metrics.append({'method': method, 'dimensions': x_train.shape[1], **result})
    if file_hash(source) != expected: raise ValueError('raw C dataset changed during smoke')
    write_json(output / 'metrics.json', metrics)
    with (output / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in row.items()} for row in metrics)
    summary = {'status': 'PASS', 'scope': 'One language, seed42 fold0/5, actual A0/A1 fits; not a paper-level replication.',
               'a0_f1': metrics[0]['f1'], 'a1_f1': metrics[1]['f1'], 'delta_pp': 100 * (metrics[1]['f1'] - metrics[0]['f1']),
               'elapsed_seconds': time.perf_counter() - started, 'cache_reused': reused}
    write_json(output / 'summary.json', summary)
    write_json(output / 'manifest.json', {'algorithm': 'SHA-256', 'scope': 'Smoke config, environment, dataset provenance, complete pair records, measured metrics and summary. This manifest excluded.', 'files': artifact_hashes(output)})
    return summary
