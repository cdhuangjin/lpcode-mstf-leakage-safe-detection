"""Recompute manuscript claims from frozen cells; no training or summary inputs."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from statistics import mean

SOURCES = {
    'A': 'results/01_transition_test_strict_origins/folds.jsonl',
    'B': 'results/02_unseen_llm/folds.jsonl',
    'C': 'results/03_style_attack/folds.jsonl',
    'D': 'repro/2502.17749/v1/results/04_cross_language/folds.jsonl',
    'ablation': 'results/05_mechanism_analysis/folds.jsonl',
    'negative': 'results/negative_pair_robustness/raw_results.json',
}
MANUSCRIPT = 'results/07_manuscript/paper_revised.md'
SEEDS = (42, 123, 2024)
LANGUAGES = ('c', 'cpp', 'java', 'py')
LLMS = ('gpt3.5', 'gemini-pro', 'wizardcoder:33b-v1.1', 'deepseek-coder:33b-instruct')
GATE_ROWS = (
    'A: strict clean; A1 vs A0; fixed XGBoost',
    'B: held-out generator; Full MSTF vs LPcode original (MLP)',
    'C: combined transformation; Full MSTF vs LPcode original (MLP)',
    'D: held-out language; Full MSTF vs LPcode original (MLP)',
)
EXPECTED_GATES = ((.9149, .9317, 1.683), (.8915, .9671, 7.563),
                  (.8415, .9531, 11.160), (.8938, .9652, 7.141))
EXPECTED_ABLATION = ((.9149, .9089), (.9317, .9205), (.9608, .9587),
                     (.9683, .9641), (.9727, .9671), (.9725, .9671))
EXPECTED_NEGATIVE = ((.9149, .9317, 1.683), (.9119, .9300, 1.805), (.8463, .8824, 3.605))
TABLE_HEADERS = {
    'Gate / comparison': ['Baseline F1', 'Method F1', 'Difference (pp)', '95% CI (pp)'],
    'Variant': ['Features', 'Representation', 'Dimensions', 'Clean F1', 'Held-out F1'],
    'Negatives': ['A0 F1', 'A1 F1', 'Difference (pp)', '95% CI (pp)'],
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_ledger(root, path, expected_hashes):
    """Reject changed bytes before interpreting any scientific measurements."""
    data = (root / path).read_bytes()
    if sha256(data) != expected_hashes.get(path):
        raise ValueError(f'source hash mismatch: {path}')
    text = data.decode('utf-8')
    return json.loads(text) if path.endswith('.json') else [json.loads(s) for s in text.splitlines() if s.strip()]


def select(rows, **criteria):
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def paired_stats(left, right, key_fields, expected_cells):
    """One-to-one cell join, exact registered coverage, and identical pair hashes."""
    def index(rows):
        indexed = {}
        for row in rows:
            key = tuple(row[k] for k in key_fields)
            if key in indexed:
                raise ValueError(f'duplicate cell: {key}')
            if not math.isfinite(row['f1']) or not 0 <= row['f1'] <= 1:
                raise ValueError(f'invalid F1: {key}')
            for field in ('train_index_sha256', 'test_index_sha256'):
                if not re.fullmatch('[0-9a-f]{64}', row.get(field, '')):
                    raise ValueError(f'invalid {field}: {key}')
            indexed[key] = row
        if set(indexed) != expected_cells:
            raise ValueError(f'missing/changed cells: expected {len(expected_cells)}, observed {len(indexed)}')
        return indexed
    a, b = index(left), index(right)
    for key in a:
        for field in ('train_index_sha256', 'test_index_sha256'):
            if a[key][field] != b[key][field]:
                raise ValueError(f'paired {field} mismatch: {key}')
    return {'baseline': mean(r['f1'] for r in a.values()),
            'method': mean(r['f1'] for r in b.values()),
            'delta_pp': mean((b[k]['f1'] - a[k]['f1']) * 100 for k in a),
            'cells': len(a), 'matching': 'PASS', 'complete_cells': 'PASS'}


def table_rows(text, header):
    """Identify a table by its first heading, then retain exact named row cells."""
    rows, active, found = {}, False, 0
    for line in text.splitlines():
        if not line.startswith('|'):
            active = False
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells[0] == header:
            if cells[1:] != TABLE_HEADERS[header]:
                raise ValueError(f'manuscript columns changed: {header}: {cells[1:]}')
            active = True
            found += 1
            continue
        if active and not re.fullmatch(r'[:\- ]+', cells[0]):
            if cells[0] in rows:
                raise ValueError(f'duplicate manuscript row: {cells[0]}')
            rows[cells[0]] = cells[1:]
    if found != 1:
        raise ValueError(f'expected one manuscript table: {header}, found {found}')
    return rows


def run_audit(root, manuscript_text=None):
    root = Path(root)
    report = {'status': 'PASS', 'scope': 'Gate A-D, A0-A5 means, negative-pair means and paired deltas; CIs not recomputed',
              'sources': {}, 'statistics': {}, 'checks': []}

    def check(name, computed, expected, observed, passed):
        report['checks'].append(dict(name=name, computed=computed, expected=expected,
                                     observed=observed, status='PASS' if passed else 'FAIL'))

    def numeric(name, computed, expected, observed, pp=False):
        precision, tolerance = (3, .001) if pp else (4, .0001)
        display = format(computed, f'+.{precision}f' if pp else f'.{precision}f')
        check(name, computed, expected, observed,
              abs(computed - expected) <= tolerance and observed == display)

    try:
        manifest = (root / 'FILE_MANIFEST.json').read_bytes()
        report['sources']['FILE_MANIFEST.json'] = sha256(manifest)
        hashes = {r['path']: r['sha256'] for r in json.loads(manifest)['files']}
        ledgers = {}
        for name, path in SOURCES.items():
            actual = sha256((root / path).read_bytes())
            report['sources'][path] = actual
            check(f'{name}: source hash', actual, hashes.get(path), actual, actual == hashes.get(path))
            ledgers[name] = load_ledger(root, path, hashes)
        manuscript_bytes = (root / MANUSCRIPT).read_bytes()
        text = manuscript_bytes.decode('utf-8') if manuscript_text is None else manuscript_text
        report['sources'][MANUSCRIPT] = sha256(manuscript_bytes)
        report['manuscript_checked_text_sha256'] = sha256(text.encode('utf-8'))
        gates = table_rows(text, 'Gate / comparison')
        ablation = table_rows(text, 'Variant')
        negatives = table_rows(text, 'Negatives')
        standard_keys = ('language', 'seed', 'fold')
        standard = set(itertools.product(LANGUAGES, SEEDS, range(5)))
        unseen_keys = standard_keys + ('heldout_llm',)
        unseen = set(itertools.product(LANGUAGES, SEEDS, range(5), LLMS))

        def compare(name, left, right, keys, cells):
            stats = paired_stats(left, right, keys, cells)
            report['statistics'][name] = stats
            check(name + ': complete cells and matched train/test hashes', stats['cells'], len(cells), stats['cells'], True)
            return stats

        for i, gate in enumerate('ABCD'):
            rows = ledgers[gate]
            if gate == 'A':
                left = select(rows, model='xgb', representation='concat')
                right = select(rows, model='xgb', representation='concat_delta')
            else:
                if gate == 'C':
                    rows = select(rows, condition='combined')
                left, right = select(rows, method='lpcode_original'), select(rows, method='mstf')
            keys, cells = (unseen_keys, unseen) if gate == 'B' else (standard_keys, standard)
            if gate == 'D':
                keys, cells = ('heldout_language', 'seed'), set(itertools.product(LANGUAGES, SEEDS))
            name = f'Gate {gate}'
            stats = compare(name, left, right, keys, cells)
            observed = gates[GATE_ROWS[i]]
            for j, metric in enumerate(('baseline', 'method', 'delta_pp')):
                numeric(name + ': ' + metric, stats[metric], EXPECTED_GATES[i][j], observed[j], j == 2)

        for i in range(6):
            for j, environment in enumerate(('clean', 'unseen')):
                rows = select(ledgers['ablation'], environment=environment)
                keys, cells = (standard_keys, standard) if j == 0 else (unseen_keys, unseen)
                name = f'A{i} {environment}'
                stats = compare(name, select(rows, method='A0'), select(rows, method=f'A{i}'), keys, cells)
                numeric(name + ': F1', stats['method'], EXPECTED_ABLATION[i][j], ablation[f'A{i}'][3 + j])

        for i, mode in enumerate(('current', 'random', 'hard')):
            rows = select(ledgers['negative'], negative_pair_mode=mode, model='xgb')
            name = f'negative {mode}'
            stats = compare(name, select(rows, representation='concat'), select(rows, representation='concat_delta'), standard_keys, standard)
            for j, metric in enumerate(('baseline', 'method', 'delta_pp')):
                numeric(name + ': ' + metric, stats[metric], EXPECTED_NEGATIVE[i][j], negatives[mode][j], j == 2)
    except (ValueError, KeyError, IndexError, OSError, TypeError) as error:
        check('audit execution', None, 'complete valid evidence and named manuscript rows', str(error), False)
    report['status'] = 'PASS' if all(c['status'] == 'PASS' for c in report['checks']) else 'FAIL'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    report = run_audit(args.root)
    output = args.output_dir or args.root / 'audit'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manuscript_evidence_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Manuscript evidence audit', '', f"Status: {report['status']}", '', report['scope'], '',
             'Arithmetic uses ledger F1 cells only. Expected claims are validation targets.',
             'Each comparison enforces the full registered Cartesian cell set and exact train/test hashes.',
             'F1 tolerance: 0.0001; delta tolerance: 0.001 pp. Named manuscript cells must match exact rounding.', '',
             '| Check | Computed | Expected | Observed | Status |', '| --- | --- | --- | --- | --- |']
    for c in report['checks']:
        lines.append('| ' + ' | '.join(str(c[k]).replace('|', '\\|') for k in ('name', 'computed', 'expected', 'observed', 'status')) + ' |')
    lines += ['', '## Source SHA-256 hashes', '']
    lines += [f'- `{p}`: `{h}`' for p, h in report['sources'].items()]
    (output / 'manuscript_evidence_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"{report['status']}: {len(report['checks'])} checks; {output}")
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
