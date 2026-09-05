"""Independent arithmetic and mutation checks for the frozen manuscript evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/audit_manuscript_evidence.py'


def audit():
    assert SCRIPT.exists(), 'The independent manuscript audit is not implemented'
    spec = importlib.util.spec_from_file_location('manuscript_audit', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cells():
    return [dict(seed=s, f1=f, train_index_sha256='a' * 64,
                 test_index_sha256='b' * 64) for s, f in [(42, .7), (123, .9)]]


def test_paired_arithmetic():
    module = audit()
    left, right = cells(), cells()
    right[0]['f1'] = .8
    result = module.paired_stats(left, right, ('seed',), {(42,), (123,)})
    assert result['baseline'] == pytest.approx(.8)
    assert result['method'] == pytest.approx(.85)
    assert result['delta_pp'] == pytest.approx(5)
    assert result['cells'] == 2


@pytest.mark.parametrize('mutation', ['missing', 'both_missing', 'duplicate', 'wrong_cell', 'hash', 'empty_hash'])
def test_reject_bad_cell_joins(mutation):
    module = audit()
    left, right = cells(), cells()
    if mutation == 'missing':
        right.pop()
    elif mutation == 'both_missing':
        left.pop(); right.pop()
    elif mutation == 'duplicate':
        right.append(right[0].copy())
    elif mutation == 'wrong_cell':
        right[0]['seed'] = 999
    elif mutation == 'hash':
        right[0]['train_index_sha256'] = 'c' * 64
    else:
        left[0]['test_index_sha256'] = right[0]['test_index_sha256'] = ''
    with pytest.raises(ValueError):
        module.paired_stats(left, right, ('seed',), {(42,), (123,)})


def test_repository_evidence():
    result = audit().run_audit(ROOT)
    assert result['status'] == 'PASS', result
    assert len(result['checks']) >= 40
    assert result['statistics']['Gate A']['cells'] == 60
    assert result['statistics']['Gate B']['cells'] == 240
    assert result['statistics']['Gate D']['cells'] == 12


@pytest.mark.parametrize('before,after', [
    ('| hard | 0.8463 |', '| hard | 0.9463 |'),
    ('| hard | 0.8463 |', '| hard | 0.84630 |'),
    ('| hard | 0.8463 |', '| hard-renamed | 0.8463 |'),
    ('| 0.9149 | 0.9317 | +1.683 |', '| 0.9149 | 0.9317 | +1.684 |'),
    ('| A5 | Enhanced 28 | Endpoints + signed + relative | 112 | 0.9725 |',
     '| A5 | Enhanced 28 | Endpoints + signed + relative | 112 | 0.9726 |'),
    ('| Clean F1 | Held-out F1 |', '| Held-out F1 | Clean F1 |'),
    ('| Baseline F1 | Method F1 |', '| Method F1 | Baseline F1 |'),
    ('| A0 F1 | A1 F1 |', '| A1 F1 | A0 F1 |'),
])
def test_named_manuscript_cell_is_checked(before, after):
    module = audit()
    text = (ROOT / module.MANUSCRIPT).read_text(encoding='utf-8')
    text = text.replace(before, after, 1)
    result = module.run_audit(ROOT, manuscript_text=text)
    assert result['status'] == 'FAIL'
    assert any(c['status'] == 'FAIL' for c in result['checks'])


def test_changed_ledger_rejected(tmp_path):
    module = audit()
    path = module.SOURCES['A']
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text('{"f1": 0.1}\n', encoding='utf-8')
    expected = {path: __import__('hashlib').sha256((ROOT / path).read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='source hash'):
        module.load_ledger(tmp_path, path, expected)


def test_cli_writes_reports_and_returns_nonzero_on_bad_source(tmp_path):
    module = audit()
    assert module.main(['--root', str(ROOT), '--output-dir', str(tmp_path / 'pass')]) == 0
    result = json.loads((tmp_path / 'pass/manuscript_evidence_audit.json').read_text())
    assert result['status'] == 'PASS'
    assert (tmp_path / 'pass/manuscript_evidence_audit.md').is_file()
    assert module.main(['--root', str(tmp_path), '--output-dir', str(tmp_path / 'fail')]) == 1
    result = json.loads((tmp_path / 'fail/manuscript_evidence_audit.json').read_text())
    assert result['status'] == 'FAIL'
