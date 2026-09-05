import importlib.util
import json
import csv
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def wrapper():
    assert (ROOT / 'scripts/reproduce.py').is_file(), 'reproduction CLI is missing'
    import reproduce
    return reproduce


def test_saved_tables_and_stable_hashes(tmp_path):
    r = wrapper()
    first, second = tmp_path / 'one', tmp_path / 'two'
    first.mkdir(); second.mkdir()
    r.run_saved(ROOT, first, 'all-saved')
    r.run_saved(ROOT, second, 'all-saved')
    assert r.artifact_hashes(first) == r.artifact_hashes(second)
    for table in range(2, 8):
        assert (first / f'table{table}.csv').is_file()
        assert (first / f'table{table}.md').is_file()
    assert '| C | 0.8922 ± 0.0046 | 0.9656 ± 0.0010 | +7.339 |' in (first / 'table6.md').read_text(encoding='utf-8')
    assert len((first / 'table5.csv').read_text().splitlines()) == 7
    assert (first / 'figure_mechanism.csv').stat().st_size > 100
    assert 'not rebootstrap' in (first / 'provenance.json').read_text()
    main = list(csv.DictReader((first / 'table2.csv').open()))
    assert [round(float(r['delta_pp']), 3) for r in main] == [1.683, 7.563, 11.160, 7.141]
    langs = list(csv.DictReader((first / 'table6.csv').open()))
    assert float(langs[0]['baseline_sample_sd']) == pytest.approx(0.004567084227907097)
    assert json.loads((first / 'provenance.json').read_text())['summary_validation'] == 'PASS: absolute means, paired differences, and sample SD'


def test_output_cannot_overwrite_frozen_and_collision_is_fresh(tmp_path):
    r = wrapper()
    with pytest.raises(ValueError):
        r.reserve_output(tmp_path, tmp_path / 'results', 'smoke')
    a = r.reserve_output(tmp_path, None, 'smoke')
    (a / 'marker').write_text('keep')
    b = r.reserve_output(tmp_path, a, 'smoke')
    assert a != b and (a / 'marker').read_text() == 'keep'


def test_missing_audit_inputs_writes_failure_report(tmp_path):
    r = wrapper()
    assert r.main(['--root', str(tmp_path), '--mode', 'audit']) == 1
    report = json.loads((tmp_path / 'audit/recomputed/audit/report.json').read_text())
    assert report['status'] == 'FAIL'
    assert report['manuscript']['status'] == 'FAIL'
    assert report['isolation']['status'] == 'FAIL'


def test_missing_saved_input_fails(tmp_path):
    with pytest.raises((OSError, ValueError)):
        wrapper().run_saved(tmp_path, tmp_path, 'table2')


def test_smoke_uses_actual_fixed_pipeline_and_train_scaling(tmp_path, monkeypatch):
    r = wrapper()
    assert (ROOT / 'scripts/reproduce_smoke.py').is_file(), 'smoke runner missing'
    import reproduce_smoke
    from lpcode_v1 import experiment
    import numpy as np
    fitted = []
    original = experiment.build_model

    def tracked(name, seed):
        model = original(name, seed)
        fit = model.fit
        def checked_fit(x, y):
            result = fit(x, y)
            np.testing.assert_allclose(model.named_steps['scaler'].mean_, np.mean(x, axis=0))
            assert model.named_steps['model'].n_estimators == 300
            fitted.append(x.shape[1])
            return result
        model.fit = checked_fit
        return model
    monkeypatch.setattr(experiment, 'build_model', tracked)
    report = reproduce_smoke.run(ROOT, tmp_path)
    assert report['status'] == 'PASS'
    assert fitted == [20, 30]
    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    assert all(0 <= m['f1'] <= 1 and m['fit_seconds'] > 0 for m in metrics)
    pairs = json.loads((tmp_path / 'pair_manifest.json').read_text())
    assert pairs['endpoint_leakage_count'] == pairs['content_leakage_count'] == 0
    assert len(list(csv.DictReader((tmp_path / 'metrics.csv').open()))) == 2
    assert (tmp_path / 'pair_manifest.csv').stat().st_size > 100
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['files']['config.json'] == r.artifact_hashes(tmp_path)['config.json']
    assert json.loads((tmp_path / 'config.json').read_text())['model_parameters']['n_estimators'] == 300


def test_saved_mode_never_fits(tmp_path, monkeypatch):
    r = wrapper()
    from lpcode_v1 import experiment
    def forbidden(*args, **kwargs):
        raise AssertionError('saved mode trained a model')
    monkeypatch.setattr(experiment, 'evaluate_fold', forbidden)
    assert r.run_saved(ROOT, tmp_path, 'all-saved')['status'] == 'PASS'


def test_duplicate_ledger_cell_rejected(tmp_path, monkeypatch):
    r = wrapper()
    import reproduce_saved
    original = reproduce_saved.load_ledger
    def duplicate(root, path, hashes):
        rows = original(root, path, hashes)
        return rows + [rows[0]]
    monkeypatch.setattr(reproduce_saved, 'load_ledger', duplicate)
    with pytest.raises(ValueError, match='duplicate cell'):
        r.run_saved(ROOT, tmp_path, 'all-saved')


def test_audit_reruns_both_independent_audits(monkeypatch):
    r = wrapper()
    import audit_isolation
    import audit_manuscript_evidence
    called = []
    def manuscript(root):
        called.append('manuscript'); return {'status': 'PASS'}
    def isolation(root):
        called.append('isolation'); return {'status': 'BLOCKED'}
    monkeypatch.setattr(audit_manuscript_evidence, 'run_audit', manuscript)
    monkeypatch.setattr(audit_isolation, 'run_audit', isolation)
    assert r.run_audit(ROOT)['status'] == 'FAIL'
    assert called == ['manuscript', 'isolation']


@pytest.mark.parametrize('change', ['value', 'missing', 'duplicate', 'schema'])
def test_mechanism_aggregate_mismatch_rejected(change):
    wrapper()
    import reproduce_saved
    assert hasattr(reproduce_saved, 'validate_mechanism'), 'mechanism aggregate validation missing'
    base = ROOT / 'results/05_mechanism_analysis/feature_importance'
    summary = json.loads((base / 'mechanism_summary.json').read_text())
    rows = list(csv.DictReader((base / 'grouped_importance.csv').open()))
    reproduce_saved.validate_mechanism(summary, rows)
    if change == 'value': rows[0]['permutation_mean'] = '0.5'
    elif change == 'missing': rows.pop()
    elif change == 'duplicate': rows.append(dict(rows[0]))
    else: rows[0]['unexpected_column'] = 'invalid'
    with pytest.raises(ValueError, match='mechanism'):
        reproduce_saved.validate_mechanism(summary, rows)


def test_output_parent_file_returns_failure(tmp_path, capsys):
    r = wrapper()
    (tmp_path / 'audit').write_text('existing file')
    assert r.main(['--root', str(tmp_path), '--mode', 'table2']) == 1
    assert capsys.readouterr().err


def test_output_oserror_returns_failure(tmp_path, capsys, monkeypatch):
    r = wrapper()
    def denied(*args):
        raise PermissionError('output unavailable')
    monkeypatch.setattr(r, 'reserve_output', denied)
    assert r.main(['--root', str(tmp_path), '--mode', 'table2']) == 1
    assert 'output unavailable' in capsys.readouterr().err
