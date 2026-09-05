from pathlib import Path
import importlib.util
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'repro/2502.17749/v1'))

def test_release_path_adapter_exists():
    assert importlib.util.find_spec('lpcode_v1') is not None, 'Complete research package must be shipped'
    assert importlib.util.find_spec('lpcode_v1.release_paths') is not None, 'Portable registry reader missing'

def test_historical_paths_are_local_and_cannot_escape():
    from lpcode_v1.release_paths import resolve_recorded_path
    target=resolve_recorded_path('C:/Users/PC/Documents/Codex/Reproduction005/results/a.json')
    assert target == ROOT/'results/a.json'
    with pytest.raises(ValueError):
        resolve_recorded_path('C:/Users/PC/Documents/Codex/Reproduction005/../../outside')

def test_complete_source_inventory():
    import json,hashlib
    inventory=ROOT/'SOURCE_INVENTORY.json'
    assert inventory.is_file(), 'Complete source inventory missing'
    for row in json.loads(inventory.read_text())['files']:
        p=ROOT/row['path'];assert p.is_file(), row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['release_sha256'], row['path']

def test_no_private_figure_dependency():
    from lpcode_v1 import paper_assets
    assert callable(paper_assets.build_paper_assets)
