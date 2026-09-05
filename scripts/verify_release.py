"""No-fit frozen evidence, numerical and public inventory validation."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'repro/2502.17749/v1'))
from lpcode_v1.integrity_audit import audit_integrity
from lpcode_v1.paper_audit import canonical_percentages
registry=ROOT/'results/06_paper_assets/frozen_result_registry.json'
report=audit_integrity(registry)
values=canonical_percentages(registry)
expected={'strict_clean_pp':1.6831184559271604,'unseen_llm_pp':7.563096014904111,'combined_attack_pp':11.160181574646067,'cross_language_pp':7.140783186266682}
for key,value in expected.items():assert abs(values[key]-value)<1e-10,(key,values[key])
for row in json.loads((ROOT/'SOURCE_INVENTORY.json').read_text())['files']:
    assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['release_sha256'],row['path']
print(json.dumps({'status':report['status'],'gates':report['gates'],'ablation':report['ablation'],'numerical_values':values,'frozen_registry_sha256':report['registry_sha256'],'model_fitting':False},indent=2))
