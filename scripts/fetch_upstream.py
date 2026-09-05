"""Fetch third-party code/data from its owner at the recorded commit."""
from pathlib import Path
import subprocess
import shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'repro/2502.17749/code'
COMMIT='b3660c8262ae57e14498528119607ee673d4257a'
URL='https://github.com/Shinwoo-Park/LPcode.git'
if TARGET.exists():
    actual=subprocess.check_output(['git','-C',str(TARGET),'rev-parse','HEAD'],text=True).strip()
    if actual!=COMMIT:raise SystemExit('Existing checkout has another revision; will not overwrite it')
else:
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout',URL,str(TARGET)],check=True)
    subprocess.run(['git','-C',str(TARGET),'checkout','--detach',COMMIT],check=True)
print('Pinned upstream:',COMMIT)
for source in (ROOT/'results/00_official_baseline/metric_pickles').glob('*/*.pkl'):
    target=TARGET/'experiment'/source.parent.name/source.name
    if target.exists() and hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(source.read_bytes()).digest():
        raise SystemExit('Existing baseline metric file differs; refusing overwrite: '+str(target))
    if not target.exists():shutil.copy2(source,target)
print('Upstream has no identified open licence. Consult its owner for reuse permissions; this release grants no upstream rights.')
