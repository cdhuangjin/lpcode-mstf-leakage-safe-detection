"""Read historical absolute paths relative to this public checkout."""
from pathlib import Path
from .paths import WORKSPACE_ROOT

def resolve_recorded_path(value):
    current=Path(value)
    if current.is_absolute() and current.resolve().is_relative_to(WORKSPACE_ROOT.resolve()):
        return current
    text=str(value).replace('\\','/')
    prefix='C:/Users/PC/Documents/Codex/Reproduction005/'
    if text.startswith(prefix):
        path=(WORKSPACE_ROOT/text[len(prefix):]).resolve()
        if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
            raise ValueError('Recorded path escapes public checkout')
        return path
    return Path(value)
