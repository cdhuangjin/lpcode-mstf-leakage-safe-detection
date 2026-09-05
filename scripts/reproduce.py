"""Public reproduction entry point. Outputs are confined to audit/recomputed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'repro/2502.17749/v1'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def artifact_hashes(output):
    return {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'}


def reserve_output(root, requested, mode):
    allowed = (root / 'audit/recomputed').resolve()
    base = (requested or allowed / mode).resolve()
    if not base.is_relative_to(allowed) or base == allowed:
        raise ValueError('--output must be a child directory of <root>/audit/recomputed')
    base.parent.mkdir(parents=True, exist_ok=True)
    for number in range(100000):
        output = base if number == 0 else base.with_name(base.name + f'-{number:03d}')
        try:
            output.mkdir(parents=True, exist_ok=False)
            return output
        except FileExistsError:
            continue
    raise ValueError('output namespace exhausted')


def run_saved(root, output, mode):
    from reproduce_saved import run
    return run(root, output, mode)


def run_audit(root):
    import audit_manuscript_evidence
    import audit_isolation
    reports = {}
    for name, runner in [('manuscript', audit_manuscript_evidence.run_audit), ('isolation', audit_isolation.run_audit)]:
        try:
            reports[name] = runner(root)
        except Exception as exc:
            reports[name] = {'status': 'FAIL', 'error': str(exc), 'error_type': type(exc).__name__}
    return {'status': 'PASS' if all(r['status'] == 'PASS' for r in reports.values()) else 'FAIL', **reports}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['smoke', 'audit', 'table2', 'table3', 'all-saved'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        output = reserve_output(root, args.output, args.mode)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    started = time.perf_counter()
    try:
        if args.mode == 'audit':
            report = run_audit(root)
        elif args.mode == 'smoke':
            from reproduce_smoke import run
            report = run(root, output)
        else:
            report = run_saved(root, output, args.mode)
    except Exception as exc:
        report = {'status': 'FAIL', 'error': str(exc), 'error_type': type(exc).__name__}
    report.update(mode=args.mode, elapsed_seconds=time.perf_counter() - started)
    write_json(output / 'report.json', report)
    write_json(output / 'artifact_hashes.json', {'algorithm': 'SHA-256', 'scope': 'All files in this run directory; manifest itself excluded. Runtime-dependent files may differ.', 'files': artifact_hashes(output)})
    print(f"{report['status']}: {output}")
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
