#!/usr/bin/env python3
"""Run tests/*_test.py modules in N worker processes; same totals as `unittest discover`, non-zero exit on any failure.
Usage: test_parallel.py [-j N] [module_test.py ...]   (default N = CPU count - 2)"""
import os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARD = {"v3_line_endings_test.py", "v3_inject_test.py", "v3_update_test.py"}   # slow modules: one unit per test method, so one module is not the critical path
SERIAL = set()   # module names that must not overlap with others (shared global state); run alone after the parallel batch

def run(unit):
    mod, k = unit
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", mod] + (["-k", k] if k else []), cwd=ROOT, capture_output=True, text=True)
    out = p.stderr
    n = re.search(r"^Ran (\d+) tests?", out, re.M)
    f = {k: int(v) for k, v in re.findall(r"(failures|errors|skipped)=(\d+)", out.splitlines()[-1] if out.strip() else "")}
    return mod + (f"::{k}" if k else ""), p.returncode, int(n.group(1)) if n else 0, f, out, time.time() - t

def main():
    a = sys.argv[1:]
    jobs = max(1, (os.cpu_count() or 4) - 2)
    if a[:1] == ["-j"]:
        jobs, a = int(a[1]), a[2:]
    mods = a or sorted(p.name for p in (ROOT / "tests").glob("*_test.py"))
    t0 = time.time()
    par = [m for m in mods if m not in SERIAL]
    units = [(m, k) for m in par for k in (re.findall(r"^    def (test\w*)\(", (ROOT / "tests" / m).read_text(), re.M) if m in SHARD else [""])]
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(run, sorted(units, key=lambda u: (-bool(u[1]), -(ROOT / "tests" / u[0]).stat().st_size))))
    res += [run((m, "")) for m in mods if m in SERIAL]
    tot = {"failures": 0, "errors": 0, "skipped": 0}
    ran, bad = 0, False
    for mod, rc, n, f, out, dt in res:
        ran += n
        for k in tot:
            tot[k] += f.get(k, 0)
        if rc or not n:
            bad = True
            print(f"=== {mod} (exit {rc}, {dt:.0f}s)\n{out}")
    print("slowest:", ", ".join(f"{m} {dt:.0f}s" for m, *_, dt in sorted(res, key=lambda r: -r[-1])[:5]))
    print(f"Ran {ran} tests in {time.time() - t0:.1f}s ({jobs} workers, {len(mods)} modules)")
    print(("FAILED" if bad else "OK") + f" (failures={tot['failures']}, errors={tot['errors']}, skipped={tot['skipped']})")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
