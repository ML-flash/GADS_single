"""Run the CRO matched seed panel with bounded local parallelism."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
from pathlib import Path

import config as CFG

HERE = Path(__file__).resolve().parent


def parse_seeds(text):
    if not text:
        return list(CFG.SEEDS)
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def one(phase, seed, probe_orgs, probes_per_org):
    script = HERE / ("run_geometry.py" if phase == "geometry" else "run_mux_orbit.py")
    cmd = [
        sys.executable,
        str(script),
        "--seed", str(seed),
        "--probe-orgs", str(probe_orgs),
        "--probes-per-org", str(probes_per_org),
    ]
    proc = subprocess.run(cmd, cwd=str(HERE), text=True)
    if proc.returncode:
        raise RuntimeError("%s seed %d exited %d" % (phase, seed, proc.returncode))
    return phase, seed


def main(argv=None):
    p = argparse.ArgumentParser(description="Run CRO matched seed panel")
    p.add_argument("--phase", choices=("geometry", "mux", "both"), default="both")
    p.add_argument("--seeds", default="1-16", help="e.g. 1-16 or 1,3,5")
    p.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--probe-orgs", type=int, default=CFG.PROBE_ORGS)
    p.add_argument("--probes-per-org", type=int, default=CFG.PROBES_PER_ORG)
    args = p.parse_args(argv)

    phases = ["geometry", "mux"] if args.phase == "both" else [args.phase]
    work = [(phase, seed) for phase in phases for seed in parse_seeds(args.seeds)]
    print("CRO: %d runs, jobs=%d" % (len(work), args.jobs), flush=True)

    failures = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(one, phase, seed, args.probe_orgs, args.probes_per_org):
            (phase, seed)
            for phase, seed in work
        }
        for fut in cf.as_completed(futs):
            phase, seed = futs[fut]
            try:
                fut.result()
                print("DONE %-8s seed %d" % (phase, seed), flush=True)
            except Exception as exc:
                failures.append((phase, seed, str(exc)))
                print("FAIL %-8s seed %d: %s" % (phase, seed, exc), flush=True)

    if failures:
        raise SystemExit("failed runs: %r" % failures)


if __name__ == "__main__":
    main()
