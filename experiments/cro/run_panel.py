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


def one(phase, seed, probe_orgs, probes_per_org, generations=None,
        fork_generation=None, checkpoint_every=None, out_root=None):
    script = HERE / ("run_geometry.py" if phase == "geometry" else "run_mux_orbit.py")
    cmd = [
        sys.executable,
        str(script),
        "--seed", str(seed),
        "--probe-orgs", str(probe_orgs),
        "--probes-per-org", str(probes_per_org),
    ]
    if generations is not None:
        cmd += ["--generations", str(generations)]
    if fork_generation is not None:
        cmd += ["--fork-generation", str(fork_generation)]
    if checkpoint_every is not None:
        cmd += ["--checkpoint-every", str(checkpoint_every)]
    if out_root is not None:
        cmd += ["--out", str(out_root / phase / ("seed_%03d.json" % seed))]
    proc = subprocess.run(cmd, cwd=str(HERE), text=True)
    if proc.returncode:
        raise RuntimeError("%s seed %d exited %d" % (phase, seed, proc.returncode))
    return phase, seed


def main(argv=None):
    p = argparse.ArgumentParser(description="Run CRO matched seed panel")
    p.add_argument("--phase", choices=("geometry", "mux", "both"), default="both")
    p.add_argument("--seeds", default=None, help="e.g. 1-16 or 1,3,5")
    p.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--probe-orgs", type=int)
    p.add_argument("--probes-per-org", type=int)
    p.add_argument(
        "--pilot", action="store_true",
        help="short exploratory panel: 4 seeds, 750 generations, fork 375, "
             "25-generation checkpoints, lighter read-only probes, separate output"
    )
    p.add_argument(
        "--geometry-long", action="store_true",
        help="duration follow-up: geometry only, 4 matched seeds, 3000 generations, "
             "fork 1500, 25-generation checkpoints, lighter read-only probes"
    )
    args = p.parse_args(argv)

    if args.geometry_long:
        if args.phase not in ("geometry", "both"):
            raise ValueError("--geometry-long requires --phase geometry or both")
        seeds = parse_seeds(args.seeds or "1-4")
        generations = CFG.GEOMETRY_GENERATIONS
        fork_generation = CFG.GEOMETRY_FORK_GENERATION
        checkpoint_every = 25
        probe_orgs = args.probe_orgs if args.probe_orgs is not None else 8
        probes_per_org = (
            args.probes_per_org if args.probes_per_org is not None else 2
        )
        out_root = HERE / CFG.RESULTS_DIR / "geometry_long"
        profile = "geometry-long"
        args.phase = "geometry"
    elif args.pilot:
        seeds = parse_seeds(args.seeds or "1-4")
        generations = 750
        fork_generation = 375
        checkpoint_every = 25
        probe_orgs = args.probe_orgs if args.probe_orgs is not None else 8
        probes_per_org = (
            args.probes_per_org if args.probes_per_org is not None else 2
        )
        out_root = HERE / CFG.RESULTS_DIR / "pilot"
        profile = "pilot"
    else:
        seeds = parse_seeds(args.seeds or "1-16")
        generations = None
        fork_generation = None
        checkpoint_every = None
        probe_orgs = (
            args.probe_orgs if args.probe_orgs is not None else CFG.PROBE_ORGS
        )
        probes_per_org = (
            args.probes_per_org
            if args.probes_per_org is not None else CFG.PROBES_PER_ORG
        )
        out_root = None
        profile = "full"

    phases = ["geometry", "mux"] if args.phase == "both" else [args.phase]
    work = [(phase, seed) for phase in phases for seed in seeds]
    print(
        "CRO %s: %d runs, jobs=%d, seeds=%s"
        % (profile, len(work), args.jobs, ",".join(map(str, seeds))),
        flush=True,
    )
    if args.pilot or args.geometry_long:
        print(
            "%s: generations=%d fork=%d checkpoint=%d probes=%dx%d"
            % (
                profile, generations, fork_generation, checkpoint_every,
                probe_orgs, probes_per_org,
            ),
            flush=True,
        )

    failures = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(
                one, phase, seed, probe_orgs, probes_per_org,
                generations, fork_generation, checkpoint_every, out_root,
            ): (phase, seed)
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
