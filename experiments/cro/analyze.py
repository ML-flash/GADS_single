"""Aggregate CRO seed-level outcomes and run paired sign-flip tests.

Generations and organisms are trajectory samples, not replicates.  The seed is
the statistical unit.  Every test below first reduces each seed to one contrast.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import config as CFG

HERE = Path(__file__).resolve().parent


def load_phase(root, phase):
    paths = sorted((root / phase).glob("seed_*.json"))
    out = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("experiment") != CFG.EXPERIMENT_ID or d.get("phase") != phase:
            raise RuntimeError("wrong experiment product: %s" % p)
        if d.get("canonical_gads_blob") != CFG.CANONICAL_GADS_BLOB:
            raise RuntimeError("canonical GADS mismatch: %s" % p)
        out.append(d)
    return out


def finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def endpoint(traj, metric):
    vals = [r.get(metric) for r in traj if finite(r.get(metric))]
    return vals[-1] if vals else None


def window_mean(traj, metric, which, frac=0.20):
    vals = [r.get(metric) for r in traj if finite(r.get(metric))]
    if not vals:
        return None
    k = max(1, int(math.ceil(len(vals) * frac)))
    use = vals[:k] if which == "first" else vals[-k:]
    return sum(use) / float(len(use))


def drift(traj, metric):
    a = window_mean(traj, metric, "first")
    b = window_mean(traj, metric, "last")
    return None if a is None or b is None else b - a


def paired_values(records, fn):
    vals = []
    seeds = []
    for r in records:
        v = fn(r)
        if finite(v):
            vals.append(float(v))
            seeds.append(int(r["seed"]))
    return seeds, vals


def signflip_p(values, alternative="greater", monte_carlo=200000, seed=20260922):
    """One-sample paired sign-flip test against zero on seed-level contrasts."""
    vals = [float(v) for v in values]
    n = len(vals)
    if not n:
        return None
    obs = sum(vals) / float(n)

    def extreme(x):
        if alternative == "greater":
            return x >= obs - 1e-15
        if alternative == "less":
            return x <= obs + 1e-15
        return abs(x) >= abs(obs) - 1e-15

    if n <= 20:
        ge = total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            m = sum(s * v for s, v in zip(signs, vals)) / float(n)
            total += 1
            ge += int(extreme(m))
        return ge / float(total)

    rng = random.Random(seed)
    ge = 1
    total = 1
    for _ in range(monte_carlo):
        m = sum((1.0 if rng.random() < 0.5 else -1.0) * v for v in vals) / float(n)
        total += 1
        ge += int(extreme(m))
    return ge / float(total)


def loo_sign_stable(values, direction="positive"):
    if len(values) < 2:
        return False
    for i in range(len(values)):
        xs = values[:i] + values[i + 1:]
        m = sum(xs) / float(len(xs))
        if direction == "positive" and m <= 0:
            return False
        if direction == "negative" and m >= 0:
            return False
    return True


def summarize(name, seeds, vals, alternative="greater"):
    mean = sum(vals) / float(len(vals)) if vals else None
    sd = None
    if len(vals) > 1:
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / float(len(vals) - 1))
    direction = (
        "positive" if alternative == "greater"
        else "negative" if alternative == "less"
        else None
    )
    return {
        "name": name,
        "n": len(vals),
        "seeds": seeds,
        "mean": mean,
        "sd": sd,
        "p_signflip": signflip_p(vals, alternative=alternative),
        "leave_one_out_sign_stable": (
            loo_sign_stable(vals, direction) if direction is not None else None
        ),
        "values": vals,
    }


def geometry_tests(records):
    tests = []

    seeds, vals = paired_values(
        records,
        lambda r: (
            drift(
                r["conditions"]["ellipse_ko"],
                "mean_low_curvature_progress_ridge",
            )
            - drift(
                r["conditions"]["circle_ko"],
                "mean_low_curvature_progress_ridge",
            )
        ),
    )
    tests.append(summarize(
        "G1 paper analogue: ellipse KO excess drift toward lower curvature > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: (
            drift(
                r["conditions"]["ellipse_full"],
                "mean_low_curvature_progress_ridge",
            )
            - drift(
                r["conditions"]["circle_full"],
                "mean_low_curvature_progress_ridge",
            )
        ),
    )
    tests.append(summarize(
        "G2 full GADS excess drift toward lower curvature > 0",
        seeds, vals, alternative="greater"
    ))

    # Secondary descriptive curvature reading.  Starting away from a curvature
    # extremum avoids the trivial "any diffusion lowers curvature" artifact,
    # but the signed circle-subtracted progress above remains the primary test.
    seeds, vals = paired_values(
        records,
        lambda r: drift(
            r["conditions"]["ellipse_ko"], "mean_curvature_ridge"
        ),
    )
    tests.append(summarize(
        "G1b ellipse KO curvature drift < 0 (secondary)",
        seeds, vals, alternative="less"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: (
            drift(r["conditions"]["ellipse_full"], "proposal_robustness")
            - drift(r["conditions"]["ellipse_ko"], "proposal_robustness")
        ),
    )
    tests.append(summarize(
        "G3 GADS excess proposal-robustness drift > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: (
            endpoint(r["conditions"]["ellipse_full"], "proposal_robustness")
            - endpoint(r["conditions"]["ellipse_flat_lock"], "proposal_robustness")
        ),
    )
    tests.append(summarize(
        "G4 intact minus FLAT-LOCK proposal robustness > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: (
            endpoint(r["conditions"]["ellipse_flat_recover"], "proposal_robustness")
            - endpoint(r["conditions"]["ellipse_flat_lock"], "proposal_robustness")
        ),
    )
    tests.append(summarize(
        "G5 FLAT-RECOVER minus FLAT-LOCK proposal robustness > 0",
        seeds, vals, alternative="greater"
    ))

    # Representation-blind control: report, do not use as support gate.
    seeds, vals = paired_values(
        records,
        lambda r: (
            endpoint(r["conditions"]["ellipse_full"], "atomic_robustness")
            - endpoint(r["conditions"]["ellipse_flat_lock"], "atomic_robustness")
        ),
    )
    tests.append(summarize(
        "G-control intact minus FLAT-LOCK atomic robustness",
        seeds, vals, alternative="two-sided"
    ))

    return tests


def mux_tests(records):
    tests = []

    seeds, vals = paired_values(
        records,
        lambda r: drift(r["conditions"]["mux_full"], "proposal_robustness"),
    )
    tests.append(summarize(
        "M1 neutral MUX full-GADS proposal robustness drift > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: drift(r["conditions"]["mux_full"], "canonical_robustness"),
    )
    tests.append(summarize(
        "M2 neutral MUX full-GADS canonical-pass robustness drift > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: (
            endpoint(r["conditions"]["mux_full"], "proposal_robustness")
            - endpoint(r["conditions"]["mux_flat_lock"], "proposal_robustness")
        ),
    )
    tests.append(summarize(
        "M3 intact minus FLAT-LOCK proposal robustness > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: (
            endpoint(r["conditions"]["mux_flat_recover"], "proposal_robustness")
            - endpoint(r["conditions"]["mux_flat_lock"], "proposal_robustness")
        ),
    )
    tests.append(summarize(
        "M4 FLAT-RECOVER minus FLAT-LOCK proposal robustness > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: drift(r["conditions"]["mux_full"], "compression_ratio"),
    )
    tests.append(summarize(
        "M5 full-GADS compression-ratio drift > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: (
            endpoint(r["conditions"]["mux_full"], "atomic_robustness")
            - endpoint(r["conditions"]["mux_flat_lock"], "atomic_robustness")
        ),
    )
    tests.append(summarize(
        "M-control intact minus FLAT-LOCK atomic robustness",
        seeds, vals, alternative="two-sided"
    ))
    return tests


def print_tests(title, tests):
    print("\n%s" % title)
    print("%-58s %4s %11s %11s %10s" % ("contrast", "n", "mean", "p", "LOO sign"))
    print("-" * 100)
    for t in tests:
        mean = "NA" if t["mean"] is None else "%.6g" % t["mean"]
        p = "NA" if t["p_signflip"] is None else "%.6g" % t["p_signflip"]
        print("%-58s %4d %11s %11s %10s" % (
            t["name"][:58], t["n"], mean, p,
            ("NA" if t["leave_one_out_sign_stable"] is None
             else "yes" if t["leave_one_out_sign_stable"] else "no"),
        ))


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze CRO seed-level contrasts")
    p.add_argument("--root", type=Path, default=HERE / CFG.RESULTS_DIR)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    geometry = load_phase(args.root, "geometry")
    mux = load_phase(args.root, "mux")
    gt = geometry_tests(geometry) if geometry else []
    mt = mux_tests(mux) if mux else []
    if gt:
        print_tests("GEOMETRY", gt)
    if mt:
        print_tests("MUX11", mt)

    result = {
        "experiment": CFG.EXPERIMENT_ID,
        "statistical_unit": "seed/population",
        "geometry_n": len(geometry),
        "mux_n": len(mux),
        "geometry_tests": gt,
        "mux_tests": mt,
    }
    out = args.out or (args.root / "analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
