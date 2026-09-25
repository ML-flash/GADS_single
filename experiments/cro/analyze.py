"""Aggregate CRO seed-level outcomes and run paired sign-flip tests.

Generations and organisms are trajectory samples, not replicates.  The seed is
the statistical unit.  Every test below first reduces each seed to one contrast.
"""
from __future__ import annotations

import argparse
import csv
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
        if phase == "mux" and d.get("probe_rng_scheme") != "family_and_trial_isolated_v1":
            raise RuntimeError("MUX probe RNG scheme mismatch: %s" % p)
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



def mean_sd(vals):
    vals = [float(v) for v in vals if finite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / float(len(vals))
    sd = None
    if len(vals) > 1:
        sd = math.sqrt(
            sum((v - mean) ** 2 for v in vals) / float(len(vals) - 1)
        )
    return mean, sd


def trajectory_by_generation(traj):
    return {
        int(row["generation"]): row
        for row in traj
        if "generation" in row
    }


def first_positive_run(rows, mean_key, n_key, run_length=5):
    """First checkpoint starting a descriptive run of positive aggregate means."""
    for i in range(0, max(0, len(rows) - run_length + 1)):
        window = rows[i:i + run_length]
        if all(
            finite(row.get(mean_key))
            and int(row.get(n_key, 0)) > 0
            and row[mean_key] > 0.0
            for row in window
        ):
            return int(rows[i]["generation"])
    return None


def geometry_trajectory_diagnostics(records):
    """Time-resolved descriptive diagnostics; seeds remain the replicate unit.

    G2(t) is the seed-level, baseline-adjusted ellipse-full minus circle-full
    low-curvature progress contrast.  G3(t) is the seed-level,
    baseline-adjusted ellipse-full minus ellipse-KO proposal-robustness
    contrast.  Structural columns are contemporaneous ellipse-full means.

    These rows are descriptive trajectory summaries, not additional
    inferential tests.
    """
    if not records:
        return None

    prepared = []
    common_generations = None
    for rec in records:
        cond = rec["conditions"]
        ef = trajectory_by_generation(cond["ellipse_full"])
        cf = trajectory_by_generation(cond["circle_full"])
        ek = trajectory_by_generation(cond["ellipse_ko"])
        gens = set(ef) & set(cf) & set(ek)
        common_generations = gens if common_generations is None else common_generations & gens
        prepared.append((int(rec["seed"]), ef, cf, ek))

    generations = sorted(common_generations or [])
    rows = []
    structural_metrics = (
        "mco_size",
        "compression_ratio",
        "composition_fraction",
        "mean_composition_depth",
        "max_composition_depth",
        "mean_dH_per_payload",
        "mean_encoded_payload",
        "mean_decoded_len",
    )

    for generation in generations:
        g2_vals = []
        g3_vals = []
        struct_vals = {metric: [] for metric in structural_metrics}

        for seed, ef, cf, ek in prepared:
            ef0 = ef.get(generations[0])
            cf0 = cf.get(generations[0])
            ek0 = ek.get(generations[0])
            erf = ef[generation]
            crf = cf[generation]
            erk = ek[generation]

            p = "mean_low_curvature_progress_ridge"
            if (
                ef0 is not None and cf0 is not None
                and finite(erf.get(p)) and finite(crf.get(p))
                and finite(ef0.get(p)) and finite(cf0.get(p))
            ):
                g2_vals.append(
                    (erf[p] - ef0[p]) - (crf[p] - cf0[p])
                )

            p = "proposal_robustness"
            if (
                ef0 is not None and ek0 is not None
                and finite(erf.get(p)) and finite(erk.get(p))
                and finite(ef0.get(p)) and finite(ek0.get(p))
            ):
                g3_vals.append(
                    (erf[p] - ef0[p]) - (erk[p] - ek0[p])
                )

            for metric in structural_metrics:
                if finite(erf.get(metric)):
                    struct_vals[metric].append(float(erf[metric]))

        g2_mean, g2_sd = mean_sd(g2_vals)
        g3_mean, g3_sd = mean_sd(g3_vals)
        row = {
            "generation": generation,
            "g2_n": len(g2_vals),
            "g2_mean": g2_mean,
            "g2_sd": g2_sd,
            "g2_positive_fraction": (
                sum(v > 0.0 for v in g2_vals) / float(len(g2_vals))
                if g2_vals else None
            ),
            "g3_n": len(g3_vals),
            "g3_mean": g3_mean,
            "g3_sd": g3_sd,
            "g3_positive_fraction": (
                sum(v > 0.0 for v in g3_vals) / float(len(g3_vals))
                if g3_vals else None
            ),
        }
        for metric in structural_metrics:
            mean, sd = mean_sd(struct_vals[metric])
            row["ellipse_full_%s_mean" % metric] = mean
            row["ellipse_full_%s_sd" % metric] = sd
        rows.append(row)

    g2_onset = first_positive_run(rows, "g2_mean", "g2_n", run_length=5)
    g3_onset = first_positive_run(rows, "g3_mean", "g3_n", run_length=5)

    def snapshot(generation):
        if generation is None:
            return None
        for row in rows:
            if row["generation"] == generation:
                return row
        return None

    return {
        "status": "descriptive_only",
        "definition": {
            "g2_time_resolved": (
                "baseline-adjusted ellipse_full minus circle_full "
                "mean_low_curvature_progress_ridge"
            ),
            "g3_time_resolved": (
                "baseline-adjusted ellipse_full minus ellipse_ko "
                "proposal_robustness"
            ),
            "onset_rule": "first 5 consecutive checkpoints with aggregate mean > 0",
            "structural_state": "contemporaneous ellipse_full checkpoint means",
        },
        "g2_first_5_checkpoint_positive_run": g2_onset,
        "g3_first_5_checkpoint_positive_run": g3_onset,
        "g2_onset_snapshot": snapshot(g2_onset),
        "g3_onset_snapshot": snapshot(g3_onset),
        "rows": rows,
    }


def print_geometry_trajectory(diag):
    if not diag or not diag["rows"]:
        return
    rows = diag["rows"]
    wanted = {
        rows[0]["generation"],
        rows[-1]["generation"],
        rows[len(rows) // 4]["generation"],
        rows[len(rows) // 2]["generation"],
        rows[(3 * len(rows)) // 4]["generation"],
        diag.get("g2_first_5_checkpoint_positive_run"),
        diag.get("g3_first_5_checkpoint_positive_run"),
    }
    wanted.discard(None)

    print("\nGEOMETRY TRAJECTORY DIAGNOSTICS (descriptive)")
    print(
        "%7s %10s %10s %8s %8s %8s %10s"
        % ("gen", "G2(t)", "G3(t)", "MG", "compress", "depth", "mean_dH")
    )
    print("-" * 78)
    for row in rows:
        if row["generation"] not in wanted:
            continue
        def fmt(key):
            v = row.get(key)
            return "NA" if not finite(v) else "%.5g" % v
        print(
            "%7d %10s %10s %8s %8s %8s %10s"
            % (
                row["generation"],
                fmt("g2_mean"),
                fmt("g3_mean"),
                fmt("ellipse_full_mco_size_mean"),
                fmt("ellipse_full_compression_ratio_mean"),
                fmt("ellipse_full_mean_composition_depth_mean"),
                fmt("ellipse_full_mean_dH_per_payload_mean"),
            )
        )

    print(
        "G2 first 5-checkpoint positive run: %s"
        % (
            diag["g2_first_5_checkpoint_positive_run"]
            if diag["g2_first_5_checkpoint_positive_run"] is not None
            else "none"
        )
    )
    print(
        "G3 first 5-checkpoint positive run: %s"
        % (
            diag["g3_first_5_checkpoint_positive_run"]
            if diag["g3_first_5_checkpoint_positive_run"] is not None
            else "none"
        )
    )


def write_geometry_trajectory_csv(path, diag):
    if not diag or not diag["rows"]:
        return
    rows = diag["rows"]
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mux_tests(records):
    tests = []

    # Direct causal fork diagnostics: identical decoded programs immediately
    # before vs. after flattening, with exactly paired per-trial perturbations.
    seeds, vals = paired_values(
        records,
        lambda r: r["fork_probe"]["delta_before_minus_flatten"]["proposal_robustness"],
    )
    tests.append(summarize(
        "M0a immediate flatten proposal-robustness loss > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: r["fork_probe"]["delta_before_minus_flatten"]["canonical_robustness"],
    )
    tests.append(summarize(
        "M0b immediate flatten canonical-robustness loss > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: r["fork_probe"]["delta_before_minus_flatten"]["atomic_robustness"],
    )
    tests.append(summarize(
        "M0-control immediate flatten atomic robustness",
        seeds, vals, alternative="two-sided"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: drift(r["conditions"]["mux_full"], "proposal_robustness"),
    )
    tests.append(summarize(
        "M1 degenerate-peak MUX full-GADS proposal robustness drift > 0",
        seeds, vals, alternative="greater"
    ))

    seeds, vals = paired_values(
        records,
        lambda r: drift(r["conditions"]["mux_full"], "canonical_robustness"),
    )
    tests.append(summarize(
        "M2 degenerate-peak MUX full-GADS canonical-pass robustness drift > 0",
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
    geometry_trajectory = geometry_trajectory_diagnostics(geometry) if geometry else None
    if gt:
        print_tests("GEOMETRY", gt)
        print_geometry_trajectory(geometry_trajectory)
    if mt:
        print_tests("MUX6", mt)

    result = {
        "experiment": CFG.EXPERIMENT_ID,
        "statistical_unit": "seed/population",
        "geometry_n": len(geometry),
        "mux_n": len(mux),
        "geometry_tests": gt,
        "geometry_trajectory": geometry_trajectory,
        "mux_tests": mt,
    }
    out = args.out or (args.root / "analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if geometry_trajectory:
        trajectory_csv = args.root / "geometry_trajectory.csv"
        write_geometry_trajectory_csv(trajectory_csv, geometry_trajectory)
        print("\nwrote %s" % trajectory_csv)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
