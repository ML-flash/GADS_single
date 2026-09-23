"""CRO Phase A: curvature drift on a controlled degenerate phenotype manifold.

One invocation runs every matched condition for one seed:

    ellipse_full
    ellipse_ko
    circle_full
    circle_ko
    ellipse_flat_lock
    ellipse_flat_recover

The full ellipse run is forked at CFG.GEOMETRY_FORK_GENERATION.  INTact continues
unchanged.  FLAT-LOCK atomically expands every organism and disables future
composition formation.  FLAT-RECOVER applies the same phenotype-preserving
flattening but immediately restores ordinary GADS structural dynamics.

No probe offspring enter the population.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import common as C
import config as CFG

G = C.G
HERE = Path(__file__).resolve().parent


class GeometryFitness:
    atoms = ("R", "L", "U", "D")

    def __init__(self, a, b, ridge_distance):
        self.a = float(a)
        self.b = float(b)
        self.ridge_distance = float(ridge_distance)
        self._projection_cache = {}

    def endpoint(self, atoms):
        x = y = 0
        for a in atoms:
            if a == "R":
                x += 1
            elif a == "L":
                x -= 1
            elif a == "U":
                y += 1
            elif a == "D":
                y -= 1
            else:
                raise ValueError("unknown geometry atom %r" % (a,))
        return x, y

    def project_xy(self, x, y):
        """Nearest point on the axis-aligned ellipse/circle.

        Phenotypes are integer endpoints, so cache by (x,y).  Newton solves the
        stationary-distance equation in the first quadrant; convexity gives one
        nearest boundary point there, and endpoint candidates guard axis/centre
        cases.  Quadrant symmetry is restored before returning theta.
        """
        key = (int(x), int(y))
        got = self._projection_cache.get(key)
        if got is not None:
            return got

        ax, ay = abs(float(x)), abs(float(y))
        half_pi = math.pi / 2.0
        if ax == 0.0 and ay == 0.0:
            t = half_pi
        else:
            t = math.atan2(self.a * ay, self.b * ax)
            t = min(half_pi, max(0.0, t))
            aa = self.a * self.a
            bb = self.b * self.b
            for _ in range(24):
                s = math.sin(t)
                c = math.cos(t)
                f = (bb - aa) * s * c + self.a * ax * s - self.b * ay * c
                fp = ((bb - aa) * (c * c - s * s)
                      + self.a * ax * c + self.b * ay * s)
                if abs(fp) < 1e-14:
                    break
                nt = min(half_pi, max(0.0, t - f / fp))
                if abs(nt - t) < 1e-13:
                    t = nt
                    break
                t = nt

        # Compare with the quadrant endpoints; this also handles the centre.
        candidates = [0.0, t, half_pi]
        best_t = None
        best_d2 = None
        for q in candidates:
            px = self.a * math.cos(q)
            py = self.b * math.sin(q)
            d2 = (px - ax) ** 2 + (py - ay) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_t = q
        t = best_t

        if x < 0 and y >= 0:
            theta = math.pi - t
        elif x < 0 and y < 0:
            theta = -math.pi + t
        elif x >= 0 and y < 0:
            theta = -t
        else:
            theta = t

        px = self.a * math.cos(theta)
        py = self.b * math.sin(theta)
        distance = math.hypot(float(x) - px, float(y) - py)
        out = (theta, px, py, distance)
        self._projection_cache[key] = out
        return out

    def on_ridge_xy(self, x, y):
        return self.project_xy(x, y)[3] <= self.ridge_distance

    def score(self, atoms):
        x, y = self.endpoint(atoms)
        distance = self.project_xy(x, y)[3]
        # Uniform Euclidean normal stiffness.  Every point on the continuous
        # target curve has exactly the same maximum fitness 0.
        return -(distance * distance)

    def theta_xy(self, x, y):
        return self.project_xy(x, y)[0]

    def curvature_theta(self, theta):
        s = math.sin(theta)
        c = math.cos(theta)
        den = ((self.a * s) ** 2 + (self.b * c) ** 2) ** 1.5
        return (self.a * self.b) / den

    def curvature_xy(self, x, y):
        return self.curvature_theta(self.theta_xy(x, y))

    def tangent_normal_xy(self, x, y):
        t = self.theta_xy(x, y)
        tx = -self.a * math.sin(t)
        ty = self.b * math.cos(t)
        norm = math.hypot(tx, ty) or 1.0
        tx /= norm
        ty /= norm
        nx, ny = -ty, tx
        return (tx, ty), (nx, ny)

def tokens_to_atoms(tokens):
    return [G.ATOMS[t - 2] for t in tokens]


def endpoint_org(org, mg, fit):
    dec, _ = mg.decode(org)
    return fit.endpoint(tokens_to_atoms(dec))


def start_xy_for_fit(fit):
    if fit.a == float(CFG.ELLIPSE_A) and fit.b == float(CFG.ELLIPSE_B):
        return tuple(CFG.ELLIPSE_START)
    if fit.a == float(CFG.CIRCLE_RADIUS) and fit.b == float(CFG.CIRCLE_RADIUS):
        return tuple(CFG.CIRCLE_START)
    raise ValueError("no preregistered start point for geometry %.6g x %.6g" % (fit.a, fit.b))


def wrap_angle(x):
    while x > math.pi:
        x -= 2.0 * math.pi
    while x <= -math.pi:
        x += 2.0 * math.pi
    return x


def lower_curvature_direction(fit):
    """Signed theta direction from the preregistered start toward lower kappa."""
    sx, sy = start_xy_for_fit(fit)
    t0 = fit.theta_xy(sx, sy)
    if abs(fit.a - fit.b) < 1e-15:
        # Circle has no curvature gradient.  Keep the same positive orientation
        # as the matched ellipse start so generic angular bias is subtracted.
        return 1.0
    eps = 1e-6
    dk = fit.curvature_theta(t0 + eps) - fit.curvature_theta(t0 - eps)
    return -1.0 if dk > 0.0 else 1.0


def make_initial_population(fit, seed):
    """Atomic, phenotypically identical, order-diverse optimum population."""
    rng = random.Random("initial:%s:%s:%s" % (seed, fit.a, fit.b))
    x0, y0 = start_xy_for_fit(fit)
    names = []
    names.extend(["R" if x0 >= 0 else "L"] * abs(int(x0)))
    names.extend(["U" if y0 >= 0 else "D"] * abs(int(y0)))

    # Both preregistered starts can be padded to the same odd length with
    # cancelling two-step loops, preserving endpoint and initial exposure.
    target_len = 49
    extra = target_len - len(names)
    if extra < 0 or extra % 2:
        raise AssertionError("geometry start cannot be neutrally padded to target length")
    for _ in range(extra // 2):
        if rng.random() < 0.5:
            names.extend(["R", "L"])
        else:
            names.extend(["U", "D"])

    pop = []
    for _ in range(G.POPULATION_SIZE):
        seq = list(names)
        rng.shuffle(seq)
        pop.append(C.atom_names_to_tokens(seq, fit.atoms))
    for org in pop:
        if endpoint_org(org, G.MetaGenome(), fit) != (x0, y0):
            raise AssertionError("initial geometry population moved off start point")
    if fit.score(tokens_to_atoms(pop[0])) != 0.0:
        raise AssertionError("preregistered geometry start is not exactly optimal")
    return pop


def geometry_population_metrics(pop, mg, evals, fit):
    curv = []
    progress = []
    flat_coordinate = []
    ridge = 0
    scores = []
    sx, sy = start_xy_for_fit(fit)
    start_theta = fit.theta_xy(sx, sy)
    direction = lower_curvature_direction(fit)
    for i, org in enumerate(pop):
        x, y = endpoint_org(org, mg, fit)
        scores.append(evals[i][0])
        if fit.on_ridge_xy(x, y):
            ridge += 1
            t = fit.theta_xy(x, y)
            curv.append(fit.curvature_theta(t))
            progress.append(direction * wrap_angle(t - start_theta))
            flat_coordinate.append(abs(math.sin(t)))
    out = {
        "mean_score": sum(scores) / float(len(scores)),
        "ridge_fraction": ridge / float(len(pop)),
        "mean_curvature_ridge": (
            sum(curv) / float(len(curv)) if curv else None
        ),
        "mean_low_curvature_progress_ridge": (
            sum(progress) / float(len(progress)) if progress else None
        ),
        "mean_flat_coordinate_ridge": (
            sum(flat_coordinate) / float(len(flat_coordinate))
            if flat_coordinate else None
        ),
        "start_theta": start_theta,
        "low_curvature_theta_direction": direction,
    }
    out.update(C.structural_metrics(pop, mg))
    return out


def probe_population(pop, mg, fit, seed, generation, org_limit, trials):
    prng = random.Random("probe:%s:%s:%s:%s" % (seed, fit.a, fit.b, generation))
    ridge_idxs = []
    for i, org in enumerate(pop):
        x, y = endpoint_org(org, mg, fit)
        if fit.on_ridge_xy(x, y):
            ridge_idxs.append(i)
    if not ridge_idxs:
        return {
            "probe_n": 0,
            "canonical_robustness": None,
            "proposal_robustness": None,
            "atomic_robustness": None,
            "canonical_tangent_var": None,
            "canonical_normal_var": None,
        }
    if len(ridge_idxs) > org_limit:
        ridge_idxs = prng.sample(ridge_idxs, org_limit)

    n = can_ok = prop_ok = atom_ok = 0
    tan2 = norm2 = 0.0
    for idx in ridge_idxs:
        org = pop[idx]
        px, py = endpoint_org(org, mg, fit)
        tangent, normal = fit.tangent_normal_xy(px, py)
        for _ in range(trials):
            mo, mm = C.canonical_pass_mutant(org, mg, prng)
            mx, my = endpoint_org(mo, mm, fit)
            can_ok += int(fit.on_ridge_xy(mx, my))
            dx, dy = mx - px, my - py
            tp = dx * tangent[0] + dy * tangent[1]
            np = dx * normal[0] + dy * normal[1]
            tan2 += tp * tp
            norm2 += np * np

            po, pm = C.proposal_mutant(org, mg, prng)
            qx, qy = endpoint_org(po, pm, fit)
            prop_ok += int(fit.on_ridge_xy(qx, qy))

            ad = C.atomic_mutant(org, mg, prng)
            ax, ay = fit.endpoint(tokens_to_atoms(ad))
            atom_ok += int(fit.on_ridge_xy(ax, ay))
            n += 1

    return {
        "probe_n": n,
        "canonical_robustness": can_ok / float(n),
        "proposal_robustness": prop_ok / float(n),
        "atomic_robustness": atom_ok / float(n),
        "canonical_tangent_var": tan2 / float(n),
        "canonical_normal_var": norm2 / float(n),
    }


def observe(pop, mg, fit, seed, generation, org_limit, trials):
    evals, service = G.evaluate_population(pop, mg)
    row = {"generation": generation}
    row.update(geometry_population_metrics(pop, mg, evals, fit))
    row.update(probe_population(
        pop, mg, fit, seed, generation, org_limit, trials
    ))
    return row, evals, service


def run_segment(
    *,
    seed,
    fit,
    structure,
    pop,
    mg,
    rng,
    tie_rng,
    start_generation,
    end_generation,
    checkpoint_every,
    org_limit,
    trials,
    events=None,
):
    params = C.condition_params(CFG.GEOMETRY_GADS_PARAMS, structure)
    C.apply_params(params)
    C.bind_fitness(fit)
    events = events or {
        "captures": 0,
        "opens": 0,
        "baseline_bounds": 0,
        "open_bounds": 0,
    }
    deaths = 0
    trajectory = []
    for generation in range(start_generation, end_generation + 1):
        take = (
            generation == start_generation
            or generation == end_generation
            or generation % checkpoint_every == 0
        )
        if take:
            row, evals, service = observe(
                pop, mg, fit, seed, generation, org_limit, trials
            )
            row["captures_cumulative"] = events["captures"]
            row["deaths_segment"] = deaths
            trajectory.append(row)
        else:
            evals, service = G.evaluate_population(pop, mg)

        if generation == end_generation:
            break
        pop, d = C.advance(pop, mg, evals, service, rng, tie_rng, events)
        deaths += d

    return {
        "pop": pop,
        "mg": mg,
        "rng": rng,
        "tie_rng": tie_rng,
        "events": events,
        "deaths": deaths,
        "trajectory": trajectory,
    }


def fresh_state(seed, fit, structure):
    C.apply_params(C.condition_params(CFG.GEOMETRY_GADS_PARAMS, structure))
    C.bind_fitness(fit)
    pop = make_initial_population(fit, seed)
    return (
        pop,
        G.MetaGenome(),
        random.Random(seed),
        random.Random("tie:%s" % seed),
    )


def run_condition(seed, fit, structure, generations, checkpoint_every, org_limit, trials):
    pop, mg, rng, tie_rng = fresh_state(seed, fit, structure)
    return run_segment(
        seed=seed,
        fit=fit,
        structure=structure,
        pop=pop,
        mg=mg,
        rng=rng,
        tie_rng=tie_rng,
        start_generation=0,
        end_generation=generations,
        checkpoint_every=checkpoint_every,
        org_limit=org_limit,
        trials=trials,
    )


def run_seed(seed, generations, fork_generation, checkpoint_every, org_limit, trials):
    C.require_canonical_identity()
    ellipse = GeometryFitness(CFG.ELLIPSE_A, CFG.ELLIPSE_B, CFG.RIDGE_DISTANCE)
    circle = GeometryFitness(
        CFG.CIRCLE_RADIUS, CFG.CIRCLE_RADIUS, CFG.RIDGE_DISTANCE
    )

    started = time.time()
    conditions = {}

    # Primary controls.
    ko_ellipse = run_condition(
        seed, ellipse, False, generations, checkpoint_every, org_limit, trials
    )
    conditions["ellipse_ko"] = ko_ellipse["trajectory"]

    full_circle = run_condition(
        seed, circle, True, generations, checkpoint_every, org_limit, trials
    )
    conditions["circle_full"] = full_circle["trajectory"]

    ko_circle = run_condition(
        seed, circle, False, generations, checkpoint_every, org_limit, trials
    )
    conditions["circle_ko"] = ko_circle["trajectory"]

    # Full ellipse once to the causal fork.
    pop, mg, rng, tie_rng = fresh_state(seed, ellipse, True)
    pre = run_segment(
        seed=seed,
        fit=ellipse,
        structure=True,
        pop=pop,
        mg=mg,
        rng=rng,
        tie_rng=tie_rng,
        start_generation=0,
        end_generation=fork_generation,
        checkpoint_every=checkpoint_every,
        org_limit=org_limit,
        trials=trials,
    )
    fork_state = C.clone_state(pre["pop"], pre["mg"], pre["rng"], pre["tie_rng"])

    # Intact continuation.
    ipop, img, irng, itie = C.restore_state(fork_state)
    intact = run_segment(
        seed=seed,
        fit=ellipse,
        structure=True,
        pop=ipop,
        mg=img,
        rng=irng,
        tie_rng=itie,
        start_generation=fork_generation,
        end_generation=generations,
        checkpoint_every=checkpoint_every,
        org_limit=org_limit,
        trials=trials,
        events=dict(pre["events"]),
    )
    conditions["ellipse_full"] = (
        pre["trajectory"][:-1] + intact["trajectory"]
    )

    # Exact same fork state, phenotype-preserving atomic flattening.
    for name, recover in (("ellipse_flat_lock", False), ("ellipse_flat_recover", True)):
        fpop, fmg, frng, ftie = C.restore_state(fork_state)
        before = [tuple(fmg.decode(o)[0]) for o in fpop]
        fpop, fmg = C.flatten_population(fpop, fmg)
        after = [tuple(fmg.decode(o)[0]) for o in fpop]
        if before != after:
            raise AssertionError("flatten branch changed phenotype")
        branch = run_segment(
            seed=seed,
            fit=ellipse,
            structure=recover,
            pop=fpop,
            mg=fmg,
            rng=frng,
            tie_rng=ftie,
            start_generation=fork_generation,
            end_generation=generations,
            checkpoint_every=checkpoint_every,
            org_limit=org_limit,
            trials=trials,
        )
        conditions[name] = branch["trajectory"]

    return {
        "experiment": CFG.EXPERIMENT_ID,
        "phase": "geometry",
        "seed": seed,
        "canonical_gads_blob": CFG.CANONICAL_GADS_BLOB,
        "canonical_mux_blob": CFG.CANONICAL_MUX_BLOB,
        "generations": generations,
        "fork_generation": fork_generation,
        "checkpoint_every": checkpoint_every,
        "probe_orgs": org_limit,
        "probes_per_org": trials,
        "ellipse": {
            "a": CFG.ELLIPSE_A,
            "b": CFG.ELLIPSE_B,
            "start": list(CFG.ELLIPSE_START),
            "ridge_distance": CFG.RIDGE_DISTANCE,
        },
        "circle": {
            "radius": CFG.CIRCLE_RADIUS,
            "start": list(CFG.CIRCLE_START),
            "ridge_distance": CFG.RIDGE_DISTANCE,
        },
        "conditions": conditions,
        "elapsed_seconds": time.time() - started,
    }


def build_parser():
    p = argparse.ArgumentParser(description="CRO Phase A geometry experiment")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--generations", type=int, default=CFG.GEOMETRY_GENERATIONS)
    p.add_argument("--fork-generation", type=int, default=CFG.GEOMETRY_FORK_GENERATION)
    p.add_argument("--checkpoint-every", type=int, default=CFG.CHECKPOINT_EVERY)
    p.add_argument("--probe-orgs", type=int, default=CFG.PROBE_ORGS)
    p.add_argument("--probes-per-org", type=int, default=CFG.PROBES_PER_ORG)
    p.add_argument("--out", type=Path)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not (0 < args.fork_generation < args.generations):
        raise ValueError("fork generation must be inside the run")
    result = run_seed(
        args.seed,
        args.generations,
        args.fork_generation,
        args.checkpoint_every,
        args.probe_orgs,
        args.probes_per_org,
    )
    out = args.out
    if out is None:
        out = HERE / CFG.RESULTS_DIR / "geometry" / ("seed_%03d.json" % args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
