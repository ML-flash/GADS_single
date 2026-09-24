"""CRO apparatus preflight.

This is intentionally cheap.  It verifies source identity, exact MUX6 seed
phenotype, geometry degeneracy, phenotype-preserving flattening, and imports all
runners before any long panel is launched.
"""
from __future__ import annotations

import argparse
import random

import common as C
import config as CFG
import run_geometry as RG
import run_mux_orbit as RM

G = C.G


def gate(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print("%-5s %-48s %s" % (status, name, detail))
    if not ok:
        raise AssertionError(name + (": " + detail if detail else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate CRO apparatus")
    p.add_argument("--smoke", action="store_true",
                   help="also run tiny geometry and MUX end-to-end smokes")
    args = p.parse_args(argv)

    C.require_canonical_identity()
    gate("canonical GADS blob", True, CFG.CANONICAL_GADS_BLOB)
    gate("canonical MUX blob", True, CFG.CANONICAL_MUX_BLOB)

    ef = RG.GeometryFitness(CFG.ELLIPSE_A, CFG.ELLIPSE_B, CFG.RIDGE_DISTANCE)
    gate("ellipse high-curvature endpoint is optimal", abs(ef.score(["R"] * int(CFG.ELLIPSE_A))) < 1e-20)
    gate("ellipse low-curvature endpoint is optimal", abs(ef.score(["U"] * int(CFG.ELLIPSE_B))) < 1e-20)
    kh = ef.curvature_xy(int(CFG.ELLIPSE_A), 0)
    kl = ef.curvature_xy(0, int(CFG.ELLIPSE_B))
    gate("ellipse has ordered curvature", kh > kl, "high=%.6g low=%.6g" % (kh, kl))
    sx, sy = CFG.ELLIPSE_START
    gate("ellipse preregistered start is exactly optimal",
         abs(ef.score(["R"] * sx + ["U"] * sy)) < 1e-20,
         "start=%r" % (CFG.ELLIPSE_START,))
    ks = ef.curvature_xy(sx, sy)
    gate("ellipse start is not a curvature extremum", kl < ks < kh,
         "start k=%.6g" % ks)
    gate("lower-curvature signed direction is defined",
         RG.lower_curvature_direction(ef) > 0.0)

    cf = RG.GeometryFitness(CFG.CIRCLE_RADIUS, CFG.CIRCLE_RADIUS, CFG.RIDGE_DISTANCE)
    kc0 = cf.curvature_xy(int(CFG.CIRCLE_RADIUS), 0)
    kc1 = cf.curvature_xy(0, int(CFG.CIRCLE_RADIUS))
    gate("circle curvature is constant", abs(kc0 - kc1) < 1e-15)

    mf = RM.CorrectOnlyMux(CFG.MUX_SELECTION_LINES)
    perfect = C.mux_perfect_program(CFG.MUX_SELECTION_LINES)
    gate("constructed MUX6 program is 64/64",
         mf.correct(perfect) == mf.n_rows,
         "len=%d" % len(perfect))
    gate("MUX orbit score is correctness only",
         mf.score(perfect) == float(mf.n_rows))

    # Explicit nested structure and flattening invariant.
    C.apply_params(CFG.GEOMETRY_GADS_PARAMS)
    C.bind_fitness(ef)
    r = C.atom_names_to_tokens(["R"], ef.atoms)[0]
    u = C.atom_names_to_tokens(["U"], ef.atoms)[0]
    child = (r, u)
    parent = (child, r)
    mg = G.MetaGenome()
    mg.age_enter(child)
    mg.age_enter(parent)
    pop = [[parent, r], [child, u, r]]
    before = [tuple(mg.decode(o)[0]) for o in pop]
    flat, empty = C.flatten_population(pop, mg)
    after = [tuple(empty.decode(o)[0]) for o in flat]
    gate("flatten preserves decoded atomic content", before == after)
    gate("flatten clears MetaGenome", empty.size() == 0)
    gate("flatten removes top-level compositions",
         all(not isinstance(t, tuple) for o in flat for t in o))

    # Tie randomization must not alter non-tied rank semantics and must create a
    # full population.  This is apparatus, not a result.
    G.POPULATION_SIZE = 4
    G.NUM_PARENTS = 2
    G.ELITE_FRAC = 0.0
    G.ENABLE_CROSSOVER = False
    toy = [[r], [u], [r, r], [u, u]]
    evals = [(1.0, 1, set()), (1.0, 1, set()), (0.0, 2, set()), (0.0, 2, set())]
    nxt, elite = C.tie_randomized_next_generation(
        toy, evals, random.Random(1), random.Random(2)
    )
    gate("tie-randomized selector returns full population", len(nxt) == 4 and elite == 0)

    if args.smoke:
        print("\nTiny end-to-end smoke...")
        g = RG.run_seed(
            1, generations=4, fork_generation=2,
            checkpoint_every=1, org_limit=2, trials=1,
        )
        gate("geometry smoke produced all conditions",
             set(g["conditions"]) == {
                 "ellipse_ko", "circle_full", "circle_ko", "ellipse_full",
                 "ellipse_flat_lock", "ellipse_flat_recover",
             })
        m = RM.run_seed(
            1, generations=4, fork_generation=2,
            checkpoint_every=1, org_limit=2, trials=1,
        )
        gate("MUX smoke produced all conditions",
             set(m["conditions"]) == {
                 "mux_ko", "mux_full", "mux_flat_lock", "mux_flat_recover",
             })
        gate("MUX smoke begins entirely perfect",
             m["conditions"]["mux_full"][0]["perfect_fraction"] == 1.0)

    print("\nCRO preflight passed.")


if __name__ == "__main__":
    main()
