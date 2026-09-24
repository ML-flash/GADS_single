"""CRO Phase B: degenerate-peak MUX6 orbit under official GADS_single GADS.

The run starts directly on an exact 64/64 MUX6 solution. Fitness during this
experiment is *correct rows only*. All perfect programs therefore have exactly
equal maximal scalar fitness, while imperfect programs remain below the peak
and are selected against. This is a degenerate-peak orbit, not globally neutral
drift. Original M-E-GA length/operation terms are absent from the orbit by
construction.

Conditions are matched within seed:

    mux_full
    mux_ko
    mux_flat_lock
    mux_flat_recover

The full run is forked at CFG.MUX_FORK_GENERATION for the two flattening
interventions.  Canonical GADS source and mutation/capture semantics are not
modified.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import common as C
import config as CFG

G = C.G
HERE = Path(__file__).resolve().parent


class CorrectOnlyMux:
    def __init__(self, selection_lines=2):
        self.base = C.MuxFitness(selection_lines=selection_lines, scoring="original")
        self.atoms = self.base.atoms
        self.n_rows = self.base.n_rows

    def correct(self, atoms):
        return self.base.correct(atoms)

    def score(self, atoms):
        # The entire point of this experiment: no parsimony, operation reward,
        # underflow penalty, residue term, robustness reward, or structural term.
        return float(self.correct(atoms))

    def report(self, atoms):
        out = self.base.report(atoms)
        out["score_correct_only"] = float(out["correct"])
        return out


def decoded_atoms(org, mg):
    dec, _ = mg.decode(org)
    return [G.ATOMS[t - 2] for t in dec]


def correct_org(org, mg, fit):
    return fit.correct(decoded_atoms(org, mg))


def make_initial_population(fit, seed):
    """Perfect, atomic, phenotypically equal population with neutral prefix diversity."""
    perfect = C.mux_perfect_program(CFG.MUX_SELECTION_LINES)
    rng = random.Random("mux-initial:%s" % seed)
    var_atoms = [a for a in fit.atoms if a.startswith("var")]
    pop = []
    for _ in range(G.POPULATION_SIZE):
        # Variables placed before a complete postfix program remain below its
        # result on the stack.  They change representation/length, not the
        # returned truth-table function.
        prefix = [rng.choice(var_atoms) for _ in range(rng.randrange(0, 13))]
        names = prefix + perfect
        if fit.correct(names) != fit.n_rows:
            raise AssertionError("neutral prefix changed MUX phenotype")
        pop.append(C.atom_names_to_tokens(names, fit.atoms))
    return pop


def mux_population_metrics(pop, mg, evals, fit):
    correct = [int(e[0]) for e in evals]
    out = {
        "mean_correct": sum(correct) / float(len(correct)),
        "best_correct": max(correct),
        "perfect_fraction": sum(1 for x in correct if x == fit.n_rows) / float(len(correct)),
    }
    out.update(C.structural_metrics(pop, mg))
    return out


def probe_population(pop, mg, fit, seed, generation, org_limit, trials):
    """Read-only matched robustness probes with family-isolated RNG streams.

    Organism selection and every family/organism/trial probe get deterministic
    independent RNGs. A variable number of draws inside canonical mutation can
    therefore never perturb proposal or atomic probes, nor later canonical
    trials. The same seed/generation/index/trial key gives matched perturbations
    across intact and flattened representations.
    """
    select_rng = random.Random("mux-probe-select:%s:%s" % (seed, generation))
    perfect = [i for i, org in enumerate(pop) if correct_org(org, mg, fit) == fit.n_rows]
    if not perfect:
        return {
            "probe_n": 0,
            "canonical_robustness": None,
            "proposal_robustness": None,
            "atomic_robustness": None,
            "canonical_mean_loss": None,
            "proposal_mean_loss": None,
            "atomic_mean_loss": None,
        }
    if len(perfect) > org_limit:
        perfect = select_rng.sample(perfect, org_limit)

    n = can_ok = prop_ok = atom_ok = 0
    can_loss = prop_loss = atom_loss = 0.0
    for idx in perfect:
        org = pop[idx]
        for trial in range(trials):
            can_rng = random.Random(
                "mux-probe-canonical:%s:%s:%s:%s"
                % (seed, generation, idx, trial)
            )
            prop_rng = random.Random(
                "mux-probe-proposal:%s:%s:%s:%s"
                % (seed, generation, idx, trial)
            )
            atom_rng = random.Random(
                "mux-probe-atomic:%s:%s:%s:%s"
                % (seed, generation, idx, trial)
            )

            mo, mm = C.canonical_pass_mutant(org, mg, can_rng)
            mc = correct_org(mo, mm, fit)
            can_ok += int(mc == fit.n_rows)
            can_loss += fit.n_rows - mc

            po, pm = C.proposal_mutant(org, mg, prop_rng)
            pc = correct_org(po, pm, fit)
            prop_ok += int(pc == fit.n_rows)
            prop_loss += fit.n_rows - pc

            ad = C.atomic_mutant(org, mg, atom_rng)
            ac = fit.correct([G.ATOMS[t - 2] for t in ad])
            atom_ok += int(ac == fit.n_rows)
            atom_loss += fit.n_rows - ac
            n += 1

    return {
        "probe_n": n,
        "canonical_robustness": can_ok / float(n),
        "proposal_robustness": prop_ok / float(n),
        "atomic_robustness": atom_ok / float(n),
        "canonical_mean_loss": can_loss / float(n),
        "proposal_mean_loss": prop_loss / float(n),
        "atomic_mean_loss": atom_loss / float(n),
    }

def observe(pop, mg, fit, seed, generation, org_limit, trials):
    evals, service = G.evaluate_population(pop, mg)
    row = {"generation": generation}
    row.update(mux_population_metrics(pop, mg, evals, fit))
    row.update(probe_population(pop, mg, fit, seed, generation, org_limit, trials))
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
    C.apply_params(C.condition_params(CFG.MUX_GADS_PARAMS, structure))
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
    C.apply_params(C.condition_params(CFG.MUX_GADS_PARAMS, structure))
    C.bind_fitness(fit)
    pop = make_initial_population(fit, seed)
    return (
        pop,
        G.MetaGenome(),
        random.Random(seed),
        random.Random("mux-tie:%s" % seed),
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
    fit = CorrectOnlyMux(CFG.MUX_SELECTION_LINES)
    perfect = C.mux_perfect_program(CFG.MUX_SELECTION_LINES)
    if fit.correct(perfect) != fit.n_rows:
        raise AssertionError("CRO MUX seed program is not exact")

    started = time.time()
    conditions = {}

    ko = run_condition(
        seed, fit, False, generations, checkpoint_every, org_limit, trials
    )
    conditions["mux_ko"] = ko["trajectory"]

    # Full GADS once to the causal fork.
    pop, mg, rng, tie_rng = fresh_state(seed, fit, True)
    pre = run_segment(
        seed=seed,
        fit=fit,
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

    # Direct causal measurement at the fork. Probe the exact same population
    # before and immediately after phenotype-preserving flattening, using the
    # same per-family/per-trial deterministic probe keys.
    bpop, bmg, _, _ = C.restore_state(fork_state)
    fork_before = probe_population(
        bpop, bmg, fit, seed, fork_generation, org_limit, trials
    )
    apop, amg = C.flatten_population(bpop, bmg)
    fork_after = probe_population(
        apop, amg, fit, seed, fork_generation, org_limit, trials
    )
    if fork_before["probe_n"] != fork_after["probe_n"]:
        raise AssertionError("fork probe sample changed under flattening")
    if fork_before["atomic_robustness"] != fork_after["atomic_robustness"]:
        raise AssertionError("flattening changed matched atomic robustness at fork")
    if fork_before["atomic_mean_loss"] != fork_after["atomic_mean_loss"]:
        raise AssertionError("flattening changed matched atomic loss at fork")
    fork_probe = {
        "before": fork_before,
        "after_flatten": fork_after,
        "delta_before_minus_flatten": {
            key: (
                fork_before[key] - fork_after[key]
                if fork_before[key] is not None and fork_after[key] is not None
                else None
            )
            for key in (
                "canonical_robustness",
                "proposal_robustness",
                "atomic_robustness",
                "canonical_mean_loss",
                "proposal_mean_loss",
                "atomic_mean_loss",
            )
        },
    }

    ipop, img, irng, itie = C.restore_state(fork_state)
    intact = run_segment(
        seed=seed,
        fit=fit,
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
    conditions["mux_full"] = pre["trajectory"][:-1] + intact["trajectory"]

    for name, recover in (("mux_flat_lock", False), ("mux_flat_recover", True)):
        fpop, fmg, frng, ftie = C.restore_state(fork_state)
        before = [tuple(fmg.decode(o)[0]) for o in fpop]
        fpop, fmg = C.flatten_population(fpop, fmg)
        after = [tuple(fmg.decode(o)[0]) for o in fpop]
        if before != after:
            raise AssertionError("MUX flatten branch changed decoded programs")
        # Strong phenotype gate: every organism's truth-table behavior must be
        # exactly unchanged by the intervention.
        for b, a in zip(before, after):
            bat = [fit.atoms[t - 2] for t in b]
            aat = [fit.atoms[t - 2] for t in a]
            if fit.correct(bat) != fit.correct(aat):
                raise AssertionError("flatten branch changed MUX phenotype")

        branch = run_segment(
            seed=seed,
            fit=fit,
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
        "phase": "mux",
        "seed": seed,
        "canonical_gads_blob": CFG.CANONICAL_GADS_BLOB,
        "canonical_mux_blob": CFG.CANONICAL_MUX_BLOB,
        "selection_lines": CFG.MUX_SELECTION_LINES,
        "truth_rows": fit.n_rows,
        "fitness": "correct_rows_only",
        "seed_program_atoms": perfect,
        "seed_program_len": len(perfect),
        "generations": generations,
        "fork_generation": fork_generation,
        "checkpoint_every": checkpoint_every,
        "probe_orgs": org_limit,
        "probes_per_org": trials,
        "conditions": conditions,
        "fork_probe": fork_probe,
        "probe_rng_scheme": "family_and_trial_isolated_v1",
        "elapsed_seconds": time.time() - started,
    }


def build_parser():
    p = argparse.ArgumentParser(description="CRO Phase B degenerate-peak MUX6 orbit")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--generations", type=int, default=CFG.MUX_GENERATIONS)
    p.add_argument("--fork-generation", type=int, default=CFG.MUX_FORK_GENERATION)
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
        out = HERE / CFG.RESULTS_DIR / "mux" / ("seed_%03d.json" % args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
