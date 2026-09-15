"""One-command standalone runner for canonical solo GADS on MUX.

Running ``python run.py`` loads ``config.json`` from this directory and executes
one GADS instance. The machine in GADS.py is not modified by this runner; all
experiment choices are applied as module parameters before the run starts.

Normal output stores only scalar run summaries and the best decoded program.
No population history is retained.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import GADS as G
from mux_fitness import MuxFitness

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"

# Parameters intentionally exposed by the standalone runner. The names are the
# actual GADS constants so a saved configuration says exactly what was changed.
GADS_PARAMS = {
    "POPULATION_SIZE",
    "MIN_LEN",
    "MAX_LEN",
    "ENABLE_CROSSOVER",
    "NUM_PARENTS",
    "CROSSOVER_PROB",
    "BOUNDARY_INSERT_PROB",
    "BOUNDARY_REMOVE_PROB",
    "CAPTURE_PROB",
    "MIN_CAPTURE_LEN",
    "MUTATION_PROB",
    "BOUNDARY_MUTATION_PROB",
    "OPEN_PROB",
    "BASE_GENE_PROB",
    "MCO_DECAY",
    "DB_THRESHOLD",
    "ELITE_FRAC",
    "USE_FITNESS",
}

EVENT_KEYS = ("captures", "opens", "baseline_bounds", "open_bounds")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_value(text: str):
    """Parse a --set value using JSON syntax, falling back to a string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(cfg: dict, args) -> dict:
    cfg = copy.deepcopy(cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.generations is not None:
        cfg["generations"] = args.generations
    if args.selection_lines is not None:
        cfg["selection_lines"] = args.selection_lines
    if args.scoring is not None:
        cfg["scoring"] = args.scoring
    if args.report_every is not None:
        cfg["report_every"] = args.report_every
    if args.no_save:
        cfg["save_result"] = None
    for assignment in args.set_values:
        if "=" not in assignment:
            raise ValueError("--set must be NAME=VALUE, got %r" % assignment)
        name, raw = assignment.split("=", 1)
        cfg.setdefault("gads", {})[name] = parse_value(raw)
    return cfg


def apply_gads_params(params: dict) -> None:
    unknown = sorted(set(params) - GADS_PARAMS)
    if unknown:
        raise ValueError("Unknown or non-configurable GADS parameter(s): %s" %
                         ", ".join(unknown))
    for name, value in params.items():
        if not hasattr(G, name):
            raise AttributeError("GADS.py has no parameter %s" % name)
        setattr(G, name, value)

    if G.POPULATION_SIZE < 2:
        raise ValueError("POPULATION_SIZE must be >= 2")
    if G.NUM_PARENTS < 1 or G.NUM_PARENTS > G.POPULATION_SIZE:
        raise ValueError("NUM_PARENTS must be between 1 and POPULATION_SIZE")
    if G.MIN_LEN < 1 or G.MAX_LEN < G.MIN_LEN:
        raise ValueError("Require 1 <= MIN_LEN <= MAX_LEN")
    if not (0.0 <= G.ELITE_FRAC <= 1.0):
        raise ValueError("ELITE_FRAC must be in [0, 1]")


def bind_mux(selection_lines: int, scoring: str) -> MuxFitness:
    fit = MuxFitness(selection_lines=selection_lines, scoring=scoring)
    G.FITNESS = fit
    G.ATOMS = fit.atoms
    G.N_BASE = len(fit.atoms)
    G.BASE_GENES = list(range(2, 2 + G.N_BASE))
    G.FIRST_COMP_ID = 2 + G.N_BASE
    return fit


def best_record(pop, evals, mg, fit, generation: int) -> dict:
    idx = max(range(len(pop)), key=lambda i: evals[i][0])
    decoded, _ = mg.decode(pop[idx])
    atoms = [G.ATOMS[t - 2] for t in decoded]
    report = fit.report(atoms)
    return {
        "generation": generation,
        "score": evals[idx][0],
        "encoded_len": len(pop[idx]),
        "decoded_len": len(decoded),
        "program": atoms,
        **report,
    }


def find_perfect(pop, evals, mg, fit, generation: int):
    """Return the highest-scoring perfect organism, if one exists."""
    best = None
    for i, org in enumerate(pop):
        decoded, _ = mg.decode(org)
        atoms = [G.ATOMS[t - 2] for t in decoded]
        if fit.correct(atoms) != fit.n_rows:
            continue
        rec = {
            "generation": generation,
            "score": evals[i][0],
            "encoded_len": len(org),
            "decoded_len": len(decoded),
            "program": atoms,
            **fit.report(atoms),
        }
        if best is None or rec["score"] > best["score"]:
            best = rec
    return best


def print_header(fit, cfg):
    print("GADS_single — standalone GADS / MUX")
    print("seed=%s  MUX=%s  rows=%s  scoring=%s  pop=%s  generations=%s" % (
        cfg["seed"],
        fit.n_var,
        fit.n_rows,
        cfg["scoring"],
        G.POPULATION_SIZE,
        cfg["generations"],
    ))
    print("participation = current census service set; no Participation Register")
    print()
    print(" gen | best score | correct | enc | dec | MCO | used | captures | deaths")
    print("-----+------------+---------+-----+-----+-----+------+----------+-------")


def print_row(generation, rec, mg, service, events, deaths):
    print("%4d | %10.2f | %3d/%-3d | %3d | %3d | %3d | %4d | %8d | %6d" % (
        generation,
        rec["score"],
        rec["correct"],
        rec["rows"],
        rec["encoded_len"],
        rec["decoded_len"],
        mg.size(),
        len(service),
        events["captures"],
        deaths,
    ))


def run_experiment(cfg: dict) -> dict:
    apply_gads_params(cfg.get("gads", {}))
    fit = bind_mux(int(cfg.get("selection_lines", 2)),
                   str(cfg.get("scoring", "rebalanced")))

    seed = int(cfg.get("seed", 42))
    generations = int(cfg.get("generations", 300))
    report_every = int(cfg.get("report_every", 10))
    stop_on_perfect = bool(cfg.get("stop_on_perfect", True))
    if generations < 0:
        raise ValueError("generations must be >= 0")
    if report_every < 1:
        raise ValueError("report_every must be >= 1")

    rng = random.Random(seed)
    mg = G.MetaGenome()
    pop = G.init_population(rng)
    events = {k: 0 for k in EVENT_KEYS}
    deaths = 0
    trajectory = []
    best_ever = None
    started = time.time()

    print_header(fit, cfg)

    # Generation g is always evaluated before its participation sweep. Captures
    # created while producing g+1 therefore first face a census at g+1.
    for generation in range(generations + 1):
        evals, service = G.evaluate_population(pop, mg)
        rec = best_record(pop, evals, mg, fit, generation)
        if best_ever is None or rec["score"] > best_ever["score"]:
            best_ever = copy.deepcopy(rec)

        should_report = (
            generation == 0 or
            generation == generations or
            generation % report_every == 0
        )
        if should_report:
            print_row(generation, rec, mg, service, events, deaths)
            trajectory.append({
                "generation": generation,
                "best_score": rec["score"],
                "correct": rec["correct"],
                "rows": rec["rows"],
                "mco_size": mg.size(),
                "service_size": len(service),
                "captures": events["captures"],
                "deaths": deaths,
            })

        if stop_on_perfect:
            perfect = find_perfect(pop, evals, mg, fit, generation)
            if perfect is not None:
                if best_ever is None or perfect["score"] > best_ever["score"]:
                    best_ever = copy.deepcopy(perfect)
                if not should_report:
                    print_row(generation, perfect, mg, service, events, deaths)
                    trajectory.append({
                        "generation": generation,
                        "best_score": perfect["score"],
                        "correct": perfect["correct"],
                        "rows": perfect["rows"],
                        "mco_size": mg.size(),
                        "service_size": len(service),
                        "captures": events["captures"],
                        "deaths": deaths,
                    })
                print("\nPerfect truth-table solution found at generation %d." % generation)
                break

        if generation == generations:
            break

        # Participation is still a generation-local service signal, not a
        # persistent register. Keep the rollback lifecycle identical everywhere:
        # a used composition clears any temporary arrival grace before sweep.
        # In a solo run this is normally inert, but the machine/lifecycle remains
        # the same as the networked case.
        for comp in service:
            if comp in mg.mco:
                mg.age_reset(comp)
        deaths += len(mg.sweep(service))

        pop, n_elite = G.make_next_generation(pop, evals, rng)
        for org in pop[n_elite:]:
            G.mutate_org(org, mg, rng, events)

    elapsed = time.time() - started
    final_evals, final_service = G.evaluate_population(pop, mg)
    final = best_record(pop, final_evals, mg, fit, generation)

    result = {
        "config": cfg,
        "elapsed_seconds": elapsed,
        "completed_generation": generation,
        "events": {**events, "deaths": deaths},
        "final_mco_size": mg.size(),
        "final_service_size": len(final_service),
        "best_ever": best_ever,
        "final_best": final,
        "trajectory": trajectory,
    }

    print("\nBest-ever selected organism")
    print("  generation : %d" % best_ever["generation"])
    print("  score      : %.2f" % best_ever["score"])
    print("  correct    : %d/%d" % (best_ever["correct"], best_ever["rows"]))
    print("  decoded len: %d" % best_ever["decoded_len"])
    print("  program    : %s" % " ".join(best_ever["program"]))
    print("  MCO size   : %d" % mg.size())
    print("  captures   : %d" % events["captures"])
    print("  deaths     : %d" % deaths)
    print("  elapsed    : %.3fs" % elapsed)

    save_result = cfg.get("save_result")
    if save_result:
        out = Path(save_result)
        if not out.is_absolute():
            out = HERE / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        print("  saved      : %s" % out)

    return result


def build_parser():
    p = argparse.ArgumentParser(description="Run one standalone GADS instance on MUX")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="JSON config file (default: config.json)")
    p.add_argument("--seed", type=int)
    p.add_argument("--generations", type=int)
    p.add_argument("--selection-lines", type=int, choices=(2, 3))
    p.add_argument("--scoring", choices=("original", "rebalanced"))
    p.add_argument("--report-every", type=int)
    p.add_argument("--set", dest="set_values", action="append", default=[],
                   metavar="NAME=VALUE",
                   help="override one GADS constant; may be repeated")
    p.add_argument("--no-save", action="store_true",
                   help="do not write the JSON result")
    return p


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
