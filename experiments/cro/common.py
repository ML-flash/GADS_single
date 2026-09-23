"""Shared apparatus for Curvature-Renormalization Orbit (CRO).

The imported machine is the repository-root GADS.py.  Everything in this file is
observation, experiment-local selection, or state preparation.  No canonical
source is copied or modified.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
import sys
from pathlib import Path

import config as CFG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = ROOT
if str(CANONICAL) not in sys.path:
    sys.path.insert(0, str(CANONICAL))

import GADS as G  # noqa: E402
from mux_fitness import MuxFitness  # noqa: E402


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = ("blob %d\0" % len(data)).encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def require_canonical_identity() -> None:
    got_gads = git_blob_sha(CANONICAL / "GADS.py")
    got_mux = git_blob_sha(CANONICAL / "mux_fitness.py")
    if got_gads != CFG.CANONICAL_GADS_BLOB:
        raise RuntimeError(
            "canonical GADS identity changed: expected %s got %s"
            % (CFG.CANONICAL_GADS_BLOB, got_gads)
        )
    if got_mux != CFG.CANONICAL_MUX_BLOB:
        raise RuntimeError(
            "canonical MUX identity changed: expected %s got %s"
            % (CFG.CANONICAL_MUX_BLOB, got_mux)
        )


def apply_params(params: dict) -> None:
    for name, value in params.items():
        if not hasattr(G, name):
            raise AttributeError("canonical GADS has no parameter %s" % name)
        setattr(G, name, value)
    if G.POPULATION_SIZE < 2:
        raise ValueError("POPULATION_SIZE must be >= 2")
    if G.NUM_PARENTS < 1 or G.NUM_PARENTS > G.POPULATION_SIZE:
        raise ValueError("NUM_PARENTS must be between 1 and POPULATION_SIZE")
    if G.MIN_LEN < 1 or G.MAX_LEN < G.MIN_LEN:
        raise ValueError("require 1 <= MIN_LEN <= MAX_LEN")
    if not (0.0 <= G.ELITE_FRAC <= 1.0):
        raise ValueError("ELITE_FRAC must be in [0,1]")


def condition_params(base: dict, structure: bool) -> dict:
    out = dict(base)
    if not structure:
        out.update(CFG.STRUCTURE_KO)
    return out


def bind_fitness(fit) -> None:
    G.FITNESS = fit
    G.ATOMS = tuple(fit.atoms)
    G.N_BASE = len(G.ATOMS)
    G.BASE_GENES = list(range(2, 2 + G.N_BASE))
    G.FIRST_COMP_ID = 2 + G.N_BASE


def tie_randomized_next_generation(pop, evals, rng, tie_rng):
    """Canonical rank selection with experiment-local random tie breaking.

    Stable sorting would otherwise make population array order an unadvertised
    direction on an exactly degenerate manifold.  Non-tied ranks are unchanged.
    """
    if not G.USE_FITNESS:
        return G.make_next_generation(pop, evals, rng)

    noise = [tie_rng.random() for _ in pop]
    ranked = sorted(
        range(len(pop)),
        key=lambda i: (evals[i][0], noise[i]),
        reverse=True,
    )
    n_elite = int(G.ELITE_FRAC * G.POPULATION_SIZE)
    n_par = min(G.NUM_PARENTS, len(ranked))
    elite = [pop[ranked[i]].copy() for i in range(min(n_elite, len(ranked)))]
    parents = [pop[ranked[i]].copy() for i in range(n_par)]
    next_pop = list(elite)
    while len(next_pop) < G.POPULATION_SIZE:
        next_pop.append(G.breed(parents, rng))
    return next_pop, len(elite)


def advance(pop, mg, evals, service, rng, tie_rng, events):
    """Advance one generation after the caller has observed the scored state."""
    for comp in service:
        if comp in mg.mco:
            mg.age_reset(comp)
    deaths = len(mg.sweep(service))
    pop, n_elite = tie_randomized_next_generation(pop, evals, rng, tie_rng)
    for org in pop[n_elite:]:
        G.mutate_org(org, mg, rng, events)
    return pop, deaths


def payload_indices(org):
    return [
        i for i, tok in enumerate(org)
        if tok not in (G.ENC_OPEN, G.ENC_CLOSE)
    ]


def token_flat_len(tok) -> int:
    if not isinstance(tok, tuple):
        return 1
    n = 0
    stack = [tok]
    while stack:
        cur = stack.pop()
        if isinstance(cur, tuple):
            stack.extend(cur)
        elif cur not in (G.ENC_OPEN, G.ENC_CLOSE):
            n += 1
    return n


def token_depth(tok) -> int:
    if not isinstance(tok, tuple):
        return 0
    best = 0
    stack = [(tok, 1)]
    while stack:
        cur, depth = stack.pop()
        best = max(best, depth)
        for el in cur:
            if isinstance(el, tuple):
                stack.append((el, depth + 1))
    return best


def token_dh(tok) -> float:
    depth = token_depth(tok)
    if depth <= 0:
        return 0.0
    return math.acosh(1.0 + depth) * token_flat_len(tok)


def structural_metrics(pop, mg) -> dict:
    payload = 0
    decoded = 0
    comps = 0
    dh = 0.0
    depths = []
    for org in pop:
        dec, _ = mg.decode(org)
        decoded += len(dec)
        for tok in org:
            if tok in (G.ENC_OPEN, G.ENC_CLOSE):
                continue
            payload += 1
            if isinstance(tok, tuple):
                comps += 1
                d = token_depth(tok)
                depths.append(d)
                dh += token_dh(tok)
    n = max(1, len(pop))
    p = max(1, payload)
    return {
        "mco_size": mg.size(),
        "mean_encoded_payload": payload / float(n),
        "mean_decoded_len": decoded / float(n),
        "compression_ratio": decoded / float(p),
        "composition_fraction": comps / float(p),
        "mean_composition_depth": (
            sum(depths) / float(len(depths)) if depths else 0.0
        ),
        "max_composition_depth": max(depths) if depths else 0,
        "mean_dH_per_payload": dh / float(p),
    }


def flatten_population(pop, mg):
    """Phenotype-preserving atomic expansion of every organism.

    Boundaries and compositions disappear; decoded atomic content is identical.
    The returned MetaGenome is empty.  This is an experiment intervention, not
    a GADS operation.
    """
    out = []
    before = []
    after = []
    for org in pop:
        dec, _ = mg.decode(org)
        before.append(tuple(dec))
        flat = list(dec)
        out.append(flat)
        after.append(tuple(flat))
    if before != after:
        raise AssertionError("flatten intervention changed decoded content")
    return out, G.MetaGenome()


def clone_state(pop, mg, rng, tie_rng):
    return (
        copy.deepcopy(pop),
        copy.deepcopy(mg),
        rng.getstate(),
        tie_rng.getstate(),
    )


def restore_state(state):
    pop, mg, rstate, tstate = state
    rng = random.Random()
    rng.setstate(rstate)
    tie_rng = random.Random()
    tie_rng.setstate(tstate)
    return copy.deepcopy(pop), copy.deepcopy(mg), rng, tie_rng


def choose_org_indices(pop, limit, rng):
    n = min(int(limit), len(pop))
    if n >= len(pop):
        return list(range(len(pop)))
    return rng.sample(range(len(pop)), n)


def canonical_pass_mutant(org, mg, rng):
    o = copy.deepcopy(org)
    m = copy.deepcopy(mg)
    ev = {
        "captures": 0,
        "opens": 0,
        "baseline_bounds": 0,
        "open_bounds": 0,
    }
    G.mutate_org(o, m, rng, ev)
    return o, m


def proposal_mutant(org, mg, rng):
    """One standardized top-level proposal event.

    This intentionally isolates the coordinate system created by GADS: one
    currently exposed payload token is replaced by one draw from the current
    canonical proposal distribution.  It is not claimed to be a full canonical
    mutation pass.
    """
    idxs = payload_indices(org)
    if not idxs:
        return list(org), mg
    o = list(org)
    i = rng.choice(idxs)
    o[i] = G.sample_token(rng, mg)
    return o, mg


def atomic_mutant(org, mg, rng):
    """Representation-blind one-atom substitution control."""
    dec, _ = mg.decode(org)
    if not dec:
        return list(dec)
    out = list(dec)
    i = rng.randrange(len(out))
    old = out[i]
    choices = [g for g in G.BASE_GENES if g != old]
    if choices:
        out[i] = rng.choice(choices)
    return out


def mux_perfect_program(selection_lines=3):
    """Return a postfix exact multiplexer program over canonical atom names."""
    fit = MuxFitness(selection_lines=selection_lines, scoring="original")
    selectors = ["var%d" % i for i in range(selection_lines)]
    data0 = selection_lines
    data = ["var%d" % (data0 + i) for i in range(2 ** selection_lines)]

    def build(sel, dat):
        if not sel:
            return [dat[0]]
        s = sel[0]
        half = len(dat) // 2
        left = build(sel[1:], dat[:half])
        right = build(sel[1:], dat[half:])
        return [s, "NOT"] + left + ["AND", s] + right + ["AND", "OR"]

    program = build(selectors, data)
    if fit.correct(program) != fit.n_rows:
        raise AssertionError("constructed MUX program is not perfect")
    return program


def atom_names_to_tokens(names, atoms):
    index = {name: i + 2 for i, name in enumerate(atoms)}
    return [index[name] for name in names]
