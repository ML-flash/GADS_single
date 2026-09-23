# Curvature-Renormalization Orbit (CRO) — Experiment v1

## Question

Can stochastic evolutionary dynamics acquire a direction when the external
objective supplies no direction along a degenerate optimum, and can GADS
renormalization itself change the local mutational geometry of phenotypically
equivalent solutions?

This experiment is motivated by:

Razi Fachareldeen and Naama Brenner, **"Evolution on degenerate fitness
landscapes is not random: Curvature drives directional drift,"** PNAS 123(32),
e2605142123 (2026), DOI 10.1073/pnas.2605142123.

The published result reports directional mutation-selection drift along smooth
equal-fitness manifolds toward regions of lower curvature / greater robustness,
despite a vanishing fitness gradient along the manifold.

CRO is an independent GADS realization of that broad experimental question. It
does **not** claim to reproduce the paper's exact stochastic equations. The
paper's qualitative causal structure is the target:

1. an optimal degenerate manifold;
2. stochastic mutation-selection dynamics;
3. no explicit reward for position along that manifold or for robustness;
4. measurement of whether population motion nevertheless becomes directional.

Phase B then asks a stronger GADS-specific question: whether an endogenous
change of representation changes the mutational neighborhood while decoded
phenotype and external fitness are held fixed.

## Source binding

The machine is the official GADS_single implementation:

    GADS.py
    git blob 829e9bd74f8f403207cc41a730a6e429501fe75e

The evaluator used by Phase B is:

    mux_fitness.py
    git blob a7bf2332ac4b6a512c11867436dbd26caac6568e

common.require_canonical_identity() recomputes Git blob SHA-1 values from the
files and fails closed on any mismatch.

The experiment imports the repository-root machine and does not edit it. Selection tie
randomization, phenotype construction, flattening and probes are all
experiment-local apparatus.

## Statistical unit

One seed is one independently evolving population and one observation.

Generations, organisms and probe mutations are repeated measurements inside that
observation. They are never treated as independent replicates.

The v1 panel is 16 matched seeds, 1..16. Conditions with the same seed are
paired.

## Phase A — controlled curvature replication

### Phenotype

The atom alphabet is:

    R L U D

A decoded GADS organism is interpreted as a 2-D walk. Its endpoint is the
phenotype p=(x,y). Capture, nesting, opening and inlining can change
representation without changing the decoded walk and therefore without changing
the phenotype.

### Landscapes

Ellipse:

    (x/a)^2 + (y/b)^2 = 1
    a = 30
    b = 15

Circle control:

    x^2 + y^2 = r^2
    r = 25

Fitness is the negative squared nearest-point Euclidean distance from the target curve. Every point
on the continuous target curve has the same optimum. Position along the curve is
not an objective term. Curvature on the ellipse is measured analytically:

    k(theta) =
      a b / (a^2 sin^2(theta) + b^2 cos^2(theta))^(3/2)

The circle has constant curvature and is the geometric negative control.

Every population begins at an exact optimum away from a curvature extremum:

    ellipse start = (24, 9)
    circle start  = (20, 15)

Both have the same normalized starting angle (cos=0.8, sin=0.6). The ellipse
there has a nonzero curvature gradient; the circle does not. Organisms are
atomic but sequence-order diverse and decode to the same endpoint within each
condition.

### Primary arms

| arm | landscape | GADS structure |
| --- | --- | --- |
| ellipse_full | ellipse | enabled |
| ellipse_ko | ellipse | disabled |
| circle_full | circle | enabled |
| circle_ko | circle | disabled |

The knockout disables only composition formation/reuse:

    BOUNDARY_INSERT_PROB = 0
    CAPTURE_PROB = 0
    OPEN_PROB = 0
    BASE_GENE_PROB = 1

Base-atom point/swap/insert/delete dynamics remain.

The first paper-analogue test is deliberately the **KO ellipse minus KO
circle** signed-drift contrast. If the non-renormalizing substrate does not move
toward lower curvature in excess of its matched constant-curvature control,
GADS-specific interpretation stops there.

### Geometry observables

At each checkpoint:

- fraction of the population within a fixed 1-step Euclidean reporting distance of the optimum;
- mean local curvature of those ridge-near phenotypes;
- signed angular displacement from the preregistered start in the local
  lower-curvature direction;
- abs(sin(theta)) as a secondary position coordinate;
- mean external score;
- canonical-pass mutational robustness;
- one-proposal robustness;
- representation-blind one-decoded-atom robustness;
- tangent and normal variance of canonical mutation displacements.

No robustness quantity enters fitness.

## Renormalization observables

The same checkpoints record:

- Meta-Genome size;
- mean top-level encoded payload length;
- mean decoded atomic length;
- decoded / encoded compression ratio;
- top-level composition fraction;
- mean and maximum composition depth;
- an Ackley-style structural quantity

      dH(m) = acosh(1 + depth(m)) * flat_length(m)

  averaged over exposed payload tokens;
- capture and removal telemetry.

These are observations, not objective terms.

## Causal flattening fork

At generation 1500, ellipse_full is cloned into three exact branches.

**INTACT**

Continues with ordinary GADS.

**FLAT-LOCK**

Every organism is recursively expanded to its decoded base atoms and the
Meta-Genome is cleared. Composition formation is then disabled.

**FLAT-RECOVER**

Receives the exact same flattening intervention, but ordinary GADS structure
formation is immediately restored.

The intervention asserts for every organism:

    Decode(before) == Decode(after)

so endpoint phenotype and external fitness are unchanged at the fork.

The intended causal signatures are therefore:

    INTACT robustness > FLAT-LOCK robustness

and, if hierarchy is rebuilt,

    FLAT-RECOVER robustness > FLAT-LOCK robustness.

A representation-blind decoded-atom probe is reported beside the
representation-sensitive probes. If only the latter move, that distinguishes a
change in GADS coordinates from a change in the underlying phenotype's ordinary
atom-level fragility.

## Phase B — GADS_single/MUX11 neutral orbit

Phase B uses canonical River MUX11 but removes the original scoring confound
from the orbit.

The runner constructs an exact 50-token postfix MUX11 program and verifies it is
2048/2048 before the run. It initializes an atomic population with neutral
prefix diversity around this exact solution.

Orbit fitness is then exactly:

    fitness(g) = correct_rows(g)

There is:

- no length reward or penalty;
- no successful-operation reward;
- no residue term;
- no robustness term;
- no structural term.

Thus every 2048/2048 program has identical maximal scalar fitness.

The full-GADS arm uses the frozen Experiment-1 GADS rate block. The KO arm uses
the same block with composition formation/reuse disabled.

The full arm is again forked at generation 1500 into INTACT, FLAT-LOCK and
FLAT-RECOVER branches under an exact decoded-program preservation gate.

### MUX robustness

For perfect organisms the experiment estimates:

    R_canonical =
      P(one canonical mutation pass remains 2048/2048)

    R_proposal =
      P(one exposed-token proposal remains 2048/2048)

    R_atomic =
      P(one decoded-base-atom substitution remains 2048/2048)

and the corresponding mean number of truth-table rows lost.

R_proposal explicitly measures the coordinate system exposed by the current
GADS representation. R_atomic ignores that representation and acts as the
control.

## Tie handling

An exactly degenerate objective exposes a normally invisible implementation
detail: Python's stable sort would make population array order a deterministic
tie-breaker.

CRO therefore restates canonical rank selection in the experiment harness and
adds an independent random key inside equal-fitness classes. Non-tied fitness
ordering, canonical G.breed, canonical mutation and canonical Meta-Genome
mechanics are unchanged.

This is apparatus required to avoid injecting an artificial direction into the
neutral manifold; it is not a GADS modification.

## Prespecified reductions

For any trajectory metric X:

    drift(X) =
      mean(last 20% of checkpoints) - mean(first 20% of checkpoints)

One seed contributes one drift value or one matched endpoint contrast.

analyze.py reports paired sign-flip tests and leave-one-seed-out sign stability.
With the frozen 16-seed panel the sign-flip enumeration is exact
(2^16 assignments).

The primary paper-analogue contrast is:

    G1: drift(signed low-curvature progress, ellipse_ko)
        - drift(the same signed angular coordinate, circle_ko) > 0

The main GADS extension contrasts are:

    G3: robustness drift(ellipse_full)
        - robustness drift(ellipse_ko) > 0

    G4: endpoint robustness(ellipse_full)
        - endpoint robustness(ellipse_flat_lock) > 0

and on MUX:

    M1: drift(R_proposal, mux_full) > 0

    M3: endpoint R_proposal(mux_full)
        - endpoint R_proposal(mux_flat_lock) > 0

Other reported quantities are mechanistic/supporting measurements. A positive
result on one contrast does not promote the others automatically.

## Falsification / interpretation rules

- If G1 is not supported, CRO has not reproduced the curvature-drift signature
  in the controlled substrate. Do not use GADS-specific arms to claim a
  replication of the paper.
- If structure/compression changes but robustness does not, renormalization
  occurred but the proposed connection to flattening/robustness is unsupported.
- If INTACT and FLAT-LOCK have the same representation-sensitive robustness,
  the causal representation claim is unsupported.
- If FLAT-RECOVER does not rebuild structure or robustness, recovery is not
  established even if INTACT differs from FLAT-LOCK.
- If R_atomic changes together with representation-sensitive robustness, the
  difference may be attributable to movement in phenotype space rather than to
  representation alone and must be interpreted accordingly.
- MUX correctness is the only Phase-B fitness. Any accidental use of original or
  rebalanced MUX scalar scoring invalidates the neutral-orbit result.

## Run

From repository root:

    python experiments/cro/validate.py --smoke

One geometry seed:

    python experiments/cro/run_geometry.py --seed 1

One MUX seed:

    python experiments/cro/run_mux_orbit.py --seed 1

Full matched panel, using up to 16 local workers:

    python experiments/cro/run_panel.py --phase both --seeds 1-16 --jobs 16

The probe budget is intentionally configurable because the read-only mutation
assays can dominate runtime:

    python experiments/cro/run_panel.py --phase both --seeds 1-16 --jobs 16
      --probe-orgs 12 --probes-per-org 4

Analyze whatever completed products exist:

    python experiments/cro/analyze.py

Products are written beneath:

    experiments/cro/results/geometry/
    experiments/cro/results/mux/
    experiments/cro/results/analysis.json

results/ should remain uncommitted outcome data.

## Order of execution

Do not start by reading the GADS extension as evidence for the biological
analogue. The intended evidential order is:

    canonical identity / preflight
        ->
    geometry KO curvature result
        ->
    full-GADS geometry comparison
        ->
    flatten/recover causal fork
        ->
    MUX11 correctness-only orbit
        ->
    MUX flatten/recover fork

This keeps the biological replication question separate from the stronger GADS
renormalization hypothesis.
