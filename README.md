# GADS_single

Reference implementation of a single **Generative Adaptive Dynamical System
(GADS)**.

A GADS is a search process that authors part of its own representational
language while it searches. It can create new units from structures found in
the population, reuse those units as components, combine them into deeper
structures, and later open or dissolve them without changing the atomic content
they represent.

That ability changes the space through which future search proceeds. It also
creates failure modes that do not exist when the representation is fixed.
`GADS.py` implements the mechanisms that create, expose, reuse, and remove
representational structure while keeping those processes inside the system's
ordinary operation. There is no central controller directing what the
representation should become.

This repository contains one GADS, one external problem environment, and a
runner that joins them. It is intentionally small enough to read as an
implementation rather than as a framework.

## Start here

Python 3.11 or newer is recommended. No third-party packages are required.

```bash
git clone https://github.com/ML-flash/GADS_single.git
cd GADS_single
python run.py
```

The default configuration runs one population against a six-variable
multiplexer problem and writes a summary to `results/last_run.json`.

For a short smoke run:

```bash
python run.py --generations 25 --report-every 25 --no-save
```

For all available overrides:

```bash
python run.py --help
```

## What the implementation contains

| Component | Role |
| --- | --- |
| Population | Holds the encoded constructions subjected to variation and differential survival. |
| Boundary mechanism | Creates mobile, paired regions in an encoded construction. Boundaries remain matched and cannot nest. |
| Capture | Replaces a bounded span with a composition carrying that span as its own content. |
| Meta-Genome | Holds the compositions currently available for reuse by the population. |
| MCO sampling | Biases composition sampling by capture recency without setting a target Meta-Genome size. |
| Open | Expands a composition back into its contents, exposing the structure to further change. |
| Participation sweep | Removes compositions that are not in top-level use when the population is evaluated. |
| Inlining and rewrites | Folds a removed composition into structures that contain it while preserving decoded atomic content. |
| External environment | Evaluates decoded constructions without seeing or directing the representation that produced them. |

The system starts with only a population of atomic tokens. Boundaries,
compositions, and compositional depth arise during the run.

## Representation

An organism is a sequence containing three kinds of token:

- **Atoms** are the primitive units supplied by the external environment.
- **Boundary markers** identify a temporary region in which capture can occur.
- **Compositions** are immutable tuples containing previously captured
  structure.

A composition carries its content by value. It is not a numeric name pointing
to a separate lookup table. Two compositions with the same content are the same
representational unit, including when that content contains other compositions.

The external environment never evaluates compositions directly. Before
evaluation, every composition is expanded to atoms:

```text
encoded:  var1  (var0, var2, OR)  XOR
decoded:  var1   var0  var2  OR   XOR
```

Fitness is assigned to the decoded construction. The environment can determine
that one construction performs better than another, but it cannot identify,
create, repair, or preserve the internal representation responsible for that
performance.

## One generation

Each generation proceeds in this order:

1. Pending local rewrites are collected by organisms that hold an affected
   top-level composition.
2. Every organism is decoded and evaluated by the external environment.
3. The evaluation records which compositions are in top-level service.
4. Compositions absent from that service set are removed from the Meta-Genome.
5. Removal inlines their content into compositions that referenced them.
6. Differential survival produces the next population.
7. Local mutation creates, moves, removes, captures, opens, or replaces
   structure.

The ordering matters. A composition created during mutation is held by the
organism that created it and faces its first participation test at the next
evaluation. Deletion does not scan through the population to repair organisms.
A changed structure is returned through the same decode interface when an
organism next presents the superseded form.

## The demonstration environment

`mux_fitness.py` supplies a postfix Boolean multiplexer environment.

With the default `selection_lines: 2`:

- the problem has six variables,
- the truth table has 64 rows,
- the atomic alphabet contains six variables plus `NOT`, `AND`, `OR`, and
  `XOR`,
- each organism is evaluated across the entire truth table.

The default `rebalanced` score rewards correct rows and separately penalizes
program length, operator underflow, and unused stack residue. The older
`original` scoring is retained for comparison.

MUX is a demonstration environment, not part of the definition of GADS. A
different environment can be bound by supplying an atomic alphabet and a
fitness object whose `score(atoms)` method evaluates decoded constructions.

## Files

| File | Purpose |
| --- | --- |
| `GADS.py` | Complete GADS machine and representational lifecycle |
| `mux_fitness.py` | Multiplexer alphabet, interpreter, and fitness functions |
| `run.py` | Configuration binding, execution loop, reporting, and result output |
| `config.json` | Reproducible default run |
| `.github/workflows/smoke.yml` | Compilation, evaluator, and execution checks |

## Configuration

Top-level settings control the run and environment:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `seed` | `42` | Random seed |
| `generations` | `300` | Maximum generation |
| `selection_lines` | `2` | MUX address-line count: 2 selects MUX6, 3 selects MUX11 |
| `scoring` | `rebalanced` | MUX scoring rule |
| `report_every` | `10` | Console and trajectory reporting interval |
| `stop_on_perfect` | `true` | Stop when every truth-table row is correct |
| `save_result` | `results/last_run.json` | JSON output path, or `null` |

The `gads` object controls the machine:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `POPULATION_SIZE` | `300` | Number of organisms |
| `MIN_LEN` | `2` | Initialization and deletion floor |
| `MAX_LEN` | `100` | Maximum initial length, not a runtime length cap |
| `NUM_PARENTS` | `200` | Highest-ranked organisms eligible as parents |
| `ELITE_FRAC` | `0.02` | Fraction copied without mutation |
| `ENABLE_CROSSOVER` | `false` | Enable boundary-safe single-point crossover |
| `CROSSOVER_PROB` | `0.5` | Crossover probability when enabled |
| `BOUNDARY_INSERT_PROB` | `0.0035` | Boundary-pair creation probability |
| `BOUNDARY_REMOVE_PROB` | `0.002` | Boundary-pair removal probability |
| `CAPTURE_PROB` | `0.09` | Capture probability inside a boundary |
| `MIN_CAPTURE_LEN` | `2` | Minimum captured span |
| `MUTATION_PROB` | `0.05` | Mutation rate outside boundaries |
| `BOUNDARY_MUTATION_PROB` | `0.02` | Mutation rate inside boundaries |
| `OPEN_PROB` | `0.005` | Composition-opening probability |
| `BASE_GENE_PROB` | `0.57` | Probability that token sampling selects an atom |
| `MCO_DECAY` | `0.9` | Recency weighting over available compositions |
| `DB_THRESHOLD` | `1` | Lifetime of an unclaimed structural rewrite |
| `USE_FITNESS` | `true` | Use differential survival rather than uniform reproduction |

Any machine setting can be overridden without editing `config.json`:

```bash
python run.py \
  --seed 7 \
  --generations 500 \
  --set POPULATION_SIZE=500 \
  --set CAPTURE_PROB=0.05
```

## Output

The console reports:

- best score and truth-table correctness,
- encoded and decoded lengths,
- active Meta-Genome size,
- current service-set size,
- cumulative captures,
- cumulative composition deaths.

The JSON result contains the resolved configuration, elapsed time, event totals,
reported trajectory, best-ever organism, and final best organism. It stores the
best decoded program, not the full population history.

With a fixed Python version, configuration, and seed, runs are deterministic.

## What this repository establishes

This repository makes the single-system architecture executable and
inspectable. It shows how the representational lifecycle is implemented and
provides a reproducible environment in which its dynamics can be observed.

It does not establish that every parameterization is stable, that GADS will
outperform fixed-representation search on every problem, or that the presence
of the architecture alone guarantees adaptation or biological equivalence.
Those are empirical questions.

For the conceptual introduction, see
[What Is GADS?](https://www.fsadb.org/what-is-gads/).

For the architectural derivation, see
[GADS Foundations](https://www.fsadb.org/gads-foundations/).

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
