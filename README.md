# GADS_single

Reference implementation of a single **Generative Adaptive Dynamical System
(GADS)**.

GADS is a theoretical object describing the architecture required for a search
process to author and regulate its own representational language without a
central controller. This repository contains one concrete implementation of
that object: a single population, its mutable compositional representation, and
an external MUX environment that converts individual performance into
differential survival.

The implementation is extracted from the canonical standalone GADS used by
Project River. It has no network or historical-inference dependency.

## Contents

- `GADS.py`: the GADS machine and representational lifecycle
- `mux_fitness.py`: the external multiplexer environment
- `run.py`: a one-command solo runner
- `config.json`: the default run configuration

## Run

Python 3.11 or newer is recommended. The implementation uses only the standard
library.

```bash
python run.py
```

The runner loads `config.json`, binds the MUX environment, initializes one
GADS instance, and executes the run. Runtime results are written under
`results/`, which is ignored by Git.

Useful overrides:

```bash
python run.py --seed 7 --generations 100 --no-save
python run.py --set POPULATION_SIZE=500 --set CAPTURE_PROB=0.05
```

Run `python run.py --help` for the complete command-line interface.

## Canonical source

The initial machine, evaluator, runner, and configuration were imported from
`canonical/gads/` at River commit
`cb0ff298bf580718333d4938a331e8727d02e369`.

At import, the source blobs were:

- `GADS.py`: `829e9bd74f8f403207cc41a730a6e429501fe75e`
- `mux_fitness.py`: `a7bf2332ac4b6a512c11867436dbd26caac6568e`
- `run.py`: `74e37cebc8dfa838d1c2bf27ff5ca8f65c7c2385`
- `config.json`: `dbd6fe1e69b086829095fa9cf89898f0727f7d6a`

These are Git blob identifiers, not checksums of scientific validity. They
record the exact River sources from which this repository began.

## Scope

This repository implements one GADS and provides a reproducible demonstration
environment. It does not include River's network, inference apparatus,
experimental histories, or claims that the architecture alone guarantees
stable dynamics, adaptation, or biological equivalence.

For an introduction, see
[What Is GADS?](https://www.fsadb.org/what-is-gads/). For the theoretical
derivation, see
[GADS Foundations](https://www.fsadb.org/gads-foundations/).

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
