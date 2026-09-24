"""Frozen configuration for the Curvature-Renormalization Orbit experiment.

This experiment imports the canonical GADS_single GADS machine by path and
never edits it.  Geometry and MUX use separate experiment-local parameter
blocks because they answer different questions:

* geometry: reproduce curvature-driven drift on a deliberately simple smooth
  degenerate landscape;
* mux: test the same broad phenomenon on the MUX6 task after replacing the
  original scalar score with correctness only.

One seed is one independent population-level replicate.  Conditions sharing a
seed are matched.
"""

CANONICAL_GADS_BLOB = "0975489901e88ea9bcc4145d7f25686948e736e4"
CANONICAL_MUX_BLOB = "a7bf2332ac4b6a512c11867436dbd26caac6568e"

EXPERIMENT_ID = "cro_v3_mux6_probe_split"
SEEDS = tuple(range(1, 17))

# Observation cadence / probe budget.  Probe mutations are read-only: they are
# never inserted into the evolving population.
CHECKPOINT_EVERY = 50
PROBE_ORGS = 24
PROBES_PER_ORG = 8

# ----------------------------- geometry ------------------------------------
GEOMETRY_GENERATIONS = 3000
GEOMETRY_FORK_GENERATION = 1500
GEOMETRY_POPULATION = 300

ELLIPSE_A = 30.0
ELLIPSE_B = 15.0
CIRCLE_RADIUS = 25.0
RIDGE_DISTANCE = 1.0
# Exact lattice points on the optimum curves with the same normalized angle:
# ellipse: 24^2 + 4*9^2 = 30^2; circle: 20^2 + 15^2 = 25^2.
# The ellipse point is not a curvature extremum, so signed motion toward lower
# curvature can be distinguished from ordinary symmetric diffusion.
ELLIPSE_START = (24, 9)
CIRCLE_START = (20, 15)

# Canonical solo-GADS defaults, copied as experiment parameters rather than
# changed in GADS.py.
GEOMETRY_GADS_PARAMS = {
    "POPULATION_SIZE": GEOMETRY_POPULATION,
    "MIN_LEN": 2,
    "MAX_LEN": 100,
    "ENABLE_CROSSOVER": False,
    "NUM_PARENTS": 200,
    "CROSSOVER_PROB": 0.50,
    "BOUNDARY_INSERT_PROB": 0.0035,
    "BOUNDARY_REMOVE_PROB": 0.0020,
    "CAPTURE_PROB": 0.09,
    "MIN_CAPTURE_LEN": 2,
    "MUTATION_PROB": 0.05,
    "BOUNDARY_MUTATION_PROB": 0.02,
    "OPEN_PROB": 0.005,
    "BASE_GENE_PROB": 0.57,
    "MCO_DECAY": 0.90,
    "DB_THRESHOLD": 1,
    "ELITE_FRAC": 0.0,
    "USE_FITNESS": True,
}

# ------------------------------- MUX6 --------------------------------------
MUX_GENERATIONS = 3000
MUX_FORK_GENERATION = 1500
MUX_POPULATION = 300
MUX_SELECTION_LINES = 2

# Official GADS_single single-node defaults from repository config.json.
# The orbit changes only the experiment-local fitness surface to correctness
# only; mutation/capture mechanics and the standalone machine regime remain
# unchanged.
MUX_GADS_PARAMS = {
    "POPULATION_SIZE": MUX_POPULATION,
    "NUM_PARENTS": 200,
    "ELITE_FRAC": 0.02,
    "ENABLE_CROSSOVER": False,
    "CROSSOVER_PROB": 0.50,
    "USE_FITNESS": True,
    "MIN_LEN": 2,
    "MAX_LEN": 100,
    "MUTATION_PROB": 0.05,
    "BOUNDARY_MUTATION_PROB": 0.02,
    "CAPTURE_PROB": 0.09,
    "BOUNDARY_INSERT_PROB": 0.0035,
    "BOUNDARY_REMOVE_PROB": 0.002,
    "OPEN_PROB": 0.005,
    "BASE_GENE_PROB": 0.57,
    "MCO_DECAY": 0.90,
    "MIN_CAPTURE_LEN": 2,
    "DB_THRESHOLD": 1,
}

# Structure knockout.  Point/swap/insert/delete over base atoms remain; only
# the machinery that can create or reuse GADS compositions is disabled.
STRUCTURE_KO = {
    "BOUNDARY_INSERT_PROB": 0.0,
    "CAPTURE_PROB": 0.0,
    "OPEN_PROB": 0.0,
    "BASE_GENE_PROB": 1.0,
}

RESULTS_DIR = "results/cro_v3_mux6_probe_split"
