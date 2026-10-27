# Stochastic energy-harvesting precision bound — simulation and analysis code

This repository contains the code and the computed results that accompany a study of a
parameter-free **precision (kinetic) bound** on the fluctuations of the work harvested from
broadband random vibration, together with the associated **nonlinear fluctuation-control
design law**. It includes the experiment scripts, the figure-generation scripts, the test
suite, the JSON result files and figures, and the input data needed to regenerate them
(with the exception of third-party measured records; see *Data provenance*).

The scientific claim has two parts. For a *linear* harvester driven by broadband Gaussian
vibration, the relative variance of the harvested work obeys the parameter-free relation
`ε²(t) · t · B_eff → 2` as the observation window `t` grows, where `B_eff` is an effective
bandwidth fixed by the signal's own autocorrelation; this is a kinetic rather than a
thermodynamic ceiling. *Nonlinearity* breaks this universal value in a way captured by an
exact two-factor decomposition, so that a stiffening (monostable) device can push the product
below two while a bistable device in its inter-well hopping regime pushes it above two. The
code establishes these results by exact analytics, by direct Langevin simulation, by an
entropy-production bridge, on a real measured ocean-wave spectrum, by an independent
cross-check on a separately specified bistable model, and on measured records of a commercial
piezoelectric harvester.

Author: Qinyou Yang · Repository: <https://github.com/QinyouYang/harvester-precision-bound>

## Requirements

Python ≥ 3.10 and three scientific-Python libraries (see `requirements.txt`):

```
numpy >= 2.0         # array math, random Generator, FFT; numpy.trapezoid requires numpy 2.0+
scipy >= 1.10        # signal.welch, linalg (Lyapunov equation), optimize.curve_fit
matplotlib >= 3.7    # plotting (Agg backend, no display required)
```

The results in `results/` were produced with **Python 3.12.3, numpy 2.4.4, scipy 1.17.1,
matplotlib 3.10.8**. The full test suite and every script were also verified with Python
3.10.12, numpy 2.2.6, scipy 1.15.3 and matplotlib 3.10.9.

*Bit-for-bit reproducibility.* With the versions above (numpy 2.4.4), re-running exp1–exp9 at
full size reproduces every number in `results/` to floating-point precision (checked against
the stored JSON). exp1–exp3, exp5 and exp8 are reproduced to the same precision with other
numpy 2.x versions as well; the Monte-Carlo integrations of the nonlinear oscillators (exp4,
exp6, exp7, exp9) are sensitive to low-level floating-point differences between numpy
releases, so other versions give statistically equivalent but not identical numbers. For exact
reproduction install the pinned versions:

```bash
pip install numpy==2.4.4 scipy==1.17.1 matplotlib==3.10.8    # requires Python >= 3.11
```

## Quick start

All commands are run from the repository root.

```bash
# 0. Get the code
git clone https://github.com/QinyouYang/harvester-precision-bound.git
cd harvester-precision-bound

# 1. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run any single experiment (writes JSON, and PNG where applicable, into results/)
python3 src/exp3_spectral_tur.py

# 4. Run the tests (see "Tests" below)
python3 tests/test_nonlinear_experiments.py
```

Environment variables:

| variable | effect |
|---|---|
| `PIDM_QUICK=1` | fast "smoke" mode (smaller samples, shorter integration, coarser grids) to confirm a script runs end to end; not for reported numbers |
| `PIDM_OUTDIR` | output directory for JSON and figures (default `results/`) |
| `PIDM_DATADIR` | input data directory used by `exp2` and `exp8` (default `data/`) |
| `PIEZO_DATADIR` | location of the measured piezo records for `exp10` (default `data/piezo_costanzo2021/`) |
| `PIDM_NDBC_LIVE=1` | `exp8` only: download the *latest* NDBC spectrum instead of the archived one (result changes over time; written to a separate file) |

Example of a quick trial run that leaves `results/` untouched:

```bash
PIDM_QUICK=1 PIDM_OUTDIR=/tmp/try python3 src/exp1_thermodynamic_bound.py
```

## Reproducing all results

Each experiment fixes its random seed, so repeated runs reproduce the same numbers.

```bash
for n in 1 2 3 4 5 6 7 8 9; do python3 src/exp${n}_*.py; done
python3 src/make_new_figures.py      # fig_fluctuation_control / fig_mechanism_decomposition / fig_activity_kinetic
python3 src/make_exp9_figure.py      # fig_independent_crosscheck (supplementary)

# exp10 needs the third-party measured records (not included; see Data provenance)
python3 src/exp10_piezo_measured_device.py
python3 src/make_exp10_figure.py     # fig_measured_device
```

All scripts write into `results/` by default.

## What each script does

The numbered scripts under `src/` correspond one-to-one to the experiments described in the
manuscript.

| script | content |
|---|---|
| `exp1_thermodynamic_bound.py` | precision-bound baseline (tilted-periodic overdamped model, TUR product vs. load) |
| `exp2_real_wave_harvesting.py` | harvesting from a JONSWAP sea state (realization stored in `data/`) |
| `exp3_spectral_tur.py` | exact work variance via Isserlis' theorem, effective bandwidth `B_eff`, power–precision trade-off |
| `exp4_nonlinear_scope.py` | linear, hardening and bistable harvesters compared |
| `exp5_entropy_bridge.py` | Markovian-embedding entropy-production bridge (Lyapunov equation): kinetic vs. thermodynamic |
| `exp6_fluctuation_control.py` | nonlinear fluctuation-control design law |
| `exp7_mechanism.py` | exact two-factor decomposition of the mechanism |
| `exp8_ndbc_real_spectrum.py` | linear harvester driven by a real measured NDBC buoy spectrum |
| `exp9_torino_bistable.py` | independent cross-check on a separately specified bistable model |
| `exp10_piezo_measured_device.py` | model-free test on measured load-voltage records of a commercial piezoelectric harvester, with phase-randomized stationary surrogates and a modulated-surrogate control |
| `make_*.py` | build the manuscript figures from the stored JSON results |
| `common.py` | shared utilities (wave spectra, spectral synthesis, integrators) |

## Repository layout

```
.
├── README.md                 # this file
├── LICENSE                   # MIT
├── CITATION.cff              # citation metadata
├── requirements.txt
├── src/                      # exp1–exp10, figure scripts, common.py
├── tests/                    # test suite
├── results/                  # computed JSON metrics and figures (the reproducible record)
├── data/
│   ├── wave_jonswap_realization.npz        # synthetic JONSWAP realization (exp2)
│   ├── ndbc_46042_swden_2026_06.txt.gz     # NDBC station 46042 spectral archive, June 2026 (exp8)
│   └── piezo_costanzo2021/README.md        # description of the (not included) measured records (exp10)
```

## Tests

| test | what it checks | runtime |
|---|---|---|
| `tests/test_nonlinear_experiments.py` | physical assertions on exp4–exp9 results (also runnable with `pytest`) | seconds |
| `tests/test_exp10_measured_device.py` | every exp10 number reported in the manuscript can be recovered from `results/exp10_metrics.json` (no raw data needed) | seconds |
| `tests/smoke_test.py` | runs exp1–exp3 in quick mode and asserts 29 physical invariants, including deliberately wrong "sentinel" inputs that must fail | ~1 min |
| `tests/final_validation.py` | checks that the key exp1–exp3 numbers reported in the manuscript are recovered from the JSON, checks figure conventions, and re-runs the smoke test | ~1 min |

```bash
python3 tests/test_nonlinear_experiments.py
python3 tests/test_exp10_measured_device.py
python3 tests/smoke_test.py
python3 tests/final_validation.py
```

Each script prints its checks and exits with a non-zero code on failure.

## Data provenance

- **`data/wave_jonswap_realization.npz`** — a synthetic JONSWAP sea state generated by this
  project (`exp2`).
- **`data/ndbc_46042_swden_2026_06.txt.gz`** — the unmodified NDBC monthly spectral-wave-density
  archive for station 46042 (Monterey, CA), June 2026, downloaded from
  <https://www.ndbc.noaa.gov/data/swden/Jun/4604262026.txt.gz>. `exp8` uses the record of
  2026-06-03 07:40 UTC and runs fully offline. (NDBC data are produced by the U.S. National
  Data Buoy Center and are in the public domain.)
- **`data/piezo_costanzo2021/`** — the measured load-voltage records of the MIDE PPA-4011
  piezoelectric harvester characterized by Costanzo, Lo Schiavo, Sarracino & Vitelli,
  *Entropy* **23**, 677 (2021), doi:10.3390/e23060677. These data belong to the authors of
  that study and are **not redistributed** here; they are available from the original authors
  on request. The computed results (`results/exp10_metrics.json`,
  `results/fig_measured_device.png`) are included, and
  `data/piezo_costanzo2021/README.md` describes how to re-run `exp10` once the data are obtained.

## How to cite

If you use this code, please cite the accompanying article and this repository
(machine-readable metadata in `CITATION.cff`; GitHub shows it under "Cite this repository").

## License

Released under the MIT License (see `LICENSE`). The third-party measured data referred to
above are not covered by this license and are not included.
