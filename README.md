# Drone SOC

This repository contains the implementation and numerical experiments for
*Scalable Multi-Agent Gas Source Localization via Continuous-Discrete
Stochastic Optimal Control*. The code compares one-body, pairwise, greedy
expected-information-gain (EIG), and patrol controllers for cooperative gas
source localization.

Run all commands below from the repository root.

## Environment

Activate the Python environment you want to use before running any command.
The reproduction script uses the `python` executable currently on `PATH`; it
does not activate or select a Conda environment.

The numerical and plotting code requires Python with NumPy, SciPy, Matplotlib,
and pandas. Running the tests also requires pytest. Building the paper requires
a TeX installation with `latexmk`.

The archived runs record Python 3.12.13, NumPy 2.4.6, SciPy 1.18.0,
Matplotlib 3.11.0, and pandas 3.0.3. The release environment matches these
versions. See [Release provenance and verification](reproducibility/README.md)
for the scope of this match.

On Linux x86-64, use the validated Conda and pip locks:

```sh
conda create -n drone-soc --file conda-linux-64.lock
conda activate drone-soc
export PYTHONNOUSERSITE=1
python -m pip install -r requirements-linux-64.lock.txt
```

On other platforms, use the portable specification with the same recorded
versions where builds are available:

```sh
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate drone-soc
export PYTHONNOUSERSITE=1
```

Keep `PYTHONNOUSERSITE=1` when running the release checks so that Python
uses the environment's packages.

You can check the active interpreter and the main dependencies with

```sh
which python
python --version
python -c "import numpy, scipy, matplotlib, pandas"
```

## Reproduce the paper

The complete paper workflow is automated by [`reproduce_paper.sh`](reproduce_paper.sh):

```sh
./reproduce_paper.sh
```

This runs all five paper experiments with the published settings, generates
all figures referenced by `paper/main.tex`, checks that the expected figure
files exist, and compiles `paper/main.pdf`. Experiment data are written under
`results/`, and publication figures are written under `paper/figures/`.

The full experiment suite is computationally expensive. Completed cases are
reused by the experiment drivers where supported. The script also accepts:

```sh
./reproduce_paper.sh --fresh        # archive current paper data, then rerun all cases
./reproduce_paper.sh --plots-only   # regenerate figures from existing results
./reproduce_paper.sh --skip-paper   # run experiments and plots without LaTeX
./reproduce_paper.sh --smoke        # reduced isolated end-to-end execution
./reproduce_paper.sh --plots-only --skip-paper
./reproduce_paper.sh --help
```

The `--smoke` profile runs seed 0 through the baseline controllers and through
one-body/pairwise cooperative planning at discounts 0.3 and 0.1. It uses two
measurement cycles, coarse spatial grids, three belief nodes, and at most eight
fixed-point iterations. Outputs and smoke figures stay under `results/smoke/`;
the command does not compile the manuscript or overwrite publication figures.
`--smoke` cannot be combined with `--fresh` or `--plots-only`.

The `--fresh` workflow archives aggregate CSV/JSON files, current paper
figures, checksums, and Git state under a timestamped
`old_results/paper_pre_patrol_audit_*` directory before removing the active
outputs. The discount sweep then reuses only the newly generated `r=0.3`
showcase cases. After the experiments finish, fixed-point failures, retries,
and boundary-dwell summaries are written to `results/paper_run_audit.json`.

The archived paper datasets record the corrected code revision
`f1ff89cba6c8d46b350503219450a8a59ff71c3f`. Their checksums, configurations,
saved-data metrics, interval tables, and case-level acceptance summaries have
been checked. See [Release provenance and verification](reproducibility/README.md).
Retrieve the existing data before considering new simulations:

```sh
git lfs install
git lfs pull --include='artifacts/paper_results/*.tar.zst'
bash experiments/paper_artifacts.sh extract
```

The archives do not include full fixed-point residual histories,
covariance-clipping logs, or detailed planning-time components. Verification
of these records is separate from an independent rerun of the simulations.

## Paper experiments

The commands in this section are the exact experiment commands used by
`reproduce_paper.sh`. Each driver also supports `-h` or `--help`.

### Wind-aligned versus free-angle HJB

This experiment compares the reduced five-dimensional, wind-aligned HJB with
the six-dimensional free-angle HJB over 100 paired seeds.

```sh
python -u experiments/exp_wind_aligned_comparison.py \
    --num-seeds 100 \
    --variant prior_tuned \
    --discount 0.3 \
    --num-belief-discr 3 \
    --out-dir results/wind_aligned_comparison
```

The main output is
`results/wind_aligned_comparison/wind_aligned_comparison.csv`.

Arguments:

- `--num-seeds`: number of paired random seeds; default `100`.
- `--variant`: pairwise-showcase scenario variant; default `prior_tuned`.
- `--discount`: HJB discount rate; default `0.3`.
- `--num-belief-discr`: grid points per belief axis; default `3`. The free-angle grid has this factor more points than the wind-aligned grid.
- `--out-dir`: result directory; default `results/wind_aligned_comparison`.

Generate its paper figure with

```sh
python -u plotting/plot_wind_aligned_comparison.py \
    --csv results/wind_aligned_comparison/wind_aligned_comparison.csv \
    --out paper/figures/wind_aligned_comparison.pdf
```

### One-body HJB, greedy EIG, and patrol baselines

This experiment compares the three baseline controllers over 50 seeds at four
signal-to-noise levels.

```sh
python -u experiments/exp_greedy_eig_comparison.py \
    --num-seeds 50 \
    --seed-offset 0 \
    --snr0-levels 50 100 150 200 \
    --methods order1 greedy_eig patrol \
    --out-dir results/greedy_eig_comparison
```

It writes per-trial and aggregate data to `per_trial_results.csv` and
`summary_results.csv` inside `results/greedy_eig_comparison/`, together with
the saved data for each run.

Arguments:

- `--num-seeds`: number of seeds; default `50`.
- `--seed-offset`: first seed index; default `0`.
- `--snr0-levels`: space-separated SNR levels; default `50 100 150 200`.
- `--methods`: space-separated controller names; paper values are `order1`, `greedy_eig`, and `patrol`.
- `--out-dir`: result directory; default `results/greedy_eig_comparison`.
- `--overwrite`: rerun cases whose saved metrics and trajectories already exist.

Generate both baseline figures with

```sh
python -u plotting/plot_greedy_eig_comparison.py \
    --results-dir results/greedy_eig_comparison \
    --paper-copy paper/figures/baseline_comparison.pdf \
    --distance-paper-copy paper/figures/baseline_distance_vs_energy.pdf \
    --figures-dir paper/figures \
    --snr0 50 100 150 200
```

### One-body versus pairwise cooperative planning

This is the main cooperative showcase: 100 paired seeds comparing the
one-body (`order1`) and pairwise (`order2`) controllers at discount rate 0.3.

```sh
python -u experiments/exp_paper_pairwise_showcase.py \
    --num-seeds 100 \
    --seed-offset 0 \
    --discount 0.3 \
    --methods order1 order2 \
    --out-dir results/paper_pairwise_showcase
```

It writes `per_trial_results.csv`, `summary_results.csv`, and one directory per
seed and method under `results/paper_pairwise_showcase/`.

Arguments:

- `--num-seeds`: number of paired seeds; default `100`.
- `--seed-offset`: first seed index; default `0`.
- `--discount`: HJB discount rate; default `0.3`.
- `--methods`: one or both of `order1` and `order2`; default is both.
- `--out-dir`: result directory; default `results/paper_pairwise_showcase`.
- `--overwrite`: rerun cases whose `metrics.json` and trajectory data already exist.

The associated paper figures are generated with

```sh
python -u plotting/plot_pairwise_showcase_setting.py \
    --seed 0 --variant prior_tuned --discount 0.3 \
    --out paper/figures/pairwise_showcase_setting.pdf

python -u plotting/plot_pairwise_showcase_summary.py \
    --csv results/paper_pairwise_showcase/per_trial_results.csv \
    --out-dir paper/figures

python -u plotting/plot_order1_vs_order2_distance_vs_energy.py \
    --results-dir results/paper_pairwise_showcase \
    --paper-copy paper/figures/pairwise_showcase_distance_vs_energy.pdf

python -u plotting/plot_pairwise_showcase_paired_improvement.py \
    --csv results/paper_pairwise_showcase/per_trial_results.csv \
    --out results/paper_pairwise_showcase/pairwise_showcase_paired_improvement.png \
    --paper-copy paper/figures/pairwise_showcase_paired_improvement.pdf

python -u plotting/plot_pairwise_showcase_seed_suite.py \
    --seed 10 \
    --methods order1 order2 \
    --discount 0.3 \
    --variant prior_tuned \
    --results-dir results/paper_pairwise_showcase \
    --figures-dir paper/figures
```

The seed-suite command reconstructs the diagnostic plots from each saved
`run_results.npz`. Its additional controls include `--out-dir`, `--t-end`,
`--dt-meas`, `--dx`, `--dx-secondary`, `--num-belief-discr`, and
`--fp-max-iter`; use `--help` for their defaults.

### Discount-rate sweep

This experiment compares the one-body and pairwise controllers at five
discount rates. It reuses the discount-0.3 showcase runs by default.

```sh
python -u experiments/exp_paper_discount_sweep.py \
    --num-seeds 100 \
    --seed-offset 0 \
    --discounts 0.7 0.5 0.3 0.2 0.1 \
    --methods order1 order2 \
    --out-dir results/paper_pairwise_discount_sweep \
    --reuse-dir results/paper_pairwise_showcase \
    --reuse-discount 0.3
```

It writes `per_trial_results.csv`, `summary_results.csv`, and per-run data
under `results/paper_pairwise_discount_sweep/`.

Arguments:

- `--num-seeds`: number of seeds per discount and method; default `100`.
- `--seed-offset`: first seed index; default `0`.
- `--discounts`: space-separated discount rates; default `0.7 0.5 0.3 0.2 0.1`.
- `--methods`: one or both of `order1` and `order2`; default is both.
- `--out-dir`: result directory; default `results/paper_pairwise_discount_sweep`.
- `--reuse-dir`: directory containing compatible showcase results; default `results/paper_pairwise_showcase`.
- `--reuse-discount`: discount represented by the reused results; default `0.3`.
- `--no-reuse`: calculate every requested discount instead of reusing showcase results.
- `--overwrite`: rerun cases whose metrics already exist.

Generate the two discount-sweep figures with

```sh
python -u plotting/plot_discount_sweep_outcomes.py \
    --results-dir results/paper_pairwise_discount_sweep \
    --paper-copy paper/figures/discount_sweep_outcomes.pdf

python -u plotting/plot_discount_sweep_summary.py \
    --results-dir results/paper_pairwise_discount_sweep \
    --paper-copy paper/figures/discount_sweep_summary.pdf
```

### Scaling with the number of drones

This experiment evaluates the pairwise controller with two, three, and four
drones over 50 seeds.

```sh
python -u experiments/exp_scaling_M.py \
    --num-seeds 50 \
    --seed-offset 0 \
    --M-values 2 3 4 \
    --discount 0.3 \
    --out-dir results/scaling_M
```

It writes `per_trial_results.csv`, `summary_results.csv`, and per-run data
under `results/scaling_M/`.

Arguments:

- `--num-seeds`: number of seeds for each team size; default `50`.
- `--seed-offset`: first seed index; default `0`.
- `--M-values`: space-separated numbers of drones; default `2 3 4`.
- `--discount`: HJB discount rate; default `0.3`.
- `--out-dir`: result directory; default `results/scaling_M`.

Generate the scaling figure with

```sh
python -u plotting/plot_scaling_M.py \
    --results-dir results/scaling_M \
    --paper-copy paper/figures/scaling_M.pdf
```

## Appendix validation experiments

The following retained validation studies are not currently part of the paper
reproduction script. They have fixed configurations rather than command-line
arguments:

```sh
python -m experiments.appendix.exp_hjb_validation
python -m experiments.appendix.exp_hjb_convergence
python -m experiments.appendix.exp_compare_plume_models
```

The first two write HJB refinement and convergence figures under
`results/hjb_validation/`. The plume-model comparison writes PDF and PNG
figures under `paper/figures/` and requires a working LaTeX installation for
Matplotlib's TeX text rendering.

## Internal validations

The audit-driven modeling checks are intentionally separate from pytest and
from the paper workflow. Run the inexpensive deterministic profile with

```sh
python -u experiments/validation/run_internal_validations.py --profile smoke
```

Use `--profile full` manually for the larger Monte Carlo profile. Outputs go to
`results/internal_validation/<profile>/`. These checks include periodic versus
no-continuation behavior, block versus full information, grid sensitivity, and
stopping-score calibration; they are candidates for a future appendix, not
current paper experiments.

## Tests

Run the test suite from the repository root:

```sh
pytest -q
```

## Repository layout

- `drone_soc.py`, `measurement.py`, `hjb.py`, and the planner modules contain the numerical implementation.
- `experiments/` contains the active experiment drivers and shared scenarios.
- `experiments/validation/` contains optional internal validation profiles.
- `plotting/` contains publication and diagnostic plotting tools.
- `paper/` contains the manuscript and generated paper figures.
- `results/` contains generated numerical outputs.
- `reproducibility/` contains the results-generating source snapshot and controller regression evidence.
