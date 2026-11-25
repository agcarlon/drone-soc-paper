#!/usr/bin/env bash
# Reproduce every numerical experiment and figure referenced by paper/main.tex.
#
# Usage:
#   ./reproduce_paper.sh                 # resume experiments, then make figures
#   ./reproduce_paper.sh --fresh         # archive paper data, then rerun from scratch
#   ./reproduce_paper.sh --plots-only    # regenerate figures from existing data
#                                             (restores Git LFS artifacts first)
#   ./reproduce_paper.sh --restore-data  # restore Git LFS artifacts before resuming
#   ./reproduce_paper.sh --skip-paper    # do not compile paper/main.tex
#   ./reproduce_paper.sh --smoke         # isolated reduced end-to-end run

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXPERIMENTS=1
BUILD_PAPER=1
FRESH=0
SMOKE=0
RESTORE_DATA=0

for argument in "$@"; do
    case "$argument" in
        --plots-only)
            RUN_EXPERIMENTS=0
            RESTORE_DATA=1
            ;;
        --restore-data)
            RESTORE_DATA=1
            ;;
        --skip-paper)
            BUILD_PAPER=0
            ;;
        --fresh)
            FRESH=1
            ;;
        --smoke)
            SMOKE=1
            BUILD_PAPER=0
            ;;
        -h|--help)
            sed -n '2,9p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $argument" >&2
            exit 2
            ;;
    esac
done

cd "$ROOT_DIR"
if ! command -v python >/dev/null 2>&1; then
    echo "python was not found on PATH; activate the intended environment first." >&2
    exit 127
fi
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/drone_soc-matplotlib-${UID:-user}"
mkdir -p "$MPLCONFIGDIR" paper/figures

if (( BUILD_PAPER )) && ! command -v latexmk >/dev/null 2>&1; then
    echo "latexmk is required to build paper/main.tex; use --skip-paper to omit this step." >&2
    exit 1
fi
python -c 'import matplotlib, numpy, pandas, scipy'
python_sources=(
    experiments/smoke_paper.py
    experiments/exp_wind_aligned_comparison.py
    experiments/exp_greedy_eig_comparison.py
    experiments/exp_paper_pairwise_showcase.py
    experiments/exp_paper_discount_sweep.py
    experiments/exp_scaling_M.py
    experiments/validate_paper_results.py
    experiments/audit_paper_runs.py
    experiments/compute_paper_interval_tables.py
    plotting/plot_pairwise_showcase_setting.py
    plotting/plot_wind_aligned_comparison.py
    plotting/plot_greedy_eig_comparison.py
    plotting/plot_pairwise_showcase_summary.py
    plotting/plot_order1_vs_order2_distance_vs_energy.py
    plotting/plot_pairwise_showcase_paired_improvement.py
    plotting/plot_pairwise_showcase_seed_suite.py
    plotting/plot_discount_sweep_outcomes.py
    plotting/plot_discount_sweep_summary.py
    plotting/plot_scaling_M.py
)
PYTHONPYCACHEPREFIX="$MPLCONFIGDIR/pycache" python -m py_compile "${python_sources[@]}"
for source in "${python_sources[@]}"; do
    python "$source" --help >/dev/null
done

if (( FRESH && ! RUN_EXPERIMENTS )); then
    echo "--fresh cannot be combined with --plots-only" >&2
    exit 2
fi
if (( FRESH && RESTORE_DATA )); then
    echo "--fresh cannot be combined with --restore-data" >&2
    exit 2
fi
if (( SMOKE && (FRESH || ! RUN_EXPERIMENTS) )); then
    echo "--smoke cannot be combined with --fresh or --plots-only" >&2
    exit 2
fi
if (( SMOKE )); then
    python -u experiments/smoke_paper.py --out-dir results/smoke
    echo "Smoke reproduction complete: outputs are isolated under results/smoke/."
    exit 0
fi

paper_result_dirs=(
    wind_aligned_comparison
    greedy_eig_comparison
    paper_pairwise_showcase
    paper_pairwise_discount_sweep
    scaling_M
    paper_statistical_intervals
)

if (( FRESH )); then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    backup_dir="old_results/paper_pre_patrol_audit_${timestamp}"
    mkdir -p "$backup_dir/results" "$backup_dir/paper"
    for result_dir in "${paper_result_dirs[@]}"; do
        mkdir -p "$backup_dir/results/$result_dir"
        find "results/$result_dir" -maxdepth 1 -type f \
            \( -name '*.csv' -o -name '*.json' \) \
            -exec cp -a {} "$backup_dir/results/$result_dir/" \; 2>/dev/null || true
    done
    if [[ -f results/paper_run_audit.json ]]; then
        cp -a results/paper_run_audit.json "$backup_dir/results/"
    fi
    if [[ -d paper/figures ]]; then
        cp -a paper/figures "$backup_dir/paper/"
    fi
    if [[ -f paper/main.pdf ]]; then
        cp -a paper/main.pdf "$backup_dir/paper/"
    fi
    git rev-parse HEAD > "$backup_dir/git-commit.txt"
    git status --short > "$backup_dir/git-status.txt"
    find "$backup_dir" -type f ! -name 'sha256.txt' -exec shasum -a 256 {} \; \
        > "$backup_dir/sha256.txt"
    shasum -a 256 -c "$backup_dir/sha256.txt" >/dev/null
    printf 'Archived existing paper outputs in %s\n' "$backup_dir"
    for result_dir in "${paper_result_dirs[@]}"; do
        rm -rf "results/$result_dir"
    done
    rm -f results/paper_run_audit.json
    mkdir -p paper/figures
    # Preserve tracked illustrative assets; numerical figures are regenerated below.
    find paper/figures -maxdepth 1 -type f ! -name 'figure_wind_angle.pdf' -delete
fi

overwrite_arg=""
if (( FRESH )); then
    overwrite_arg="--overwrite"
fi

if (( RUN_EXPERIMENTS )); then
    # Wind-aligned 5-D versus full-angle 6-D HJB comparison (100 paired seeds).
    python -u experiments/exp_wind_aligned_comparison.py \
        --num-seeds 100 \
        --variant prior_tuned \
        --discount 0.3 \
        --num-belief-discr 3 \
        --out-dir results/wind_aligned_comparison

    # One-body HJB versus greedy EIG versus patrol (50 seeds at four SNR levels).
    python -u experiments/exp_greedy_eig_comparison.py \
        --num-seeds 50 \
        --seed-offset 0 \
        --snr0-levels 50 100 150 200 \
        --methods order1 greedy_eig patrol \
        --passive-patrol-period 40 \
        --passive-patrol-y-scale 0.99 \
        --out-dir results/greedy_eig_comparison \
        ${overwrite_arg:+"$overwrite_arg"}

    # One-body versus pairwise cooperative showcase (100 paired seeds, r=0.3).
    python -u experiments/exp_paper_pairwise_showcase.py \
        --num-seeds 100 \
        --seed-offset 0 \
        --discount 0.3 \
        --methods order1 order2 \
        --out-dir results/paper_pairwise_showcase \
        ${overwrite_arg:+"$overwrite_arg"}

    # Discount sweep. The r=0.3 cases are reused from the preceding showcase.
    python -u experiments/exp_paper_discount_sweep.py \
        --num-seeds 100 \
        --seed-offset 0 \
        --discounts 0.7 0.5 0.3 0.2 0.1 \
        --methods order1 order2 \
        --out-dir results/paper_pairwise_discount_sweep \
        --reuse-dir results/paper_pairwise_showcase \
        --reuse-discount 0.3

    # Pairwise-controller scaling with two, three, and four drones (50 seeds).
    python -u experiments/exp_scaling_M.py \
        --num-seeds 50 \
        --seed-offset 0 \
        --M-values 2 3 4 \
        --discount 0.3 \
        --out-dir results/scaling_M \
        ${overwrite_arg:+"$overwrite_arg"}

fi

if (( RESTORE_DATA )); then
    experiments/paper_artifacts.sh extract
fi

# Fail with a single completeness report before statistics or plotting consume
# partial, duplicate, or stale aggregate tables.
python -u experiments/validate_paper_results.py --results-dir results
python -u experiments/audit_paper_runs.py \
    --results-dir results \
    --out results/paper_run_audit.json

# Derived Wilson and BCa interval tables from the per-trial CSV files.
python -u experiments/compute_paper_interval_tables.py \
    --out-dir results/paper_statistical_intervals

# Scenario and wind-alignment figures.
python -u plotting/plot_pairwise_showcase_setting.py \
    --seed 0 \
    --variant prior_tuned \
    --discount 0.3 \
    --out paper/figures/pairwise_showcase_setting.pdf
python -u plotting/plot_wind_aligned_comparison.py \
    --csv results/wind_aligned_comparison/wind_aligned_comparison.csv \
    --out paper/figures/wind_aligned_comparison.pdf

# Baseline aggregate and distance-versus-effort figures.
python -u plotting/plot_greedy_eig_comparison.py \
    --results-dir results/greedy_eig_comparison \
    --paper-copy paper/figures/baseline_comparison.pdf \
    --distance-paper-copy paper/figures/baseline_distance_vs_energy.pdf \
    --figures-dir paper/figures \
    --snr0 50 100 150 200

# Cooperative showcase aggregate, paired, and representative-seed figures.
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
    --seed 45 \
    --methods order1 order2 \
    --discount 0.3 \
    --variant prior_tuned \
    --results-dir results/paper_pairwise_showcase \
    --figures-dir paper/figures

# Discount-sensitivity and team-size figures.
python -u plotting/plot_discount_sweep_outcomes.py \
    --results-dir results/paper_pairwise_discount_sweep \
    --paper-copy paper/figures/discount_sweep_outcomes.pdf
python -u plotting/plot_discount_sweep_summary.py \
    --results-dir results/paper_pairwise_discount_sweep \
    --paper-copy paper/figures/discount_sweep_summary.pdf
python -u plotting/plot_scaling_M.py \
    --results-dir results/scaling_M \
    --paper-copy paper/figures/scaling_M.pdf

expected_figures=(
    # Tracked illustrative asset; intentionally not regenerated.
    figure_wind_angle.pdf
    pairwise_showcase_setting.pdf
    wind_aligned_comparison.pdf
    baseline_comparison.pdf
    baseline_distance_vs_energy.pdf
    pairwise_showcase_summary.pdf
    pairwise_showcase_distance_vs_energy.pdf
    pairwise_showcase_paired_improvement.pdf
    pairwise_showcase_seed045_order1_path.pdf
    pairwise_showcase_seed045_order2_path.pdf
    pairwise_showcase_seed045_belief_convergence_comparison.pdf
    discount_sweep_outcomes.pdf
    discount_sweep_summary.pdf
    scaling_M.pdf
)
for figure in "${expected_figures[@]}"; do
    if [[ ! -s "paper/figures/$figure" ]]; then
        echo "Expected paper figure was not generated: paper/figures/$figure" >&2
        exit 1
    fi
done

if (( BUILD_PAPER )); then
    latexmk -g -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
fi

echo "Reproduction complete: experiment data are in results/ and paper figures are in paper/figures/."
