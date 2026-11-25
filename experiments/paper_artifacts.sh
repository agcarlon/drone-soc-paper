#!/usr/bin/env bash
# Manage the Git LFS archives containing complete paper experiment outputs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$ROOT_DIR/artifacts/paper_results"
DATASETS=(
    wind_aligned_comparison
    greedy_eig_comparison
    paper_pairwise_showcase
    paper_pairwise_discount_sweep
    scaling_M
)
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage: experiments/paper_artifacts.sh pack|extract|verify

  pack     Create compressed archives from the populated results directories.
  extract  Verify and extract all archives into results/.
  verify   Verify archive checksums without extracting them.
EOF
}

require_archives() {
    [[ -f "$ARCHIVE_DIR/SHA256SUMS" ]] || {
        echo "Missing artifact checksums: $ARCHIVE_DIR/SHA256SUMS" >&2
        exit 1
    }
    for dataset in "${DATASETS[@]}"; do
        [[ -f "$ARCHIVE_DIR/${dataset}.tar.zst" ]] || {
            echo "Missing paper artifact: $ARCHIVE_DIR/${dataset}.tar.zst" >&2
            exit 1
        }
    done
}

verify_archives() {
    require_archives
    (cd "$ARCHIVE_DIR" && shasum -a 256 -c SHA256SUMS)
}

pack_archives() {
    command -v zstd >/dev/null 2>&1 || {
        echo "zstd is required to pack paper artifacts." >&2
        exit 127
    }
    mkdir -p "$ARCHIVE_DIR"
    temporary_manifest="$(mktemp "${TMPDIR:-/tmp}/paper-artifacts.XXXXXX")"
    trap 'rm -f "$temporary_manifest"' EXIT

    for dataset in "${DATASETS[@]}"; do
        result_dir="$ROOT_DIR/results/$dataset"
        [[ -d "$result_dir" ]] || {
            echo "Missing result directory: $result_dir" >&2
            exit 1
        }
        find "results/$dataset" -type f \
            ! -name '*.png' ! -name '*.pdf' \
            ! -name 'fixed_point_diagnostics.json' ! -name '.DS_Store' \
            -print | LC_ALL=C sort > "$temporary_manifest"
        [[ -s "$temporary_manifest" ]] || {
            echo "No artifacts found under $result_dir" >&2
            exit 1
        }
        archive="$ARCHIVE_DIR/${dataset}.tar.zst"
        temporary_archive="${archive}.tmp"
        echo "Packing $dataset ..."
        (cd "$ROOT_DIR" && tar -cf - -T "$temporary_manifest") \
            | zstd -q -T0 -8 -f -o "$temporary_archive"
        mv "$temporary_archive" "$archive"
    done
    (cd "$ARCHIVE_DIR" && shasum -a 256 ./*.tar.zst > SHA256SUMS)
    verify_archives
}

extract_archives() {
    verify_archives
    for dataset in "${DATASETS[@]}"; do
        archive="$ARCHIVE_DIR/${dataset}.tar.zst"
        echo "Extracting $(basename "$archive") ..."
        tar -xf "$archive" -C "$ROOT_DIR"
    done
}

case "${1:-}" in
    pack) pack_archives ;;
    extract) extract_archives ;;
    verify) verify_archives ;;
    *) usage >&2; exit 2 ;;
esac
