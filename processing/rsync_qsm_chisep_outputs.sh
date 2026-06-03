#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-tsalo@bblsub2.pmacs.upenn.edu:/project/nibs_data/chisep_20260522}"
PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
RSYNC_RSH="${RSYNC_RSH:-ssh -S none -o ServerAliveInterval=60 -o ServerAliveCountMax=5}"
JOBS=1
SUBJECTS=()

usage() {
    cat <<'USAGE'
Usage:
  rsync_qsm_chisep_outputs.sh [options] [<subject> ...]

Copy chi-separation outputs generated on bblsub2 back to CUBIC while
preserving paths relative to PROJECT_ROOT.

Arguments:
  subject                 BIDS subject label, with or without "sub-"
                          default: all subjects under PROJECT_ROOT/dset

Options:
  -r, --remote REMOTE     rsync source
                          default: tsalo@bblsub2.pmacs.upenn.edu:/project/nibs_data/chisep_20260522
  -p, --project-root DIR  local CUBIC project root
                          default: /cbica/projects/nibs
  -j, --jobs N            number of parallel rsync workers
                          default: 1
  -n, --dry-run           show what would be copied
  -h, --help              show this help

Environment:
  REMOTE                  override the default remote source
  PROJECT_ROOT            override the default local destination root
  RSYNC_RSH               override the SSH command used by rsync
                          default: ssh -S none -o ServerAliveInterval=60 -o ServerAliveCountMax=5

Example:
  ./processing/rsync_qsm_chisep_outputs.sh
  ./processing/rsync_qsm_chisep_outputs.sh sub-60526
  ./processing/rsync_qsm_chisep_outputs.sh --jobs 4 60526 60522
  ./processing/rsync_qsm_chisep_outputs.sh --dry-run
USAGE
}

# With --files-from, rsync does not let -a imply recursion, so --recursive is
# required when the file list contains directories.
RSYNC_OPTS=(-avh --recursive --copy-links --relative --prune-empty-dirs --ignore-missing-args)
VARIANTS=(
    "E12345+chisep+r2p"
    "E2345+chisep+r2p"
    "E12345+chisep+r2primenet"
    "E2345+chisep+r2primenet"
    "E12345+chisep+r2s"
    "E2345+chisep+r2s"
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--remote)
            REMOTE="$2"
            shift 2
            ;;
        -p|--project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -n|--dry-run)
            RSYNC_OPTS+=(--dry-run)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            SUBJECTS+=("${1#sub-}")
            shift
            ;;
    esac
done

if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
    exit 1
fi

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer, got: $JOBS" >&2
    exit 2
fi

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    while IFS= read -r -d '' subject_dir; do
        SUBJECTS+=("$(basename "$subject_dir" | sed 's/^sub-//')")
    done < <(compgen -G "$PROJECT_ROOT/dset/sub-*" | while IFS= read -r path; do [[ -d "$path" ]] && printf '%s\0' "$path"; done)

    if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
        echo "No subjects found under $PROJECT_ROOT/dset." >&2
        exit 1
    fi
fi

tmp_filelist="$(mktemp)"
tmp_chunk_dir="$(mktemp -d)"
cleanup() {
    rm -f "$tmp_filelist"
    rm -rf "$tmp_chunk_dir"
}
trap cleanup EXIT

for subject in "${SUBJECTS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        printf '%s\0' "work/qsm-${variant}/sub-${subject}" >> "$tmp_filelist"
    done
done

print_transfer_summary() {
    local dry_run="no"
    local rel_path

    for opt in "${RSYNC_OPTS[@]}"; do
        if [[ "$opt" == "--dry-run" ]]; then
            dry_run="yes"
            break
        fi
    done

    echo "Preparing to copy chi-sep outputs"
    echo "  Source root: $REMOTE/"
    echo "  Target root: $PROJECT_ROOT/"
    echo "  Workers: $JOBS"
    echo "  Dry run: $dry_run"
    echo "  SSH command: $RSYNC_RSH"
    echo "  Missing source folders: skipped"
    echo "  Relative folders:"
    while IFS= read -r -d '' rel_path; do
        echo "    Source: $REMOTE/$rel_path/"
        echo "    Target: $PROJECT_ROOT/$rel_path/"
    done < "$tmp_filelist"
}

run_rsync() {
    local filelist="$1"
    rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" --files-from="$filelist" --from0 "$REMOTE/" "$PROJECT_ROOT/"
}

print_transfer_summary
if [[ "$JOBS" -eq 1 ]]; then
    run_rsync "$tmp_filelist"
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required when --jobs is greater than 1." >&2
        exit 1
    fi

    python3 - "$tmp_filelist" "$tmp_chunk_dir" "$JOBS" <<'PY'
from __future__ import annotations

import pathlib
import sys

filelist = pathlib.Path(sys.argv[1])
chunk_dir = pathlib.Path(sys.argv[2])
jobs = int(sys.argv[3])

paths = [path for path in filelist.read_bytes().split(b"\0") if path]
chunks = [bytearray() for _ in range(jobs)]
for index, path in enumerate(paths):
    chunks[index % jobs].extend(path + b"\0")

for index, chunk in enumerate(chunks):
    (chunk_dir / f"chunk_{index:03d}.lst").write_bytes(chunk)
PY

    pids=()
    for chunk_file in "$tmp_chunk_dir"/chunk_*.lst; do
        [[ -s "$chunk_file" ]] || continue
        run_rsync "$chunk_file" &
        pids+=("$!")
    done

    status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    exit "$status"
fi
