#!/usr/bin/env bash

# Sourcing this instead of running it makes "set -e" and the final "exit" apply
# to the interactive shell, so any failure closes the terminal. Catch that
# before "set -e" is even in effect.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Run this script (./rsync_pc_to_pmacs.sh), do not source it: sourcing lets a" >&2
    echo "failed transfer exit your shell." >&2
    return 1
fi

set -euo pipefail

# rsync calls getcwd() while starting up, so if the shell that launched this
# script is sitting in a directory that has since been deleted or replaced --
# easy to hit on the /mnt/c DrvFs mount -- every transfer aborts with
# "getcwd(): No such file or directory" before copying a single byte. Relative
# paths are unusable in that state anyway, so re-anchor at /. Note that plain
# "pwd" cannot detect this: the builtin just echoes a stale $PWD.
if ! pwd -P >/dev/null 2>&1; then
    echo "Warning: the working directory no longer exists; running from / instead" >&2
    cd /
fi

REMOTE="${REMOTE:-tsalo@bblsub2.pmacs.upenn.edu:/home/tsalo/nibs/derivatives_20260729}"
DERIVATIVES_ROOT="${DERIVATIVES_ROOT:-/mnt/c/Users/tsalo/Documents/datasets/nibs/derivatives}"
# ControlMaster reuses a single authenticated connection for every dataset, so a
# password-auth account gets prompted once for the whole run instead of once per
# rsync. ServerAlive* only guards an established session, so ConnectTimeout is
# there to keep a dead network from hanging at connect time.
RSYNC_RSH="${RSYNC_RSH:-ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 -o ServerAliveCountMax=5 -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=8h}"
# bblsub2 runs at its sshd MaxStartups limit much of the time. Over its limit,
# sshd drops unauthenticated connections at random: sometimes it answers
# "Exceeded MaxStartups", sometimes the pre-auth child simply never sends a
# version string and the client dies with "timed out during banner exchange".
# Roughly one connection in four fails this way, so a single attempt is a coin
# toss and every connection has to be retried.
SSH_RETRIES="${SSH_RETRIES:-10}"
SSH_RETRY_DELAY="${SSH_RETRY_DELAY:-15}"
RSYNC_RETRIES="${RSYNC_RETRIES:-5}"
RSYNC_RETRY_DELAY="${RSYNC_RETRY_DELAY:-30}"
PRESERVE_PERMS=0
JOBS=1
DRY_RUN=0
DATASETS=()

DEFAULT_DATASETS=(
    "ihmt"
    "pymp2rage"
    "mese"
    "t1wt2w_ratio"
)

usage() {
    cat <<'USAGE'
Usage:
  rsync_pc_to_pmacs.sh [options] [<dataset> ...]

Copy myelin derivative datasets from the local NIBS derivatives root up to
PMACS, one dataset directory per rsync worker.

Arguments:
  dataset                 derivative dataset directory name
                          default: ihmt pymp2rage mese t1wt2w_ratio

Options:
  -r, --remote REMOTE     rsync destination
                          default: tsalo@bblsub2.pmacs.upenn.edu:/home/tsalo/nibs/derivatives_20260729
  -d, --derivatives DIR   local derivatives root
                          default: /mnt/c/Users/tsalo/Documents/datasets/nibs/derivatives
  -j, --jobs N            number of parallel rsync workers
                          default: 1
  --preserve-perms        send the source modes as-is instead of normalizing
                          them to 755/644
  -n, --dry-run           show what would be copied
  -h, --help              show this help

Environment:
  REMOTE                  override the default remote destination
  DERIVATIVES_ROOT        override the default local derivatives root
  RSYNC_RSH               override the SSH command used by rsync
                          default: ssh, with ConnectTimeout/ServerAlive settings
                          and ControlMaster connection sharing
  SSH_RETRIES             attempts to open the shared SSH connection
                          default: 10
  SSH_RETRY_DELAY         seconds between those attempts
                          default: 15
  RSYNC_RETRIES           attempts per dataset after a dropped connection
                          default: 5
  RSYNC_RETRY_DELAY       seconds between those attempts
                          default: 30

Example:
  ./processing/rsync_pc_to_pmacs.sh
  ./processing/rsync_pc_to_pmacs.sh --jobs 4
  ./processing/rsync_pc_to_pmacs.sh --dry-run ihmt
USAGE
}

# --partial keeps interrupted transfers resumable, which matters because these
# datasets run to tens of GB apiece. --timeout turns a silently stalled
# connection into a visible error instead of an indefinite hang; it covers the
# initial protocol handshake, not just bulk transfer.
RSYNC_OPTS=(-avh --copy-links --partial --timeout=600)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--remote)
            REMOTE="$2"
            shift 2
            ;;
        -d|--derivatives)
            DERIVATIVES_ROOT="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --preserve-perms)
            PRESERVE_PERMS=1
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=1
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
            DATASETS+=("${1%/}")
            shift
            ;;
    esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

if [[ ! -d "$DERIVATIVES_ROOT" ]]; then
    echo "DERIVATIVES_ROOT does not exist: $DERIVATIVES_ROOT" >&2
    exit 1
fi

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer, got: $JOBS" >&2
    exit 2
fi

if [[ "$REMOTE" != *:* ]]; then
    echo "REMOTE must be of the form [user@]host:/path, got: $REMOTE" >&2
    exit 2
fi
REMOTE_HOST="${REMOTE%%:*}"
REMOTE_PATH="${REMOTE#*:}"
REMOTE="${REMOTE%/}"

# The Windows mount reports everything as 0777, so copying the source modes
# verbatim would leave world-writable files on the cluster.
if [[ "$PRESERVE_PERMS" -eq 0 ]]; then
    RSYNC_OPTS+=(--no-perms --no-group --chmod=D755,F644)
fi

missing=0
for dataset in "${DATASETS[@]}"; do
    if [[ ! -d "$DERIVATIVES_ROOT/$dataset" ]]; then
        echo "Missing dataset directory: $DERIVATIVES_ROOT/$dataset" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    exit 1
fi

echo "Copying derivatives from $DERIVATIVES_ROOT to $REMOTE with $JOBS rsync worker(s)"
echo "  Datasets: ${DATASETS[*]}"
echo "  Dry run: $([[ "$DRY_RUN" -eq 1 ]] && echo yes || echo no)"
echo "  SSH command: $RSYNC_RSH"

# Establish the shared connection up front, retrying past MaxStartups refusals.
# Once the master is up, the mkdir and every rsync ride over it and skip
# authentication entirely, so only this one connection has to win the lottery.
open_ssh_master() {
    local attempt=1 rc errfile
    errfile="$(mktemp)"
    while :; do
        rc=0
        # shellcheck disable=SC2086  # RSYNC_RSH intentionally word-splits into ssh + flags
        $RSYNC_RSH -N -f "$REMOTE_HOST" 2>"$errfile" || rc=$?
        cat "$errfile" >&2
        if [[ "$rc" -eq 0 ]]; then
            rm -f "$errfile"
            return 0
        fi

        # Never retry a rejected credential; repeated attempts can lock the account.
        if grep -qiE 'permission denied|too many authentication' "$errfile"; then
            echo "Authentication failed, so not retrying." >&2
            rm -f "$errfile"
            return 1
        fi

        if [[ "$attempt" -ge "$SSH_RETRIES" ]]; then
            echo "Could not open an SSH connection after $SSH_RETRIES attempts." >&2
            rm -f "$errfile"
            return 1
        fi

        echo "SSH attempt $attempt/$SSH_RETRIES failed before authentication; retrying in ${SSH_RETRY_DELAY}s" >&2
        sleep "$SSH_RETRY_DELAY"
        attempt=$((attempt + 1))
    done
}

# rsync only creates the final component of the destination, so make sure the
# whole remote path is there before any worker starts.
if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ "$RSYNC_RSH" == *ControlMaster* ]]; then
        open_ssh_master
    fi
    # shellcheck disable=SC2086  # RSYNC_RSH intentionally word-splits into ssh + flags
    $RSYNC_RSH "$REMOTE_HOST" mkdir -p "$REMOTE_PATH"
fi

run_rsync() {
    local dataset="$1"
    local attempt=1 rc
    while :; do
        rc=0
        rsync "${RSYNC_OPTS[@]}" \
            -e "$RSYNC_RSH" \
            "$DERIVATIVES_ROOT/$dataset/" \
            "$REMOTE/$dataset/" || rc=$?
        if [[ "$rc" -eq 0 ]]; then
            return 0
        fi

        # Retry only the connection-level failures: 12 protocol stream error,
        # 30/35 io timeout, 255 ssh itself failed. Anything else (missing files,
        # permissions) will fail again identically. --partial makes the retry
        # resume rather than restart.
        case "$rc" in
            12|30|35|255) ;;
            *) return "$rc" ;;
        esac

        if [[ "$attempt" -ge "$RSYNC_RETRIES" ]]; then
            return "$rc"
        fi

        echo "rsync for $dataset lost its connection (exit $rc); retry $attempt/$RSYNC_RETRIES in ${RSYNC_RETRY_DELAY}s" >&2
        sleep "$RSYNC_RETRY_DELAY"
        attempt=$((attempt + 1))
    done
}

status=0
if [[ "$JOBS" -eq 1 ]]; then
    for dataset in "${DATASETS[@]}"; do
        echo "=== $dataset ==="
        if ! run_rsync "$dataset"; then
            echo "rsync failed for $dataset" >&2
            status=1
        fi
    done
else
    # Run the datasets in batches of JOBS. There are only a handful of them, so
    # batching is plenty and keeps the per-dataset logs from interleaving more
    # than they have to.
    batch_start=0
    while [[ "$batch_start" -lt "${#DATASETS[@]}" ]]; do
        pids=()
        names=()
        for ((index = batch_start; index < batch_start + JOBS && index < ${#DATASETS[@]}; index++)); do
            run_rsync "${DATASETS[index]}" &
            pids+=("$!")
            names+=("${DATASETS[index]}")
        done

        for index in "${!pids[@]}"; do
            if ! wait "${pids[index]}"; then
                echo "rsync failed for ${names[index]}" >&2
                status=1
            fi
        done

        batch_start=$((batch_start + JOBS))
    done
fi

exit "$status"
