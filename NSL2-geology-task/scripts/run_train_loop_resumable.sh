#!/usr/bin/env bash
# Start a resumable training run. Writes a state file so a reboot during the
# run is picked up by the nsl2-resume.service systemd user unit on next boot.
#
# State file path is intentionally shared with the sister nsl2 repo
# (~/.local/state/nsl2/active-run), so only one resumable run can be active
# at a time across both repos. PROJECT_DIR in the state file is what makes
# the existing nsl2-resume.service repo-agnostic — it cd's there and re-runs
# `scripts/run_train_loop.py --run-id $RUN_ID`, which combo's run loop now
# supports.
#
# Usage:
#   scripts/run_train_loop_resumable.sh <config-path> [<run-id>]
#
# Exit code semantics drive the state file lifecycle:
#   - 0                       : run finished cleanly      -> state file removed
#   - SIGINT / SIGTERM (130/143): user aborted             -> state file removed
#   - any other non-zero exit : crash / hardware failure  -> state file kept,
#                                                            boot resumer retries

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/nsl2"
STATE_FILE="$STATE_DIR/active-run"

usage() {
    echo "usage: $(basename "$0") <config-path> [<run-id>]" >&2
    exit 2
}

[[ $# -lt 1 || $# -gt 2 ]] && usage

CONFIG_ARG="$1"
RUN_ID_ARG="${2:-}"

if [[ ! -f "$CONFIG_ARG" ]]; then
    echo "error: config file not found: $CONFIG_ARG" >&2
    exit 1
fi
CONFIG_PATH="$(cd -- "$(dirname -- "$CONFIG_ARG")" && pwd)/$(basename -- "$CONFIG_ARG")"

if [[ -e "$STATE_FILE" ]]; then
    echo "error: state file already exists at $STATE_FILE" >&2
    echo "       another resumable run appears to be active or crashed." >&2
    echo "       inspect it, then remove it manually if you want a fresh run." >&2
    exit 1
fi

if [[ -z "$RUN_ID_ARG" ]]; then
    # Mirrors src.helper.generate_readable_run_id (YYYYMMDD-<6 lowercase alnum>).
    RUN_ID="$(date +%Y%m%d)-$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c6)"
else
    RUN_ID="$RUN_ID_ARG"
fi

mkdir -p "$STATE_DIR"
TMP_STATE="$(mktemp "$STATE_FILE.XXXXXX")"
cat >"$TMP_STATE" <<EOF
PROJECT_DIR=$PROJECT_DIR
CONFIG_PATH=$CONFIG_PATH
RUN_ID=$RUN_ID
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
mv "$TMP_STATE" "$STATE_FILE"

echo "[resumable] state -> $STATE_FILE"
echo "[resumable]   config = $CONFIG_PATH"
echo "[resumable]   run_id = $RUN_ID"

cleanup_state() {
    rm -f "$STATE_FILE"
}

# Treat user-initiated aborts as "do not resume on next boot".
# SIGKILL bypasses these traps, so a watchdog --force reboot leaves
# the state file in place for nsl2-resume.service to pick up.
trap 'cleanup_state; exit 130' INT
trap 'cleanup_state; exit 143' TERM

cd "$PROJECT_DIR"
set +e
nix develop --command bash -lc \
    "uv run python scripts/run_train_loop.py --config '$CONFIG_PATH' --run-id '$RUN_ID'"
status=$?
set -e

if [[ $status -eq 0 ]]; then
    cleanup_state
fi
exit "$status"
