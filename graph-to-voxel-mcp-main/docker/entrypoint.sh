#!/usr/bin/env bash
# Dispatch entry for the g2v container — see docs/design/09-docker-runtime.md §5.4.
#
# Usage (set via `CMD` or `docker run g2v:latest <profile> [args...]`):
#   mcp   [args...]   FastMCP stdio server (g2v-mcp)         — default
#   cli   [args...]   Typer CLI (g2v)
#   test  [args...]   pytest -q (or whatever args are passed)
#   shell [args...]   exec the remaining args, else /bin/bash
#   bash  [args...]   alias for shell
#   *                 exec the args verbatim (e.g. `python -c '...'`)
set -euo pipefail

profile="${1:-mcp}"
shift || true

case "$profile" in
  mcp)
    exec g2v-mcp "$@"
    ;;
  cli)
    exec g2v "$@"
    ;;
  test)
    if [ "$#" -eq 0 ]; then
      exec pytest -q
    else
      exec pytest "$@"
    fi
    ;;
  shell|bash)
    if [ "$#" -eq 0 ]; then
      exec /bin/bash
    else
      exec "$@"
    fi
    ;;
  *)
    # Run the verbatim command so `docker run g2v:latest python -c '...'` works.
    exec "$profile" "$@"
    ;;
esac
