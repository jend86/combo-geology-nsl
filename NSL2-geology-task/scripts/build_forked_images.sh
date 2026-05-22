#!/usr/bin/env bash
# Stage variation rpc_caches into the compose build context and build images.
#
# Usage:
#     ./scripts/build_forked_images.sh

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
variations_root="${repo_root}/tasks/forked_exploit_variations"
compose_dir="${repo_root}/docker/forked-exploit-compose"
staging="${compose_dir}/staging/rpc_caches"

echo "Staging rpc_caches → ${staging}"
mkdir -p "${staging}"
# Clear previously staged caches but preserve .gitkeep so the tracked
# dir survives a clean build.
find "${staging}" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +

shopt -s nullglob
for vdir in "${variations_root}"/*/; do
    name="$(basename "${vdir}")"
    if [[ "${name}" == _* ]]; then
        continue
    fi
    src="${vdir}rpc_cache"
    if [[ -d "${src}" ]]; then
        echo "  ${name}"
        # Merge into staging — each variation's cache is already namespaced
        # by chain/block, so overlaying is safe.
        cp -a "${src}/." "${staging}/"
    fi
done

total_bytes=$(du -sb "${staging}" 2>/dev/null | cut -f1)
printf "Staged %d bytes (%s)\n" "${total_bytes}" "$(du -sh "${staging}" | cut -f1)"

echo "Running docker compose build..."
cd "${compose_dir}"
docker compose build

echo "Done."
echo
echo "Next: warm the runtime rpc-cache sqlite (shared across slots + runs):"
echo "    uv run python scripts/warm_rpc_sqlite_cache.py"
echo "Populates \$RPC_CACHE_HOST_DIR/cache.sqlite with fork-startup + grid reads"
echo "for every variation, so the first real training run avoids Alchemy."
