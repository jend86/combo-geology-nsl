#!/usr/bin/env python3
"""Provision / inspect / tear down a RunPod L40S pod serving gemma-4-31B-AWQ via vLLM.

Supplemental *external* inference endpoint for the Kazakhstan feature-hypothesis
run: one L40S targeting 16 external concurrent slots alongside any local
endpoint capacity.

The pod runs the SAME pinned vLLM image as the local box for chat-template / tool
parser parity, single-GPU (TP=1/PP=1 — no pipeline bubble), AWQ auto-detected,
fp8 KV cache, prefix caching on. Port 8000 is exposed over **raw TCP** (not the
RunPod HTTP proxy) so the no-streaming, uncapped-output request path is not
capped by the proxy's ~100 s timeout. The vLLM bearer key is passed via the
``VLLM_API_KEY`` env var (vLLM reads it natively) so it never appears in the
pod's visible docker args.

Usage (runpod SDK provided ephemerally by uv; secrets sourced out-of-repo):

    source ~/.config/nsl-runpod/secrets.env
    uv run --with runpod python scripts/provision_runpod_vllm.py up
    uv run --with runpod python scripts/provision_runpod_vllm.py status <pod_id>
    uv run --with runpod python scripts/provision_runpod_vllm.py down <pod_id>

Reads RUNPOD_API_KEY (account, for provisioning) and RUNPOD_VLLM_KEY (the vLLM
bearer the endpoint pool sends) from the environment. Secrets are never written
to disk or echoed to stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

# Pinned to the stable RELEASE matching the local box's known-good lineage.
# The journey here (2026-06-01): the local box runs nightly-01d4d1ad (vLLM
# 0.20.2rc1.dev9) and serves gemma-4-31B-AWQ perfectly -- but that SHA tag was
# PRUNED from Docker Hub (unpullable on a fresh RunPod host). Switching to floating
# :nightly made it pullable but pulled a NEWER build that REGRESSED gemma-4: vLLM
# crash-looped at startup (uptime resetting ~20-30s, VRAM 0%, never served). :latest
# is now v0.22.0 -- also past the regression. The fix is a stable release on the SAME
# 0.20.2 lineage as the working local image: release tags are NOT pruned, and v0.20.2
# final is rc1+N commits => it INCLUDES 01d4d1ad's gemma-4 support while predating the
# regression window. Re-verify pullable + that it still serves gemma-4 before bumping.
PINNED_VLLM_IMAGE = "vllm/vllm-openai:v0.20.2"
DEFAULT_MODEL = "QuantTrio/gemma-4-31B-it-AWQ"
GPU_TYPE_L40S = "NVIDIA L40S"  # exact gpuTypeId from RunPod gpuTypes()

# AMD / ROCm path — for the MI300X throughput-per-dollar pilot (192 GB VRAM to attack the
# re-prefill waste; see docs/design/throughput-per-dollar-and-mi300x-pilot-2026-06-10.md).
# The CUDA image + cuda guard do NOT apply on AMD: pass --image ROCM_VLLM_IMAGE,
# --allowed-cuda-versions with NO values (drops the guard), and --kv-cache-dtype auto (fp16 KV;
# 192 GB has the room, and ROCm fp8-KV is unproven). VERIFY the exact gpuTypeId via gpuTypes()
# before launch — RunPod has listed both "AMD Instinct MI300X" and "...MI300X OAM".
ROCM_VLLM_IMAGE = "rocm/vllm:latest"
GPU_TYPE_MI300X = "AMD Instinct MI300X OAM"


def _bearer(vllm_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {vllm_key}"} if vllm_key else {}


def _http_ok(url: str, headers: dict[str, str], timeout: float = 6.0) -> bool:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout
        ) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError:  # 401/403 => server up but auth mismatch
        return False
    except Exception:
        return False


def _public_tcp(pod: dict, private_port: int = 8000) -> tuple[str | None, int | None]:
    """Return (ip, public_port) for the public TCP mapping of ``private_port``."""
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if (
            port.get("privatePort") == private_port
            and port.get("isIpPublic")
            and str(port.get("type", "")).lower() == "tcp"
        ):
            return port.get("ip"), port.get("publicPort")
    return None, None


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(
            f"ERROR: {name} is not set. Run `source ~/.config/nsl-runpod/secrets.env` first."
        )
    return value


def _docker_args(args: argparse.Namespace) -> str:
    # Bare api_server args (the pinned image's entrypoint takes them directly,
    # exactly like the local _build_container_command). No --api-key here: vLLM
    # reads VLLM_API_KEY from the env we inject. No --tensor/pipeline-parallel:
    # single L40S defaults to 1/1. AWQ is auto-detected from the repo config.
    # --max-num-batched-tokens stays >= 2496 (gemma-4 multimodal floor) but is
    # NOT the local box's crippled 4096 decode-starvation hack — the L40S has KV
    # headroom for a healthy prefill budget.
    # serve_cmd prefix: the NVIDIA vllm/vllm-openai image has ENTRYPOINT
    # `python3 -m vllm.entrypoints.openai.api_server`, so docker_args are bare flags. ROCm
    # nightly images (rocm/vllm-dev:nightly_*) have CMD ["/bin/bash"] and NO server entrypoint,
    # so docker_args must START with the server command. Pass
    # --serve-cmd "python3 -m vllm.entrypoints.openai.api_server" for those. RunPod injects the
    # AMD devices (kfd/dri) for AMD gpu-types, so no --device flags are needed here.
    prefix = args.serve_cmd.split() if getattr(args, "serve_cmd", None) else []
    return " ".join(
        prefix
        + [
            "--model", args.model,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", str(args.gpu_mem_util),
            "--max-model-len", str(args.max_model_len),
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            # KV dtype is arch-sensitive: fp8(=e4m3) needs Ada/Hopper (sm_89+, L40S/4090).
            # On Ampere (A40, sm_86) use 'auto': e4m3 dies at config validation ("fp8e4nv not
            # supported in this architecture", ~20s) AND fp8_e5m2 passes config but asserts in
            # the attention forward (backend allows only fp8/e4m3/nvfp4) in v0.20.2. See flag.
            "--kv-cache-dtype", args.kv_cache_dtype,
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
        ]
        + _compile_args(args)
        + _lora_args(args)
    )


def _compile_args(args: argparse.Namespace) -> list[str]:
    """CUDA-graph / torch.compile controls. DEFAULT IS NOW GRAPHS-ON (no --enforce-eager).

    History: --enforce-eager was hardcoded here (commit 0e1c001, 2026-06-03) purely to
    dodge cold-boot cost on ephemeral pods ("fresh pod has no compile cache => default
    compilation adds minutes + OOM risk"), NOT to fix a crash. Its comment guessed eager
    cost "~10-20% decode". A 2026-06-08 /metrics read first looked like ~20x (the eager decode
    loop is CPU-launch-bound), but the 2026-06-10 throughput/$ analysis CORRECTED this: the
    workload is PREFILL-time-dominated (~84:1 input:output; each step is mostly a chunked-prefill
    of ~10K-tok prompts), so eager's real penalty is the decode SLICE only — ~10-25% overall, not
    20x (decode is ~5% of GPU-time). cudagraph is therefore a minor, risk-gated lever (GH#32834
    awq_marlin full-graph crash, GH#39914 gemma-4 >4K-prefill hang), and --enforce-eager is a safe
    default. See docs/design/throughput-per-dollar-and-mi300x-pilot-2026-06-10.md. Eager is OPT-IN.

      --enforce-eager      : disable BOTH torch.compile and cudagraphs. Fast, OOM-safe cold
                             boot; cripples decode throughput. Keep for boot-constrained or
                             cudagraph-incompatible pods only.
      --compilation-config : raw JSON passthrough to vLLM (the middle ground / fallback
                             ladder for the awq_marlin full-graph crash GH#32834):
                               '{"cudagraph_mode":"PIECEWISE"}'         attention stays
                                   eager -> sidesteps the awq_marlin full-graph replay
                                   crash, still captures the bulk of the launches.
                               '{"cudagraph_mode":"FULL_DECODE_ONLY"}'  full decode graphs
                                   WITHOUT the inductor compile (fast boot + duty fix).
                             NOTE: pass COMPACT JSON (no spaces) — docker_args is a single
                             space-joined string. Verify the field name against the pinned
                             v0.20.2 image before relying on it.

    Both unset => vLLM V1 default (FULL_AND_PIECEWISE): full piecewise compile + graphs,
    highest throughput but slowest-boot / highest-OOM. Mitigate boot OOM with a modest
    --max-num-seqs (fewer graph sizes to capture) + gpu-mem-util headroom.
    """
    out: list[str] = []
    if getattr(args, "enforce_eager", False):
        out.append("--enforce-eager")
    if getattr(args, "compilation_config", None):
        out += ["--compilation-config", args.compilation_config]
    return out


def _lora_args(args: argparse.Namespace) -> list[str]:
    """vLLM LoRA flags when serving a finetuned adapter on top of the AWQ base.

    vLLM applies LoRA on a quantized (AWQ) base WITHOUT merging; for AWQ it patches only the
    attention/MLP projections — exactly this adapter's target_modules (q/k/v/o/gate/up/down,
    no embeddings/lm_head). ``--lora-modules adapter=<hf_repo_id>`` lets vLLM resolve+download
    the adapter from the HF Hub at startup (HF_TOKEN injected for private repos). Served module
    name is "adapter" so the endpoint pool's single request-model name ("adapter") works across
    the H200 pod and the local box (which serves the same adapter from a local path).
    """
    if not getattr(args, "adapter_repo", None):
        return []
    return [
        "--enable-lora",
        "--lora-modules",
        f"adapter={args.adapter_repo}",
        "--max-lora-rank",
        str(args.max_lora_rank),
        "--max-loras",
        "1",
    ]


def cmd_up(args: argparse.Namespace) -> int:
    import runpod  # type: ignore[import-not-found]  # provided via `uv run --with runpod`

    runpod.api_key = _require_env("RUNPOD_API_KEY")
    vllm_key = _require_env("RUNPOD_VLLM_KEY")

    env = {"VLLM_API_KEY": vllm_key}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    if args.network_volume_id:
        # Persist the HF model cache on the network volume so future pods skip the
        # ~18 GB cold download (binds in ~2-3 min instead of ~20+). The volume is
        # region-locked: it must live in a storage DC (EU-RO-1 / EUR-IS-1) and the
        # pod is auto-placed there by the SDK. See runpod-network-volume-region-lock.
        env["HF_HOME"] = f"{args.volume_mount_path.rstrip('/')}/hf"

    docker_args = _docker_args(args)
    print(f"Creating pod '{args.name}' on {args.gpu_type} ({args.cloud})...")
    print(f"  image      : {args.image}")
    print(f"  vllm args  : {docker_args}")
    print(f"  ports      : {args.ports}  (8000 = TCP, public)")
    print(f"  cuda guard : {args.allowed_cuda_versions}  (host driver must support these)")
    try:
        pod = runpod.create_pod(
            name=args.name,
            image_name=args.image,
            gpu_type_id=args.gpu_type,
            cloud_type=args.cloud,
            allowed_cuda_versions=args.allowed_cuda_versions or None,  # None on AMD (empty list)
            gpu_count=1,
            container_disk_in_gb=args.container_disk,
            min_vcpu_count=8,
            min_memory_in_gb=32,
            ports=args.ports,
            docker_args=docker_args,
            env=env,
            support_public_ip=True,
            start_ssh=True,
            network_volume_id=args.network_volume_id or None,
            volume_mount_path=args.volume_mount_path,
        )
    except Exception as exc:  # noqa: BLE001 — surface RunPod stock/quota errors plainly
        print(f"ERROR: create_pod failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"  ({args.gpu_type} may be out of stock in {args.cloud} — retry or try --cloud COMMUNITY)", file=sys.stderr)
        return 1

    pod_id = pod.get("id")
    # The estimate is GPU-dependent; `status {pod_id}` prints the real costPerHr from RunPod.
    print(f"\n  POD ID: {pod_id}   (billing now — `status {pod_id}` shows the real $/hr; "
          f"`down {pod_id}` stops billing)\n")

    # Phase 1: wait for the pod to be RUNNING with a public TCP mapping for 8000.
    ip = port = None
    deadline = time.time() + args.running_timeout
    while time.time() < deadline:
        full = runpod.get_pod(pod_id)
        pod_obj = full.get("pod", full) if isinstance(full, dict) else {}
        ip, port = _public_tcp(pod_obj)
        if ip and port:
            break
        print(f"  ...waiting for public port (status={pod_obj.get('desiredStatus')})")
        time.sleep(10)
    if not (ip and port):
        print("ERROR: pod did not expose a public TCP port for 8000 in time.", file=sys.stderr)
        print(f"  Inspect with: scripts/provision_runpod_vllm.py status {pod_id}", file=sys.stderr)
        return 1

    base_url = f"http://{ip}:{port}"
    print(f"  public endpoint: {base_url}  (TCP)")

    # Phase 2: wait for vLLM to finish loading (~18 GB AWQ download + warmup).
    models_url = f"{base_url}/v1/models"
    headers = _bearer(vllm_key)
    print("  ...waiting for vLLM /v1/models (model download + warmup, can take several minutes)")
    deadline = time.time() + args.ready_timeout
    ready = False
    while time.time() < deadline:
        if _http_ok(models_url, headers):
            ready = True
            break
        time.sleep(15)
    if not ready:
        print(f"WARNING: {models_url} not ready yet; the pod is up — re-check shortly with `status {pod_id}`.", file=sys.stderr)

    state = "READY" if ready else "STARTING"
    print(f"\n=== {state} ===")
    print(f"pod_id   : {pod_id}")
    print(f"base_url : {base_url}")
    print("\nPaste into the run config (key stays in env via api_key_env):\n")
    print("[[vllm.endpoints]]")
    print(f'id          = "{args.name}"')
    print(f'base_url    = "{base_url}"')
    print(f"capacity    = {args.capacity}")
    print('api_key_env = "RUNPOD_VLLM_KEY"')
    print(f"\nBake-off metrics:\n  uv run python scripts/inspect/vllm_live_metrics.py "
          f"--url {base_url}/metrics --window 60 --count 10 --jsonl /tmp/bakeoff_l40s.jsonl")
    print(f"\nTear down when done:\n  uv run --with runpod python scripts/provision_runpod_vllm.py down {pod_id}")
    return 0 if ready else 0


def cmd_status(args: argparse.Namespace) -> int:
    import runpod  # type: ignore[import-not-found]  # provided via `uv run --with runpod`

    runpod.api_key = _require_env("RUNPOD_API_KEY")
    full = runpod.get_pod(args.pod_id)
    pod_obj = full.get("pod", full) if isinstance(full, dict) else {}
    if not pod_obj:
        print(f"No pod found for id {args.pod_id}", file=sys.stderr)
        return 1
    ip, port = _public_tcp(pod_obj)
    print(f"pod_id        : {args.pod_id}")
    print(f"desiredStatus : {pod_obj.get('desiredStatus')}")
    print(f"uptime (s)    : {pod_obj.get('uptimeSeconds')}")
    print(f"cost/hr       : {pod_obj.get('costPerHr')}")  # actual RunPod rate, not the estimate
    print(f"public 8000   : {f'http://{ip}:{port}' if ip and port else '(not exposed yet)'}")
    vllm_key = os.environ.get("RUNPOD_VLLM_KEY")
    if ip and port:
        ok = _http_ok(f"http://{ip}:{port}/v1/models", _bearer(vllm_key))
        print(f"/v1/models    : {'OK (vLLM ready)' if ok else 'not ready'}")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    import runpod  # type: ignore[import-not-found]  # provided via `uv run --with runpod`

    runpod.api_key = _require_env("RUNPOD_API_KEY")
    runpod.terminate_pod(args.pod_id)
    print(f"Terminated pod {args.pod_id} (billing stopped).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="provision an L40S vLLM pod")
    up.add_argument("--name", default="nsl-vllm-l40s")
    up.add_argument("--image", default=PINNED_VLLM_IMAGE)
    up.add_argument("--model", default=DEFAULT_MODEL)
    up.add_argument("--gpu-type", default=GPU_TYPE_L40S)
    up.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY", "ALL"],
                    help="SECURE (~$0.86/hr, stable IP) vs COMMUNITY (~$0.79/hr, IP may change on restart)")
    up.add_argument("--allowed-cuda-versions", nargs="*", default=["13.0"],
                    help="NVIDIA-only guard: only land on hosts whose driver supports these CUDA "
                         "versions. The pinned v0.20.2 image ships CUDA 13 / PyTorch 2.11, which "
                         "hard-crashes at EngineCore init on an older driver ('NVIDIA driver too "
                         "old (found version 12040)' = CUDA 12.4). Community hosts are a driver "
                         "lottery, so default to ['13.0']. Widen only if you pin an older image. "
                         "On an AMD/ROCm pod (MI300X) pass it with NO values "
                         "(--allowed-cuda-versions) to DROP the guard — CUDA is meaningless there.")
    up.add_argument("--capacity", type=int, default=16, help="endpoint pool capacity to print in the TOML block")
    up.add_argument("--max-num-seqs", type=int, default=18, help="vLLM concurrent seqs (>= capacity + slack)")
    up.add_argument("--max-model-len", type=int, default=65536)
    up.add_argument("--max-num-batched-tokens", type=int, default=8192,
                    help=">= 2496 (gemma-4 multimodal floor); NOT the local 4096 decode hack")
    up.add_argument("--kv-cache-dtype", default="fp8",
                    help="fp8(=e4m3) needs Ada/Hopper sm_89+ (L40S/4090). On Ampere sm_86 (A40) "
                         "use 'auto' (fp16 KV): e4m3 fails the arch check AND fp8_e5m2 trips an "
                         "attention-backend assert (allows only fp8/e4m3/nvfp4) in v0.20.2.")
    up.add_argument("--serve-cmd", default=None,
                    help="Command PREFIX prepended to the vLLM flags, for images WITHOUT an "
                         "api_server ENTRYPOINT. NVIDIA vllm/vllm-openai needs none (entrypoint runs "
                         "the server). ROCm nightly (rocm/vllm-dev:nightly_*) has CMD /bin/bash, so "
                         "pass --serve-cmd 'python3 -m vllm.entrypoints.openai.api_server'.")
    up.add_argument("--gpu-mem-util", type=float, default=0.92)
    up.add_argument("--enforce-eager", action="store_true",
                    help="OPT-IN: disable torch.compile + cudagraphs. Default is GRAPHS-ON "
                         "(omit this flag). Eager penalty is the decode slice only (~10-25%%, "
                         "NOT 20x — the workload is prefill-dominated); a safe default on "
                         "cudagraph-risky stacks (AWQ GH#32834, ROCm). See the 2026-06-10 doc.")
    up.add_argument("--compilation-config", default=None,
                    help="Raw COMPACT JSON passthrough to vLLM --compilation-config (no spaces; "
                         "docker_args is space-joined). Fallback ladder for the awq_marlin "
                         "full-graph crash GH#32834: '{\"cudagraph_mode\":\"PIECEWISE\"}' (attn "
                         "eager, dodges the crash) or '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' "
                         "(full decode graphs without the inductor compile). Verify vs v0.20.2.")
    up.add_argument("--adapter-repo", default=None,
                    help="HF repo id of a LoRA adapter to serve on top of the base model "
                         "(e.g. 0xEljh/recency-c-sft-r64-1ep-20260607). Adds --enable-lora "
                         "--lora-modules adapter=<repo>; served module name is 'adapter'. "
                         "vLLM resolves the repo from HF at startup (HF_TOKEN handles private).")
    up.add_argument("--max-lora-rank", type=int, default=64, help="must be >= the adapter's r (r64)")
    up.add_argument("--container-disk", type=int, default=60,
                    help="GB; holds image + ~18 GB AWQ weights (+ ~2 GB LoRA — use 80 with --adapter-repo)")
    up.add_argument("--ports", default="8000/tcp,22/tcp")
    up.add_argument("--network-volume-id", default=None,
                    help="attach a RunPod network volume to persist the HF model cache "
                         "(sets HF_HOME=<mount>/hf; pod auto-placed in the volume's DC). "
                         "Volume must live in a storage DC: EU-RO-1 or EUR-IS-1.")
    up.add_argument("--volume-mount-path", default="/runpod-volume",
                    help="mount path for --network-volume-id (HF_HOME becomes <path>/hf)")
    up.add_argument("--running-timeout", type=float, default=420.0, help="seconds to wait for a public port")
    up.add_argument("--ready-timeout", type=float, default=1200.0, help="seconds to wait for /v1/models")
    up.set_defaults(func=cmd_up)

    st = sub.add_parser("status", help="show a pod's status + readiness")
    st.add_argument("pod_id")
    st.set_defaults(func=cmd_status)

    dn = sub.add_parser("down", help="terminate a pod (stops billing)")
    dn.add_argument("pod_id")
    dn.set_defaults(func=cmd_down)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
