#!/usr/bin/env python3
"""Provision / inspect / tear down a RunPod RTX 5090 pod for a qLoRA finetune.

Sibling of ``provision_runpod_vllm.py`` but for *training* rather than inference:
this brings up a bare CUDA-12.8 PyTorch pod with SSH exposed, and the caller then
ships ``src/train/qlora.py`` + the SFT dataset over SSH, installs the unsloth stack
(see ``scripts/runpod_train_bootstrap.sh``), and runs training. The lifecycle bits
that need the RunPod SDK (create / status / terminate) live here; the interactive
env-setup + training run are driven over plain ssh/scp so the (Blackwell-sensitive)
install can be watched and patched live.

Why a 5090 + CUDA 12.8: gemma-4-31B is the *dense* variant, which fits ~18-22GB at
4-bit on the 32GB 5090 (the MoE Gemma-4 is the one that OOMs). Blackwell (sm_120)
needs a CUDA>=12.8 base image; the unsloth stack is version-sensitive there
(unslothai/unsloth#5154: torch 2.11.0+cu129 + bitsandbytes 0.49.2 known-good;
cu130 torch breaks the bnb ABI). See memory runpod-5090-gemma4-qlora.

Usage (runpod SDK provided ephemerally by uv; secrets sourced out-of-repo):

    source ~/.config/nsl-runpod/secrets.env
    uv run --with runpod python scripts/provision_runpod_train.py up \
        --pubkey "$(cat ~/.ssh/nsl_runpod_train.pub)"
    uv run --with runpod python scripts/provision_runpod_train.py status <pod_id>
    uv run --with runpod python scripts/provision_runpod_train.py down <pod_id>

``up`` prints the public SSH host:port once the pod is reachable. Reads
RUNPOD_API_KEY from the environment; the SSH public key is injected via the
``PUBLIC_KEY`` env var (RunPod's official images append it to authorized_keys).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

# CUDA 12.8 = the Blackwell (sm_120) floor. torch 2.8 + cu128 in this image already
# has sm_120 cuBLAS kernels; `pip install unsloth` on top pulls the rest. See memory.
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
GPU_TYPE_5090 = "NVIDIA GeForce RTX 5090"  # exact gpuTypeId from RunPod get_gpus()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(
            f"ERROR: {name} is not set. Run `source ~/.config/nsl-runpod/secrets.env` first."
        )
    return value


def _public_tcp(pod: dict, private_port: int = 22) -> tuple[str | None, int | None]:
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


def _tcp_open(ip: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def cmd_up(args: argparse.Namespace) -> int:
    import runpod  # type: ignore[import-not-found]  # provided via `uv run --with runpod`

    runpod.api_key = _require_env("RUNPOD_API_KEY")

    pubkey = args.pubkey or os.environ.get("RUNPOD_TRAIN_PUBKEY")
    if not pubkey:
        sys.exit("ERROR: pass --pubkey '<contents of your .pub>' (RunPod injects it as PUBLIC_KEY).")

    env = {"PUBLIC_KEY": pubkey}
    # HF token helps dodge anon download throttling on the ~18GB 4-bit weights.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    print(f"Creating pod '{args.name}' on {args.gpu_type} ({args.cloud})...")
    print(f"  image          : {args.image}")
    print(f"  container disk : {args.container_disk} GB")
    print(f"  ports          : {args.ports}  (22 = TCP, public, SSH)")
    try:
        pod = runpod.create_pod(
            name=args.name,
            image_name=args.image,
            gpu_type_id=args.gpu_type,
            cloud_type=args.cloud,
            gpu_count=1,
            container_disk_in_gb=args.container_disk,
            min_vcpu_count=8,
            min_memory_in_gb=32,
            ports=args.ports,
            env=env,
            support_public_ip=True,
            start_ssh=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface RunPod stock/quota errors plainly
        print(f"ERROR: create_pod failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  (5090 may be out of stock in the selected cloud — retry or try --cloud SECURE/ALL)", file=sys.stderr)
        return 1

    pod_id = pod.get("id")
    print(f"\n  POD ID: {pod_id}   (billing live while running — `down {pod_id}` to stop)\n")

    # Phase 1: wait for a public TCP mapping for 22.
    ip = port = None
    deadline = time.time() + args.running_timeout
    while time.time() < deadline:
        full = runpod.get_pod(pod_id)
        pod_obj = full.get("pod", full) if isinstance(full, dict) else {}
        ip, port = _public_tcp(pod_obj)
        if ip and port:
            break
        print(f"  ...waiting for public SSH port (status={pod_obj.get('desiredStatus')})")
        time.sleep(10)
    if not (ip and port):
        print("ERROR: pod did not expose a public TCP port for 22 in time.", file=sys.stderr)
        print(f"  Inspect with: scripts/provision_runpod_train.py status {pod_id}", file=sys.stderr)
        return 1

    # Phase 2: wait for sshd to actually accept connections (image boot + start.sh).
    print(f"  public SSH: {ip}:{port}  ...waiting for sshd to accept")
    deadline = time.time() + args.ready_timeout
    ready = False
    while time.time() < deadline:
        if _tcp_open(ip, port):
            ready = True
            break
        time.sleep(10)

    state = "READY" if ready else "PORT-OPEN-BUT-SSH-NOT-UP"
    print(f"\n=== {state} ===")
    print(f"pod_id : {pod_id}")
    print(f"ssh    : ssh -i ~/.ssh/nsl_runpod_train -o StrictHostKeyChecking=no root@{ip} -p {port}")
    print(f"scp    : scp -i ~/.ssh/nsl_runpod_train -P {port} <local> root@{ip}:<remote>")
    print(f"\nTear down when done:\n  uv run --with runpod python scripts/provision_runpod_train.py down {pod_id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    import runpod  # type: ignore[import-not-found]

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
    print(f"cost/hr       : {pod_obj.get('costPerHr')}")
    if ip and port:
        up = _tcp_open(ip, port)
        print(f"public SSH    : {ip}:{port}  ({'accepting' if up else 'not accepting yet'})")
    else:
        print("public SSH    : (not exposed yet)")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    import runpod  # type: ignore[import-not-found]

    runpod.api_key = _require_env("RUNPOD_API_KEY")
    runpod.terminate_pod(args.pod_id)
    print(f"Terminated pod {args.pod_id} (billing stopped).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="provision a 5090 training pod with SSH exposed")
    up.add_argument("--name", default="nsl-train-5090")
    up.add_argument("--image", default=DEFAULT_IMAGE)
    up.add_argument("--gpu-type", default=GPU_TYPE_5090)
    up.add_argument("--cloud", default="COMMUNITY", choices=["SECURE", "COMMUNITY", "ALL"],
                    help="5090s are usually on COMMUNITY; SECURE has stabler IPs but less 5090 stock")
    up.add_argument("--pubkey", default=None, help="contents of your SSH public key (injected as PUBLIC_KEY)")
    up.add_argument("--container-disk", type=int, default=100,
                    help="GB; holds image + ~18GB 4-bit weights + ~4GB adapter + pip wheels")
    up.add_argument("--ports", default="22/tcp")
    up.add_argument("--running-timeout", type=float, default=420.0, help="seconds to wait for a public port")
    up.add_argument("--ready-timeout", type=float, default=600.0, help="seconds to wait for sshd to accept")
    up.set_defaults(func=cmd_up)

    st = sub.add_parser("status", help="show a pod's status + SSH readiness")
    st.add_argument("pod_id")
    st.set_defaults(func=cmd_status)

    dn = sub.add_parser("down", help="terminate a pod (stops billing)")
    dn.add_argument("pod_id")
    dn.set_defaults(func=cmd_down)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
