#!/usr/bin/env bash
# Remote bootstrap for a gemma-4-31B qLoRA finetune on a RunPod RTX 5090 (Blackwell).
# Runs ON THE POD. Ship this + src/train/qlora.py + the SFT .jsonl, then:
#
#   bash runpod_train_bootstrap.sh install   # one-time: unsloth/bnb stack for sm_120
#   bash runpod_train_bootstrap.sh smoke      # verify torch/cuda/bnb + GPU capability
#   bash runpod_train_bootstrap.sh train      # run the finetune (params via env, see below)
#
# Blackwell notes (unslothai/unsloth#5154): dense gemma-4-31B fits ~18-22GB 4-bit on the
# 32GB 5090 (the MoE variant OOMs). torch 2.11.0+cu129 + bitsandbytes 0.49.2 is known-good;
# a cu130 torch breaks the bnb ABI. Gemma-4 hybrid attention has 512-dim heads → FA2 is
# rejected, SDPA required. We install on top of the image's torch (2.8+cu128) and only
# repin if the smoke check shows a broken CUDA/bnb combo.
set -euo pipefail

WORKDIR="${WORKDIR:-/root/train}"
QLORA="${QLORA:-$WORKDIR/qlora.py}"
PYBIN="${PYBIN:-python}"

phase() { echo; echo "======== $* ========"; }

do_install() {
  phase "install: unsloth qLoRA stack (arch ${TORCH_CUDA_ARCH_LIST:-12.0})"
  # Default to Blackwell sm_120; override (e.g. TORCH_CUDA_ARCH_LIST=8.9) for Ada cards
  # so triton/unsloth JIT kernels target the actual device.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
  export DEBIAN_FRONTEND=noninteractive
  $PYBIN -m pip install --upgrade pip
  # Let unsloth resolve its own matched torch/transformers/peft/trl/bnb/triton set.
  # Pin bitsandbytes to the cu12-compatible 0.49.2 so a stray cu130 torch can't strand it.
  $PYBIN -m pip install "unsloth" "bitsandbytes==0.49.2" "triton>=3.3.1"
  # qlora.py's own non-ML deps.
  $PYBIN -m pip install python-dotenv loguru wandb hf_transfer
  phase "install done"
}

do_smoke() {
  phase "smoke: torch / cuda / bnb / GPU"
  $PYBIN - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
import bitsandbytes as bnb
print("bitsandbytes", bnb.__version__)
import transformers, trl, peft
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)
import unsloth
print("unsloth", getattr(unsloth, "__version__", "?"))
PY
  phase "smoke done"
}

do_fetch() {
  # hf_transfer ("fast downloading") is fast (many parallel chunks) but intermittently
  # FREEZES at 0 B/s mid-shard on RunPod's CDN peering; the plain downloader is reliable
  # but throttled to ~0.45 MB/s (>10h for ~18GB). So: keep hf_transfer ON for speed and run
  # snapshot_download under a STALL WATCHDOG — kill the instant cache byte-growth stops for
  # STALL_TICKS*TICK_S seconds, then retry (resumes from the .incomplete blobs). Each retry
  # captures a fresh fast burst; a natural exit 0 means the snapshot is complete.
  phase "fetch: pre-download ${BASE_MODEL:-base model} (hf_transfer + stall watchdog)"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  cd "$WORKDIR"
  [ -f "$WORKDIR/.env" ] && set -a && . "$WORKDIR/.env" && set +a || true
  BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-31B-it-unsloth-bnb-4bit}"
  local cache="${HF_HOME:-$HOME/.cache/huggingface}"
  local tick_s="${FETCH_TICK_S:-8}" stall_ticks="${FETCH_STALL_TICKS:-5}"
  local attempt=0 max_attempts="${FETCH_MAX_ATTEMPTS:-60}"
  while [ "$attempt" -lt "$max_attempts" ]; do
    attempt=$((attempt+1))
    $PYBIN -c "from huggingface_hub import snapshot_download; snapshot_download('$BASE_MODEL')" > /dev/null 2>&1 &
    local dlpid=$! killed=0 stall=0 last=-1 cur
    while kill -0 "$dlpid" 2>/dev/null; do
      cur=$(du -sb "$cache" 2>/dev/null | cut -f1)
      if [ "$cur" = "$last" ]; then stall=$((stall+1)); else stall=0; last="$cur"; fi
      if [ "$stall" -ge "$stall_ticks" ]; then
        echo "[fetch] attempt $attempt: stalled at $((cur/1024/1024))MB — killing, will resume"
        kill -9 "$dlpid" 2>/dev/null; killed=1; break
      fi
      sleep "$tick_s"
    done
    if [ "$killed" -eq 0 ]; then
      if wait "$dlpid"; then echo "[fetch] complete on attempt $attempt ($((last/1024/1024))MB)"; return 0; fi
    else
      wait "$dlpid" 2>/dev/null || true
    fi
  done
  echo "[fetch] FAILED after $max_attempts attempts" >&2; return 1
}

do_train() {
  phase "train: gemma-4-31B qLoRA (r=${LORA_RANK:-128})"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
  cd "$WORKDIR"   # so qlora's find_dotenv(usecwd) picks up ./.env (WANDB_API_KEY, HF_TOKEN)
  [ -f "$WORKDIR/.env" ] && set -a && . "$WORKDIR/.env" && set +a || true

  BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-31B-it-unsloth-bnb-4bit}"
  DATA="${DATA:-$WORKDIR/sft_training_rows.jsonl}"
  OUT="${OUT:-$WORKDIR/adapter}"
  # RESUME=1 picks up the latest checkpoint-* in $OUT (after a process/pod restart).
  RESUME_FLAG=""
  [ -n "${RESUME:-}" ] && RESUME_FLAG="--resume-from-checkpoint"
  # GROUP_BY_LENGTH=1 batches similar-length rows (use with BS>1 to cut padding waste).
  GROUP_FLAG=""
  [ -n "${GROUP_BY_LENGTH:-}" ] && GROUP_FLAG="--group-by-length"

  set -x
  $PYBIN "$QLORA" \
    --base-model "$BASE_MODEL" \
    --training-data "$DATA" \
    --output "$OUT" \
    --max-seq-length "${MAXSEQ:-4096}" \
    --max-steps -1 \
    --num-train-epochs "${EPOCHS:-4}" \
    --per-device-train-batch-size "${BS:-1}" \
    --gradient-accumulation-steps "${GRADACC:-16}" \
    --learning-rate "${LR:-5e-5}" \
    --warmup-ratio "${WARMUP_RATIO:-0.03}" \
    --lr-scheduler-type "${SCHED:-linear}" \
    --weight-decay "${WD:-0.001}" \
    --lora-rank "${LORA_RANK:-128}" \
    --lora-alpha "${LORA_ALPHA:-128}" \
    --lora-dropout "${LORA_DROPOUT:-0.04}" \
    --seed "${SEED:-3407}" \
    --rehearsal-dataset "${REHEARSAL_DS:-ClickNoow/5k-dataset-geogpt-fineweb}" \
    --rehearsal-split train \
    --rehearsal-text-field text \
    --rehearsal-rows-per-epoch "${REHEARSAL_ROWS:-500}" \
    --rehearsal-seed "${SEED:-3407}" \
    --wandb-project "${WANDB_PROJECT:-feature-hypothesis-kazakhstan-r128}" \
    --save-steps "${SAVE_STEPS:-0}" \
    $RESUME_FLAG \
    $GROUP_FLAG \
    --export-format lora
  set +x
  phase "train done -> $OUT"
}

case "${1:-}" in
  install) do_install ;;
  smoke)   do_smoke ;;
  fetch)   do_fetch ;;
  # pre-fetch the weights (watchdog) so the model loads from cache, then train.
  train)   do_fetch && do_train ;;
  *) echo "usage: $0 {install|smoke|fetch|train}" >&2; exit 2 ;;
esac
