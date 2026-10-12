#!/usr/bin/env bash
# run_grpo.sh ARM SEED CAP TAG [extra train_grpo_vllm.py args]: one step 7 GRPO run (train_grpo_vllm.py, colocated vLLM
# rollouts + PyTorch training) on ONE GPU. Set CUDA_VISIBLE_DEVICES to that GPU (in the lab a per-GPU job scheduler did
# this; it is not part of the release). Resumable: every run passes --resume, so rerunning the same command continues
# from the newest checkpoint in the run dir (checkpoints every 25 steps and at --stop_after).
#   ARM  wide | wide1 | ctrl | ctrlpm      SEED  integer      CAP  thinking cap (no default)
#        ctrlpm (Qwen3.5-4B record, parameter-matched control, owner approval 2026-10-11 ~23:55 Israel): the trainer runs
#        --arm ctrl (no gate) with the LoRA rank and the widened warm-up state given in the extra args (--rank 20
#        --init_from <pad_rank.py output>, both required); only the run dir carries the name ctrlpm
#   TAG  run-name tag: the run dir is GRPO/{ARM}_s{SEED}_cap{CAP}_{TAG} (fmt_box = format warm-up, r_box = arms and
#        cap pilots branched from it; the 2B extension uses r_box beside the GB10 record's r_gb10 runs)
# Extra args must include --length_on (the trainer requires it) and carry the recipe (--alpha, --steps, --stop_after,
# --init_from, --answer_max); the recipe values are the record's, written in its PIPELINE.md, never defaults here.
# Env: GRPO_MODEL (model dir, required), GRPO_MANIFEST (the record's split_manifest.json, required), GRPO_ROOT (run
# root, default $WG_RUNS/ml7/grpo; WG_RUNS defaults to <repo>/runs),
# GPU_MEM (vLLM gpu_memory_utilization, default 0.30 = the GB10 setting), MICRO (training micro-row tokens, default
# 16384 = the GB10 2B setting; the 4B needs a smaller row, set from the calibration's memory probe), GRPO_VPY, GRPO_PYLIB.
# micro_tokens and gpu_mem change memory and speed only: the loss is a token sum over the batch divided by the
# batch's sampled-token count, so the micro-row split does not change the update (G3 in test_grpo_vllm_cpu.py).
# PARITY_RETRY_K (default empty = off; set only with the owner's approval, as for the 2B wide s1 run on 2026-10-10):
# when the startup engine/trainer parity check fails on the decode pair ALONE (every prefill, effect and red-arm check
# and the start check pass), the failed meta.json is kept as meta_parity_fail.json and the run starts once more with
# --parity_k PARITY_RETRY_K; parity_retry.json records both. Any other parity failure stays a failure.
set -u
[ $# -ge 4 ] || { echo "usage: run_grpo.sh ARM SEED CAP TAG [extra args]" >&2; exit 2; }
ARM=$1; SEED=$2; CAP=$3; TAG=$4; shift 4
case "$ARM" in wide|wide1|ctrl|ctrlpm) ;; *) echo "ARM must be wide, wide1, ctrl or ctrlpm" >&2; exit 2;; esac
TRAIN_ARM=$ARM; [ "$ARM" = ctrlpm ] && TRAIN_ARM=ctrl
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be an integer" >&2; exit 2; }
[[ "$CAP" =~ ^[0-9]+$ ]] || { echo "CAP must be an integer" >&2; exit 2; }
[[ "$TAG" =~ ^[a-z0-9_]+$ ]] || { echo "TAG must be [a-z0-9_]+" >&2; exit 2; }
EXTRA="$*"
[[ " $EXTRA " =~ [[:space:]]--length_on([[:space:]]+|=)(completion|reasoning)[[:space:]] ]] || {
  echo "extra args must include --length_on completion|reasoning" >&2; exit 2; }
for k in --alpha --answer_max --steps; do
  [[ " $EXTRA " =~ [[:space:]]$k[[:space:]=] ]] || { echo "extra args must set $k (the record's recipe, no default here)" >&2; exit 2; }
done
if [ "$ARM" = ctrlpm ]; then
  for k in --rank --init_from; do
    [[ " $EXTRA " =~ [[:space:]]$k[[:space:]=] ]] || { echo "ctrlpm: extra args must set $k (rank 20, the widened warm-up)" >&2; exit 2; }
  done
fi
HERE=$(cd "$(dirname "$0")" && pwd)
M=${GRPO_MODEL:?GRPO_MODEL: the model dir}
MAN=${GRPO_MANIFEST:?GRPO_MANIFEST: the record split_manifest.json}
VPY=${GRPO_VPY:-$HOME/.cache/gdn_gate/vllm/venv/bin/python}
PYLIB=${GRPO_PYLIB:-$HOME/.cache/gdn_gate/ml7/pylib}
RUNS=${WG_RUNS:-$(cd "$HERE/../../.." && pwd)/runs}
DATA=${GRPO_DATA:-$RUNS/ml7/data}
ROOT=${GRPO_ROOT:-$RUNS/ml7/grpo}
GPU_MEM=${GPU_MEM:-0.30}; MICRO=${MICRO:-16384}
OUT=$ROOT/${ARM}_s${SEED}_cap${CAP}_${TAG}
bad=()
[ -f "$M/config.json" ] || bad+=("no model at $M")
[ -f "$MAN" ] || bad+=("no split manifest $MAN")
[ -d "$PYLIB/fla" ] || bad+=("no fla in $PYLIB (the trainer refuses to train on GPU without it)")
[ -x "$VPY" ] || bad+=("no vLLM venv python $VPY")
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || bad+=("CUDA_VISIBLE_DEVICES unset: run under the box scheduler (one GPU per run)")
if [ ${#bad[@]} -gt 0 ]; then echo "REFUSED:" >&2; printf '  %s\n' "${bad[@]}" >&2; exit 2; fi
[ -f "$OUT/DONE" ] && { echo "grpo_box: $OUT already DONE"; exit 0; }
mkdir -p "$ROOT" "$OUT"
CUDA_VISIBLE_DEVICES= "$VPY" "$HERE/preflight_grader.py" "$HERE/../broad" || { echo "REFUSED: grader preflight" >&2; exit 2; }
G=${GRPO_GEN:-$HOME/.cache/gdn_gate/ml7}/gen_grpo_$(basename "$OUT")
mkdir -p "$G"
echo "$(date -Is) gpu $CUDA_VISIBLE_DEVICES host $(hostname) GPU_MEM=$GPU_MEM MICRO=$MICRO model $M" >> "$OUT/hardware.txt"
export WG_ML7_PYLIB=$PYLIB WG_CACHE_DIR=$G PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PATH="$PATH:$(dirname "$VPY")"  # FlashInfer JIT spawns ninja by name (as the box readouts)
train() {
  # shellcheck disable=SC2086
  "$VPY" "$HERE/train_grpo_vllm.py" --model "$M" --pool "$DATA/rl_pool.jsonl" --val_data "$DATA/val.jsonl" \
    --manifest "$MAN" --arm "$TRAIN_ARM" --seed "$SEED" --cap "$CAP" --gpu_mem "$GPU_MEM" --micro_tokens "$MICRO" \
    --out "$OUT" --resume $EXTRA "$@" >> "$OUT.log" 2>&1
}
train; rc=$?
if [ $rc != 0 ] && [ -n "${PARITY_RETRY_K:-}" ] && [ ! -f "$OUT/train_log.jsonl" ] && [ -f "$OUT/meta.json" ] \
   && [ ! -f "$OUT/parity_retry.json" ]; then
  if "$VPY" - "$OUT/meta.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))["parity"]
others = {k: v for k, v in p["checks"].items() if k != "probe_decode"}
sys.exit(0 if (not p["checks"]["probe_decode"] and all(others.values()) and p["start_pass"]) else 1)
PY
  then
    cp -p "$OUT/meta.json" "$OUT/meta_parity_fail.json"
    printf '{"retry_k": %s, "failed_meta": "meta_parity_fail.json", "when": "%s", "reason": "decode pair alone over 2 x floor; every prefill, effect, red-arm and start check passed"}\n' \
      "$PARITY_RETRY_K" "$(date -Is)" > "$OUT/parity_retry.json"
    echo "$(date -Is) parity: decode pair alone failed at k 2; retry with --parity_k $PARITY_RETRY_K" >> "$OUT/hardware.txt"
    train --parity_k "$PARITY_RETRY_K"; rc=$?
  fi
fi
exit $rc
