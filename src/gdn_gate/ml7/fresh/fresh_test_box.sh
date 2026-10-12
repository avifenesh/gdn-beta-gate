#!/usr/bin/env bash
# fresh_test_box.sh [plan]: the Phase 5 fresh held-out set (AIME 2024/2025/2026, AMC 2023; pre-registration in
# ml_pipeline_4b/PIPELINE.md, "Fresh test set") on the rented box. The step14_test_box.sh pattern: the box scheduler
# (box/orch.py) runs this as a CPU job; PHASE=readouts queues one PHASE=one job per readout (orch.py submit, 2 readouts
# per GPU at gpu_mem 0.45) and waits; fresh_guard.py final wraps readouts + analyses in ONE guard.final_test() of the
# fresh set's own guard state (ml_pipeline_4b/fresh_test, touch 1).
# Readouts (readout_vllm.py, unchanged; code locked by sha256 at the first readout): caps 256..8192, samples 8, seed
# 20261006, gpu_mem 0.45, --probe_effect and --answer_max per record (4b: 0.05 / 1280; 2b: 0.2 / 1792), ctrlpm --arm
# ctrl --rank 20. Checkpoints from fresh/ckpt_expect.json (paths under $WG_RUNS, written <runs>/... in the file).
# Env: FRESH_RECORDS ("4b 2b"; "4b" drops the 2b block, decided before the read), FRESH_M4, FRESH_M2 (model dirs),
#   FRESH_DATA, FRESH_RES (results/fresh), FRESH_ML (ml_pipeline_4b), FRESH_GD (FRESH_ML/fresh_test),
#   FRESH_PRIO_4B (30), FRESH_PRIO_2B (55), FRESH_JOB_PREFIX (fr_), GPU_MEM (0.45), TTL (20h).
#   bash fresh_test_box.sh plan    # checks + plan (exit 0 = launchable)
#   bash fresh_test_box.sh         # what the scheduler runs
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SELF=$HERE/$(basename "$0")
ML7=$(cd "$HERE/.." && pwd)
NEXT=$ML7/..
WT=$(cd "$HERE/../../../.." && pwd)
VPY=${FRESH_VPY:-$HOME/.cache/gdn_gate/vllm/venv/bin/python}
ORCH=${FRESH_ORCH:-$NEXT/box/orch.py}
M4=${FRESH_M4:-${WG_HF_CACHE:-$HOME/.cache/huggingface/hub}/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
M2=${FRESH_M2:-${WG_HF_CACHE:-$HOME/.cache/huggingface/hub}/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc}
DATA=${FRESH_DATA:-${WG_RUNS:-$WT/runs}/ml7/data}
RES=${FRESH_RES:-$WT/results/fresh}
ML=${FRESH_ML:-$WT/ml_pipeline_4b}
GD=${FRESH_GD:-$ML/fresh_test}
EXPECT=${FRESH_EXPECT:-$HERE/ckpt_expect.json}
S=${FRESH_SCRATCH:-$HOME/.cache/gdn_gate/ml7/fresh_box}
PYLIB=${FRESH_PYLIB:-$HOME/.cache/gdn_gate/ml7/pylib}
GPU_MEM=${GPU_MEM:-0.45}
RECORDS=${FRESH_RECORDS:-4b 2b}
CAPS=256,512,1024,2048,4096,8192; SAMPLES=8; RSEED=20261006
CODE="$ML7/readout_vllm.py $ML7/train_grpo_vllm.py $NEXT/vllm/gate_vllm.py $NEXT/broad/rl_common.py
      $NEXT/broad/arms.py $NEXT/broad/packing.py $ML7/step14_analysis.py $ML7/ctrlpm_analysis.py $WT/ml_pipeline/guard.py
      $HERE/fresh_analysis.py $HERE/fresh_guard.py $HERE/build_fresh.py $HERE/ckpt_expect.json"

labels_of() {
  case $1 in
    4b) echo "4b_untouched 4b_warmup"; for a in ctrl wide wide1 ctrlpm; do for s in 0 1 2; do echo "4b_${a}_$s"; done; done;;
    2b) echo "2b_untouched"; for a in ctrl wide wide1; do for s in 0 1 2 3 4 5; do echo "2b_${a}_$s"; done; done;;
  esac
}
LABELS=$(for r in $RECORDS; do labels_of "$r"; done | tr '\n' ' '); LABELS=${LABELS% }
rec_of() { echo "${1%%_*}"; }
arm_of() { local x=${1#*_}; case $x in untouched) echo stock;; warmup|ctrlpm_*) echo ctrl;; *) echo "${x%_*}";; esac; }
ckpt_of() {
  case $1 in *_untouched) echo none; return;; esac
  "$VPY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["path"].replace("<runs>", sys.argv[3]))' "$EXPECT" "$1" "${WG_RUNS:-$WT/runs}"
}
model_of() { [ "$(rec_of "$1")" = 4b ] && echo "$M4" || echo "$M2"; }
amax_of() { [ "$(rec_of "$1")" = 4b ] && echo 1280 || echo 1792; }
probe_of() { [ "$(rec_of "$1")" = 4b ] && echo 0.05 || echo 0.2; }
rank_of() { case $1 in *_ctrlpm_*) echo 20;; *) echo 16;; esac; }
out_of() { echo "$RES/readout_fresh_$1.json"; }
mark_of() { echo "$RES/done/$1.sha256"; }
done_ok() { local m; m=$(mark_of "$1"); [ -f "$m" ] && [ -f "$(out_of "$1")" ] && sha256sum -c --status "$m" 2>/dev/null; }
write_plan() {
  local l
  for l in $LABELS; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$l" "$(rec_of "$l")" "$(arm_of "$l")" "$(ckpt_of "$l")" "$(out_of "$l")" \
      "$(model_of "$l")" "$(amax_of "$l")" "$(probe_of "$l")" "$(rank_of "$l")"
  done
}
readout_argv() {
  local l=$1 ck; ck=$(ckpt_of "$l")
  ARGV=("$VPY" "$ML7/readout_vllm.py" --model "$(model_of "$l")" --data "$DATA/test_fresh.jsonl" --arm "$(arm_of "$l")")
  [ "$ck" = none ] || ARGV+=(--ckpt "$ck")
  ARGV+=(--caps "$CAPS" --answer_max "$(amax_of "$l")" --samples "$SAMPLES" --seed "$RSEED" --gpu_mem "$GPU_MEM"
         --probe_effect "$(probe_of "$l")" --rank "$(rank_of "$l")" --out "$(out_of "$l")")
}
env_prefix() { local v; for v in ${!FRESH_@}; do [ "$v" = FRESH_UNDER_GUARD ] || printf '%s=%q ' "$v" "${!v}"; done; }
log() { echo "$*" | tee -a "$RES/steps.log"; }
# order: longest first (4B ctrl-type, then 4B gated, then 2B); cost 2 (one readout per GPU: on this box a second
# readout cannot start beside a running one at gpu_mem 0.45, Phase 5 calibration)
order_of() {
  local w
  case $1 in 4b_untouched|4b_warmup|4b_ctrl_*|4b_ctrlpm_*) w=4;; 4b_*) w=3;; 2b_untouched|2b_ctrl_*) w=2;; *) w=1;; esac
  echo "-$w"
}
prio_of() { [ "$(rec_of "$1")" = 4b ] && echo "${FRESH_PRIO_4B:-30}" || echo "${FRESH_PRIO_2B:-55}"; }

if [ -n "${PHASE:-}" ]; then
  [ "${FRESH_UNDER_GUARD:-}" = 1 ] || { echo "PHASE=$PHASE runs only under fresh_guard.py final" >&2; exit 2; }
  mkdir -p "$RES/logs" "$RES/done"
  case "$PHASE" in
    one)
      l=${LABEL:?LABEL}
      [ -f "$RES/code.sha256" ] || { log "REFUSED: $l: no code lock"; exit 2; }
      sha256sum -c --quiet "$RES/code.sha256" || { log "REFUSED: instrument code changed since the first readout"; exit 4; }
      if done_ok "$l"; then log "=== $l kept"; exit 0; fi
      rm -f "$(mark_of "$l")"
      # two readouts starting on one GPU at once race for memory (vLLM: "max_num_seqs exceeds available Mamba cache
      # blocks", seen in the Phase 5 calibration): a per-GPU start lock held FRESH_STAGGER s after each start
      exec 8> "/tmp/wg_gpu${CUDA_VISIBLE_DEVICES:-x}.start.lock"; flock 8; ( sleep "${FRESH_STAGGER:-150}"; flock -u 8 ) &
      G=$S/gen_$l; rm -rf "$G"; mkdir -p "$G"
      export VLLM_ENABLE_V1_MULTIPROCESSING=0 WG_CACHE_DIR="$G" PYTHONUNBUFFERED=1 WG_ML7_PYLIB="$PYLIB" \
             PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
      export PATH="$PATH:$(dirname "$VPY")"
      readout_argv "$l"
      log "=== $l start $(date -Is) gpu ${CUDA_VISIBLE_DEVICES:-?}: ${ARGV[*]}"
      t0=$(date +%s)
      "${ARGV[@]}" > "$RES/logs/$l.log" 2>&1 < /dev/null; rc=$?
      out=$(out_of "$l")
      if [ $rc = 0 ] && "$VPY" -c 'import json, sys
d = json.load(open(sys.argv[1])); n = sum(1 for x in open(sys.argv[2]) if x.strip())
assert len(d["rows"]) == n * int(sys.argv[3]) and d["args"]["answer_max"] == int(sys.argv[4]) and not d["red"]' \
          "$out" "$DATA/test_fresh.jsonl" "$SAMPLES" "$(amax_of "$l")" >> "$RES/logs/$l.log" 2>&1; then
        sha256sum "$out" > "$(mark_of "$l")"
        log "=== $l ok wall=$(( $(date +%s) - t0 ))s gpu ${CUDA_VISIBLE_DEVICES:-?}"
        rm -rf "$G"; exit 0
      fi
      log "=== $l FAIL rc=$rc wall=$(( $(date +%s) - t0 ))s"; exit 1;;
    readouts)
      if [ -f "$RES/code.sha256" ]; then
        sha256sum -c --quiet "$RES/code.sha256" || { log "REFUSED: instrument code changed since the first readout"; exit 4; }
      else
        # shellcheck disable=SC2086
        sha256sum $CODE > "$RES/code.sha256"
      fi
      names=()
      for l in $LABELS; do
        if done_ok "$l"; then log "=== $l kept"; continue; fi
        n=${FRESH_JOB_PREFIX:-fr_}$l
        /usr/bin/python3 "$ORCH" submit --name "$n" --cost 2 --prio "$(prio_of "$l")" --order "$(order_of "$l")" \
          --timeout 28800 --max_failures 4 --fresh -- \
          "$(env_prefix)PHASE=one FRESH_UNDER_GUARD=1 LABEL=$l GPU_MEM=$GPU_MEM bash $SELF" >> "$RES/steps.log" 2>&1 \
          || { log "=== $l: submit failed"; exit 1; }
        names+=("$n")
      done
      if [ ${#names[@]} -gt 0 ]; then
        log "=== readouts queued $(date -Is): ${names[*]}"
        /usr/bin/python3 "$ORCH" wait "${names[@]}" >> "$RES/steps.log" 2>&1
      fi
      fail=0
      for l in $LABELS; do done_ok "$l" || { log "=== $l not done"; fail=1; }; done
      exit $fail;;
    analysis)
      fail=0
      for r in $RECORDS; do
        rm -f "$RES/analysis_fresh_$r.json"
        slots=(); W=()
        for l in $(labels_of "$r"); do
          case $l in *_untouched) ;; *_warmup) W=(--warmup "$(out_of "$l")");; *) slots+=("${l#*_}=$(out_of "$l")");; esac
        done
        log "=== analysis $r start $(date -Is)"
        "$VPY" "$HERE/fresh_analysis.py" --record "$r" --baseline "$(out_of "${r}_untouched")" "${W[@]}" "${slots[@]}" \
          --manifest "$GD/split_manifest.json" --out "$RES/analysis_fresh_$r.json" > "$RES/logs/analysis_$r.log" 2>&1 \
          && log "=== analysis $r ok" || { log "=== analysis $r FAIL (log $RES/logs/analysis_$r.log)"; fail=1; }
      done
      exit $fail;;
    *) echo "unknown PHASE $PHASE" >&2; exit 2;;
  esac
fi

if [ "${INNER:-0}" = 1 ]; then
  mkdir -p "$RES/logs" "$RES/done" "$S"
  { echo "--- inner start $(date -Is) GPU_MEM=$GPU_MEM host $(hostname) records $RECORDS"
    nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv,noheader 2>&1
    # shellcheck disable=SC2086
    sha256sum $CODE "$SELF" 2>&1; } >> "$RES/env.txt"
  cp "$S/plan.tsv" "$RES/plan.tsv"
  exec "$VPY" "$HERE/fresh_guard.py" final --gd "$GD" --md "$ML/PIPELINE.md" --data "$DATA" --plan "$RES/plan.tsv" \
    --samples "$SAMPLES" --runner "$SELF" --res "$RES" --out "$RES/final.json"
fi

MODE=${1:-launch}
case "$MODE" in plan|launch) ;; *) echo "usage: fresh_test_box.sh [plan]" >&2; exit 2;; esac
mkdir -p "$S" "$RES"
write_plan > "$S/plan.tsv"
rc=0
[ -f "$M4/config.json" ] || { echo "REFUSED: no 4B model $M4" >&2; rc=2; }
case " $RECORDS " in *" 2b "*) [ -f "$M2/config.json" ] || { echo "REFUSED: no 2B model $M2" >&2; rc=2; };; esac
[ -d "$PYLIB/fla" ] || { echo "REFUSED: no fla in $PYLIB" >&2; rc=2; }
[ -f "$ORCH" ] || { echo "REFUSED: no scheduler $ORCH" >&2; rc=2; }
"$VPY" "$HERE/fresh_guard.py" preflight --gd "$GD" --md "$ML/PIPELINE.md" --data "$DATA" --plan "$S/plan.tsv" --expect "$EXPECT" || rc=2
CUDA_VISIBLE_DEVICES="" "$VPY" "$ML7/preflight_grader.py" "$NEXT/broad" || { echo "REFUSED: grader preflight" >&2; rc=2; }
echo "plan: $(echo $LABELS | wc -w) readouts of test_fresh.jsonl (records $RECORDS), samples $SAMPLES, caps $CAPS"
for l in $LABELS; do st=todo; done_ok "$l" && st=done; echo "  $l $st arm=$(arm_of "$l") ckpt=$(ckpt_of "$l")"; done
if [ "$MODE" = plan ]; then [ $rc = 0 ] && echo "plan: launchable" || echo "plan: refused"; exit $rc; fi
[ $rc = 0 ] || exit $rc
rm -f "$RES/FAIL"
INNER=1 timeout --kill-after=120 "${TTL:-20h}" bash "$SELF"
rc=$?
if [ $rc = 0 ] && [ -f "$RES/final.json" ]; then date -Is > "$RES/DONE"
else echo "rc=$rc $(date -Is)" > "$RES/FAIL"; fi
exit $rc
