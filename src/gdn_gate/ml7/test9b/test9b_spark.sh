#!/usr/bin/env bash
# test9b_spark.sh [plan]: the 9B record's single read of AIME 2025 (30 problems x 8 samples; pre-registration in
# ml_pipeline_9b/PIPELINE.md, "AIME 2025 test read at 9B") on the two DGX Spark GB10 units, run from the workstation.
# The fresh_test_box.sh pattern with sparklock in place of the box scheduler:
#   * test9b_guard.py final wraps readouts + analysis in ONE guard.final_test() of the AIME 2025 guard state
#     (ml_pipeline_9b/aime2025_test, touch 1); it runs this script with PHASE=readouts, then PHASE=analysis;
#   * PHASE=readouts starts one worker per unit; a worker runs its unit's queue (static, below), one readout at a time,
#     each as PHASE=one under `sparklock run --unit U`; with its own queue done it takes any label not yet claimed,
#     starting from the end of the other unit's queue (a unit that is down or busy leaves its rest to the other);
#   * PHASE=one (on the workstation, under the lease): code lock check, stage code (spark/sync_code.sh plus the vllm, broad
#     and ml7 trees), the data file and the checkpoint (from the workstation master copy into <work>/runs/ml7/
#     test9b_stage/<run>/) onto the unit, then spark/spark_job.sh runs the readout detached under its own timeout;
#     on the unit the job first checks the data, code, model and checkpoint sha256, then runs readout_vllm.py; the
#     job dir comes back to the workstation, the readout and its receipts are copied to RES, checked, and marked done.
#     An earlier unit job of the same label with the same command (workstation died mid-wait) is waited for and adopted.
# Readouts (readout_vllm.py, code locked by sha256 at the first readout; the 9B validation readouts' arguments):
#   --caps 256,512,1024,2048,4096,8192 --answer_max 1280 --samples 8 --seed 20261006 --gpu_mem 0.35
#   --probe_effect 0.05 --rank 16; untouched --arm stock; LoRA --arm ctrl; Gate(0,1) --arm wide1; ckpt step 150.
# Placement (owner 2026-10-15: gated and LoRA arms interleaved across the units, both finishing about together):
#   gb10a: untouched ctrl_0 wide1_0          gb10b: ctrl_1 wide1_1 ctrl_2 wide1_2
#   each seed pair (ctrl_s, wide1_s) on one unit, so a unit difference cancels in the seed-paired contrast.
# Outputs: RES (results/ml7_9b/aime2025/): readout_aime2025_<label>.json (+ .traces.jsonl.gz),
#   done/<label>.sha256, jobs/<unit>-<job>/ (cmd, log, meta, gpu.csv, memguard.log, rc), logs/, steps.log, plan.tsv,
#   code.sha256, env.txt, analysis_fresh_9b.json, final.json, DONE or FAIL.
#   bash test9b_spark.sh plan    # every check + the plan (exit 0 = launchable)
#   bash test9b_spark.sh         # the read (resumable: done markers; rerun after a failure)
# Env: NINE_MAX_S (per-readout bound on the unit, 28800), NINE_WAIT (lease wait, 12h),
#   NINE_TTL (whole read, 30h), NINE_MAX_FAILURES (2 per label per launch), NINE_QUEUE_B23D, NINE_QUEUE_CB33.
#   Paths: NINE_VPY, NINE_DATA, NINE_ML, NINE_GD, NINE_RES, NINE_EXPECT, NINE_MODEL_SUMS, NINE_SCRATCH.
#   Test seams (CPU tests only): NINE_SP (spark helper dir), NINE_SPARKLOCK, NINE_RSYNC, NINE_UNIT_CHECKS=0.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SELF=$HERE/$(basename "$0")
ML7=$(cd "$HERE/.." && pwd)
NEXT=$(cd "$ML7/.." && pwd)
WT=$(cd "$HERE/../../../.." && pwd)
SP=${NINE_SP:-$NEXT/spark}
SPARKLOCK=${NINE_SPARKLOCK:-sparklock}
RSYNC=${NINE_RSYNC:-rsync}
VPY=${NINE_VPY:-$HOME/.cache/gdn_gate/vllm/venv/bin/python}
DATA=${NINE_DATA:-${WG_RUNS:-$WT/runs}/ml7/data}
ML=${NINE_ML:-$WT/ml_pipeline_9b}
GD=${NINE_GD:-$ML/aime2025_test}
RES=${NINE_RES:-$WT/results/ml7_9b/aime2025}
EXPECT=${NINE_EXPECT:-$HERE/ckpt_expect_9b.json}
MODEL_SUMS=${NINE_MODEL_SUMS:-$ML/model_Qwen3.5-9B.sha256}
S=${NINE_SCRATCH:-$HOME/.cache/gdn_gate/ml7/test9b}
LOCAL=${SPARK_JOB_LOCAL:-$HOME/.cache/gdn_gate/spark/jobs}
UNIT_CHECKS=${NINE_UNIT_CHECKS:-1}
FILE=test_aime2025.jsonl; MN=Qwen3.5-9B; GPU_NAME="NVIDIA GB10"
CAPS=256,512,1024,2048,4096,8192; ANSWER_MAX=1280; PROBE=0.05; RANK=16; SAMPLES=8; RSEED=20261006
GPU_MEM=0.35  # pre-registered (ml_pipeline_9b/PIPELINE.md); no override
MAXS=${NINE_MAX_S:-28800}
WAIT=${NINE_WAIT:-12h}
MAXF=${NINE_MAX_FAILURES:-2}
UNITS="gb10a gb10b"
Q_b23d=${NINE_QUEUE_B23D:-untouched ctrl_0 wide1_0}
Q_cb33=${NINE_QUEUE_CB33:-ctrl_1 wide1_1 ctrl_2 wide1_2}
LABELS="untouched ctrl_0 ctrl_1 ctrl_2 wide1_0 wide1_1 wide1_2"
SON="$SP/son.sh"
CODE="$ML7/readout_vllm.py $ML7/train_grpo_vllm.py $NEXT/vllm/gate_vllm.py $NEXT/broad/rl_common.py
      $NEXT/broad/arms.py $NEXT/broad/packing.py $ML7/step14_analysis.py $ML7/ctrlpm_analysis.py
      $ML7/fresh/fresh_analysis.py $ML7/fresh/fresh_guard.py $WT/ml_pipeline/guard.py $HERE/test9b_analysis.py
      $HERE/test9b_guard.py $HERE/build_aime2025.py $EXPECT $MODEL_SUMS"

queue_of() { case $1 in gb10a) echo "$Q_b23d";; gb10b) echo "$Q_cb33";; esac; }
unit_of() { local u; for u in $UNITS; do case " $(queue_of "$u") " in *" $1 "*) echo "$u"; return;; esac; done; echo none; }
arm_of() { case $1 in untouched) echo stock;; *) echo "${1%_*}";; esac; }
exp_field() {  # exp_field LABEL FIELD (run, step, path, sha256) from ckpt_expect_9b.json
  /usr/bin/python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]]).replace("<runs>", sys.argv[4]))' "$EXPECT" "$1" "$2" "${WG_RUNS:-$WT/runs}"
}
ckpt_of() { [ "$1" = untouched ] && echo none || exp_field "$1" path; }
out_of() { echo "$RES/readout_aime2025_$1.json"; }
mark_of() { echo "$RES/done/$1.sha256"; }
done_ok() { local m; m=$(mark_of "$1"); [ -f "$m" ] && [ -f "$(out_of "$1")" ] && sha256sum -c --status "$m" 2>/dev/null; }
# projected hours per readout (ordering and the plan table only; basis in PIPELINE.md, "Wall time")
est_of() { case $1 in untouched) echo 3.3;; ctrl_*) echo 3.1;; *) echo 2.9;; esac; }
log() { mkdir -p "$RES"; echo "$*" | tee -a "$RES/steps.log"; }
write_plan() {
  local l
  for l in $LABELS; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$l" 9b "$(arm_of "$l")" "$(ckpt_of "$l")" "$(out_of "$l")" \
      "\$NN/models/$MN" "$ANSWER_MAX" "$PROBE" "$RANK"
  done
}
remote_code_lines() {  # the locked readout code at its unit paths, relative to <work> ($NN)
  awk '{ n = split($2, p, "/"); b = p[n]; d = "";
         if (b == "readout_vllm.py" || b == "train_grpo_vllm.py") d = "ml7";
         else if (b == "gate_vllm.py") d = "vllm";
         else if (b == "rl_common.py" || b == "arms.py" || b == "packing.py") d = "broad";
         if (d != "") printf "%s  code/gdn_gate/%s/%s\n", $1, d, b }' "$RES/code.sha256"
}
remote_cmd() {  # remote_cmd LABEL: the unit job (deterministic for the label under one code lock)
  local l=$1 a ck run step sums code dsha c=""
  a=$(arm_of "$l"); dsha=$(sha256sum "$DATA/$FILE" | cut -d' ' -f1)
  code=$(remote_code_lines | sed "s/.*/'&'/" | tr '\n' ' ')
  sums=$(sed "s/.*/'&'/" "$MODEL_SUMS" | tr '\n' ' ')
  if [ "$l" != untouched ]; then
    run=$(exp_field "$l" run); step=$(exp_field "$l" step)
    ck="\$NN/runs/ml7/test9b_stage/$run/ckpt_step$step.pt"
    c="echo \"$(exp_field "$l" sha256)  $ck\" | sha256sum -c --quiet; "
  fi
  printf '%s' "source ./env.sh; set -e; echo \"$dsha  \$NN/data/ml7/$FILE\" | sha256sum -c --quiet; "
  printf '%s' "printf '%s\\n' $code| (cd \$NN && sha256sum -c --quiet -); "
  printf '%s' "printf '%s\\n' $sums| (cd \$NN/models/$MN && sha256sum -c --quiet -); $c"
  printf '%s' "\$VPY \$NN/code/gdn_gate/ml7/preflight_grader.py \$NN/code/gdn_gate/broad; "
  printf '%s' "codehash \$NN/code/gdn_gate/ml7/readout_vllm.py \$NN/code/gdn_gate/ml7/train_grpo_vllm.py \$NN/code/gdn_gate/vllm/gate_vllm.py \$NN/code/gdn_gate/broad/rl_common.py; "
  printf '%s' "export WG_CACHE_DIR=\$JOB/gen WG_ML7_PYLIB=\$NN/pylib_ml7; "
  printf '%s' "\$VPY \$NN/code/gdn_gate/ml7/readout_vllm.py --model \$NN/models/$MN --data \$NN/data/ml7/$FILE --arm $a"
  [ "$l" = untouched ] || printf '%s' " --ckpt $ck"
  printf '%s' " --caps $CAPS --answer_max $ANSWER_MAX --samples $SAMPLES --seed $RSEED --gpu_mem $GPU_MEM --probe_effect $PROBE --rank $RANK --out \$JOB/readout.json"
}

# ------------------------------------------------------------------------------------------- phases (under the guard)
if [ -n "${PHASE:-}" ]; then
  [ "${FRESH_UNDER_GUARD:-}" = 1 ] || { echo "PHASE=$PHASE runs only under test9b_guard.py final" >&2; exit 2; }
  mkdir -p "$RES/logs" "$RES/done" "$RES/jobs" "$S"
  case "$PHASE" in
    one)  # one readout on UNIT, run by a worker under `sparklock run --unit UNIT`
      l=${LABEL:?LABEL}; U=${UNIT:?UNIT}
      [ -f "$RES/code.sha256" ] || { log "REFUSED: $l: no code lock (PHASE=one runs only after PHASE=readouts)"; exit 2; }
      sha256sum -c --quiet "$RES/code.sha256" || { log "REFUSED: instrument code changed since the first readout ($RES/code.sha256)"; exit 4; }
      date -Is > "$S/ran_${l}_$U"
      if done_ok "$l"; then log "=== $l kept ($(out_of "$l"))"; exit 0; fi
      rm -f "$(mark_of "$l")"
      t0=$(date +%s)
      RS="$SP/rsh.sh $U"
      ck=$(ckpt_of "$l"); run=""
      [ "$ck" = none ] || run=$(exp_field "$l" run)
      { "$SP/sync_code.sh" "$U" &&
        "$SON" "$U" "mkdir -p gdn-gate/code/gdn_gate/vllm gdn-gate/code/gdn_gate/broad gdn-gate/code/gdn_gate/ml7 gdn-gate/data/ml7 gdn-gate/runs/ml7/test9b_stage/${run:-none}" &&
        "$RSYNC" -a --exclude __pycache__ -e "$RS" "$NEXT/vllm/" x:gdn-gate/code/gdn_gate/vllm/ &&
        "$RSYNC" -a --exclude __pycache__ --exclude jobs -e "$RS" "$NEXT/broad/" x:gdn-gate/code/gdn_gate/broad/ &&
        "$RSYNC" -a --exclude __pycache__ -e "$RS" "$ML7/" x:gdn-gate/code/gdn_gate/ml7/ &&
        "$RSYNC" -a -e "$RS" "$DATA/$FILE" x:gdn-gate/data/ml7/ &&
        { [ "$ck" = none ] || "$RSYNC" -a -e "$RS" "$ck" "x:gdn-gate/runs/ml7/test9b_stage/$run/"; }
      } >> "$RES/logs/$l.stage.log" 2>&1 || { log "=== $l FAIL: staging to $U (log $RES/logs/$l.stage.log)"; exit 1; }
      CMD=$(remote_cmd "$l")
      want=$(printf '%s\n' "$CMD" | sha256sum | cut -c1-64)
      lst() { "$SON" "$U" "bash gdn-gate/code/gdn_gate/ml7/test9b/t9b_remote.sh list" 2>/dev/null; }
      # an earlier launch's unit job still running (workstation side died mid-wait) holds the GPU: wait for it under this lease
      while read -r name lab st csha; do
        [ "${st:-}" = running ] || continue
        log "=== $l: earlier unit job $U:$name ($lab) is still running; waiting for it before anything starts on $U"
        waitfor --timeout $((MAXS + 300))s --every 60s --fail-after 20 --probe-timeout 40s \
          --until "$SON $U 'test -e gdn-gate/jobs/$name/rc'" --quiet \
          || { log "=== $l FAIL: $U:$name did not end within $((MAXS + 300))s"; exit 1; }
      done <<< "$(lst)"
      # a finished earlier job of this label with this exact command is adopted instead of read again
      NAME=$(lst | awk -v l="$l" -v w="$want" '$2 == l && $3 == "0" && $4 == w { n = $1 } END { print n }')
      if [ -n "$NAME" ]; then
        log "=== $l: adopting the finished unit job $U:$NAME (same command)"
        rm -rf "$LOCAL/$U-$NAME"; mkdir -p "$LOCAL/$U-$NAME"
        "$SON" "$U" "tar -C gdn-gate/jobs/$NAME -cf - ." | tar -C "$LOCAL/$U-$NAME" -xf -
        rc=$(cat "$LOCAL/$U-$NAME/rc" 2>/dev/null || echo none)
      else
        NAME=t9b-aime2025-$l-$(date +%m%d%H%M%S)
        log "=== $l start $(date -Is) on $U job $NAME"
        "$SP/spark_job.sh" "$U" "$NAME" "$MAXS" "$CMD" 16 >> "$RES/logs/$l.log" 2>&1; rc=$?
      fi
      J=$LOCAL/$U-$NAME
      mkdir -p "$RES/jobs/$U-$NAME"
      for f in cmd log meta.txt memguard.log gpu.csv rc perf_before.txt perf_after.txt readout.json.parity_fail.json; do
        [ -f "$J/$f" ] && cp -p "$J/$f" "$RES/jobs/$U-$NAME/"
      done
      out=$(out_of "$l")
      if [ "$rc" = 0 ] && [ -f "$J/readout.json" ]; then
        cp -p "$J/readout.json" "$out"
        [ -f "$J/readout.json.traces.jsonl.gz" ] && cp -p "$J/readout.json.traces.jsonl.gz" "$out.traces.jsonl.gz"
        if "$VPY" -c 'import hashlib, json, sys
d = json.load(open(sys.argv[1])); n = sum(1 for x in open(sys.argv[2]) if x.strip())
a = d["args"]; dsha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert len(d["rows"]) == n * int(sys.argv[3]) and a["answer_max"] == int(sys.argv[4]) and not d["red"], "rows/answer_max/red"
assert d["arm"] == sys.argv[5] and d["env"]["gpu"] == sys.argv[6] and d["env"]["sha256"]["data"] == dsha, "arm/gpu/data"
assert (d["env"]["sha256"].get("ckpt") or "none") == sys.argv[7], "ckpt sha256"' \
            "$out" "$DATA/$FILE" "$SAMPLES" "$ANSWER_MAX" "$(arm_of "$l")" "$GPU_NAME" \
            "$([ "$l" = untouched ] && echo none || exp_field "$l" sha256)" >> "$RES/logs/$l.log" 2>&1; then
          sha256sum "$out" > "$(mark_of "$l")"
          log "=== $l ok on $U wall=$(( $(date +%s) - t0 ))s job $NAME"
          exit 0
        fi
        log "=== $l FAIL: readout check (log $RES/logs/$l.log)"; rm -f "$out" "$out.traces.jsonl.gz"; exit 1
      fi
      log "=== $l FAIL on $U rc=$rc wall=$(( $(date +%s) - t0 ))s (job $RES/jobs/$U-$NAME)"; exit 1;;
    readouts)
      if [ -f "$RES/code.sha256" ]; then
        sha256sum -c --quiet "$RES/code.sha256" || { log "REFUSED: instrument code changed since the first readout ($RES/code.sha256)"; exit 4; }
      else
        # shellcheck disable=SC2086
        sha256sum $CODE > "$RES/code.sha256"
      fi
      rm -rf "$S/claims" "$S/attempts"; mkdir -p "$S/claims" "$S/attempts"
      claim_next() {  # claim_next UNIT PASS: claim and print the next label (own queue, then the other's from its end)
        local U=$1 pass=$2 l list n
        if [ "$pass" = own ]; then list=$(queue_of "$U")
        else list=$(for u in $UNITS; do [ "$u" = "$U" ] || queue_of "$u"; done | tr ' ' '\n' | sed '/^$/d' | tac | tr '\n' ' '); fi
        for l in $list; do
          done_ok "$l" && continue
          n=$(cat "$S/attempts/$l" 2>/dev/null || echo 0)
          [ "$n" -lt "$MAXF" ] || continue
          mkdir "$S/claims/$l" 2>/dev/null || continue
          echo "$l"; return 0
        done
        return 1
      }
      worker() {
        local U=$1 pass l rc n
        for pass in own other; do
          while l=$(claim_next "$U" "$pass"); do
            rm -f "$S/ran_${l}_$U"
            log "=== worker $U: $l ($pass queue) $(date -Is)"
            SPARKLOCK_WHO=wg-9b-aime PHASE=one FRESH_UNDER_GUARD=1 LABEL=$l UNIT=$U \
              "$SPARKLOCK" run --unit "$U" --wait "$WAIT" --ttl "$((MAXS + 1800))s" \
              --for "wg 9B AIME 2025 test read (owner 2026-10-15): $l" -- bash "$SELF" >> "$RES/logs/worker_$U.log" 2>&1
            rc=$?
            if done_ok "$l"; then rmdir "$S/claims/$l"; continue; fi
            if [ ! -f "$S/ran_${l}_$U" ]; then
              log "=== worker $U: no lease within $WAIT (or the unit is unreachable); $l left to the other unit"
              rmdir "$S/claims/$l"; return 0
            fi
            n=$(( $(cat "$S/attempts/$l" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$S/attempts/$l"
            log "=== worker $U: $l failed rc=$rc (attempt $n of $MAXF)"
            rmdir "$S/claims/$l"
          done
        done
      }
      log "=== readouts $(date -Is): gb10a [$Q_b23d] gb10b [$Q_cb33]"
      for U in $UNITS; do worker "$U" & done
      wait
      fail=0
      for l in $LABELS; do done_ok "$l" || { log "=== $l not done"; fail=1; }; done
      exit $fail;;
    analysis)
      rm -f "$RES/analysis_fresh_9b.json"
      slots=(); for l in $LABELS; do [ "$l" = untouched ] || slots+=("$l=$(out_of "$l")"); done
      log "=== analysis start $(date -Is)"
      "$VPY" "$HERE/test9b_analysis.py" --baseline "$(out_of untouched)" "${slots[@]}" \
        --manifest "$GD/split_manifest.json" --out "$RES/analysis_fresh_9b.json" > "$RES/logs/analysis.log" 2>&1 \
        && { log "=== analysis ok ($RES/analysis_fresh_9b.json)"; exit 0; }
      log "=== analysis FAIL (log $RES/logs/analysis.log)"; exit 1;;
    *) echo "unknown PHASE $PHASE" >&2; exit 2;;
  esac
fi

# ------------------------------------------------------------------------------------------- inside the bound
if [ "${INNER:-0}" = 1 ]; then
  mkdir -p "$RES/logs" "$RES/done" "$S"
  { echo "--- inner start $(date -Is) host $(hostname) GPU_MEM=$GPU_MEM queues gb10a [$Q_b23d] gb10b [$Q_cb33]"
    echo "git: $(git -C "$HERE" describe --always --dirty --tags 2>&1)"
    "$SPARKLOCK" status 2>&1
    # shellcheck disable=SC2086
    sha256sum $CODE "$SELF" "$HERE/t9b_remote.sh" 2>&1; } >> "$RES/env.txt"
  cp "$S/plan.tsv" "$RES/plan.tsv"
  exec "$VPY" "$HERE/test9b_guard.py" final --gd "$GD" --md "$ML/PIPELINE.md" --data "$DATA" --plan "$RES/plan.tsv" \
    --samples "$SAMPLES" --runner "$SELF" --res "$RES" --out "$RES/final.json"
fi

# ------------------------------------------------------------------------------------------- outer: checks, plan, run
MODE=${1:-launch}
case "$MODE" in plan|launch) ;; *) echo "usage: test9b_spark.sh [plan]" >&2; exit 2;; esac
mkdir -p "$S"
if [ -f "$RES/final.json" ]; then
  echo "test9b: the AIME 2025 read is recorded ($RES/final.json); the set is read once, nothing to run"
  [ "$MODE" = plan ] && exit 0 || exit 2
fi
write_plan > "$S/plan.tsv"
rc=0
[ -x "$VPY" ] || { echo "REFUSED: no workstation vLLM venv python $VPY" >&2; rc=2; }
[ -f "$MODEL_SUMS" ] || { echo "REFUSED: no model sha256 list $MODEL_SUMS" >&2; rc=2; }
[ -f "$DATA/$FILE" ] || { echo "REFUSED: no $DATA/$FILE (build_aime2025.py)" >&2; rc=2; }
if [ -x "$VPY" ]; then
  "$VPY" "$HERE/test9b_guard.py" preflight --gd "$GD" --md "$ML/PIPELINE.md" --data "$DATA" --plan "$S/plan.tsv" \
    --expect "$EXPECT" || rc=2
  CUDA_VISIBLE_DEVICES="" "$VPY" "$ML7/preflight_grader.py" "$NEXT/broad" > "$S/grader_preflight.log" 2>&1 \
    && echo "grader preflight (workstation): OK" || { echo "REFUSED: grader preflight (math_verify), log $S/grader_preflight.log" >&2; rc=2; }
fi
if [ -f "$RES/code.sha256" ]; then
  sha256sum -c --quiet "$RES/code.sha256" && echo "code lock: $RES/code.sha256 holds" \
    || { echo "REFUSED: code differs from the lock of the first readout ($RES/code.sha256)" >&2; rc=2; }
else
  echo "code lock: written at the first readout (PHASE=readouts)"
fi
for f in $CODE; do [ -f "$f" ] || { echo "REFUSED: code file missing: $f" >&2; rc=2; }; done
[ -f "$HERE/t9b_remote.sh" ] || { echo "REFUSED: no $HERE/t9b_remote.sh" >&2; rc=2; }
q=$(for u in $UNITS; do queue_of "$u"; done | tr ' ' '\n' | sed '/^$/d' | sort | tr '\n' ' ')
want=$(echo $LABELS | tr ' ' '\n' | sort | tr '\n' ' ')
[ "$q" = "$want" ] || { echo "REFUSED: the unit queues [$q] are not the labels [$want], each once" >&2; rc=2; }
exec 7> "$S/launch.lock"
flock -n 7 || { echo "REFUSED: another test9b launch holds $S/launch.lock" >&2; rc=2; }
if [ "$UNIT_CHECKS" = 1 ]; then
  for U in $UNITS; do
    sums=$(sed "s/.*/'&'/" "$MODEL_SUMS" | tr '\n' ' ')
    chk=$("$SON" "$U" "test -x gdn-gate/venv-vllm/bin/python && echo venv; test -d gdn-gate/pylib_ml7/fla && echo fla; test -x gdn-gate/memguard.sh && echo memguard; df -BG --output=avail \$HOME | tail -1 | tr -dc 0-9; echo; printf '%s\\n' $sums | (cd gdn-gate/models/$MN 2>/dev/null && sha256sum -c --quiet -) && echo model" 2>&1)
    for w in venv fla memguard model; do
      echo "$chk" | grep -qx "$w" || { echo "REFUSED: $U: check '$w' failed (vLLM venv, pylib_ml7/fla, memguard.sh, $MN sha256 vs $MODEL_SUMS)" >&2; rc=2; }
    done
    gfree=$(echo "$chk" | grep -x '[0-9]*' | head -1)
    [ "${gfree:-0}" -ge 20 ] || { echo "REFUSED: $U: ${gfree:-?} GB free on the unit's disk (need 20)" >&2; rc=2; }
    [ $rc = 0 ] && echo "unit $U: reachable; vLLM venv, fla, memguard, $MN (13 files sha256 OK), ${gfree} GB free"
    st=$("$SPARKLOCK" status --unit "$U" 2>&1)
    echo "lease $st$(case "$st" in *free*) ;; *) echo " (busy now: the worker waits up to $WAIT for it)";; esac)"
  done
fi
echo "plan: $(echo $LABELS | wc -w) readouts of $FILE ($(grep -c . "$DATA/$FILE" 2>/dev/null) items) x $SAMPLES samples, caps $CAPS, answer_max $ANSWER_MAX, probe $PROBE, rank $RANK, gpu_mem $GPU_MEM, seed $RSEED; one guard touch (touch 1) of $GD"
printf '  %-10s %-6s %-5s %-6s %-28s %s\n' label arm unit est_h checkpoint status
for U in $UNITS; do
  for l in $(queue_of "$U"); do
    st=todo; done_ok "$l" && st=done
    ck=$(ckpt_of "$l"); [ "$ck" = none ] || ck="$(exp_field "$l" run)@$(exp_field "$l" step)"
    printf '  %-10s %-6s %-5s %-6s %-28s %s\n' "$l" "$(arm_of "$l")" "$U" "$(est_of "$l")" "$ck" "$st"
  done
  echo "  $U projected: $(for l in $(queue_of "$U"); do est_of "$l"; done | awk '{ s += $1 } END { printf "%.1f", s }') h (central; range in PIPELINE.md)"
done
if [ "$MODE" = plan ]; then
  [ $rc = 0 ] && echo "plan: launchable" || echo "plan: refused (reasons above)"
  exit $rc
fi
[ $rc = 0 ] || exit $rc
mkdir -p "$RES"
rm -f "$RES/FAIL"
INNER=1 timeout --kill-after=120 "${NINE_TTL:-30h}" bash "$SELF"
rc=$?
if [ $rc = 0 ] && [ -f "$RES/final.json" ]; then date -Is > "$RES/DONE"
else echo "rc=$rc $(date -Is) (TTL hit, a readout or the analysis failed, or the guard refused; rerun resumes)" > "$RES/FAIL"; fi
exit $rc
