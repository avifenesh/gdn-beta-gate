#!/usr/bin/env bash
# mech_box.sh g1|g2 LABEL | g3 PART: the mechanism GPU jobs of lane wg on the Phase 5 rented box (validation split
# only, never a test set; job list from the CPU mechanism lane, FINDINGS "## 2026-10-12 mechanism"). Run by the box
# scheduler (orch.py, BOXQ_PHASE=5) on one GPU slot.
#   g1 LABEL  gate ablation at inference (ablate_spark.sh's conditions on the box): <arm>_s<k>_full (checkpoint as
#             trained), <arm>_s<k>_strip (same LoRA, no gate: readout --strip), <arm>_s<k>_hybrid (warm-up LoRA + trained
#             gate, ~/.cache/gdn_gate/mech/hybrid), warmup, ctrl_s<k>; arm wide1|wide, k 0-2.
#   g2 LABEL  gate dose-response: wide1_s<k>_x<f> = the wide1 step-150 checkpoint with every gate tensor times f
#             (f 0, 0.5, 1.5, 2; mech/dose_ckpts.py, receipt beside the files).
#   g3 PART   gate activations under teacher forcing (mech/gate_stats.py) on the 64 fixed validation traces x 2,048
#             tokens (~/.cache/gdn_gate/mech/inputs.json), the 43 checkpoints of the mechanism list split in PARTs
#             a and b.
# Readouts: readout_vllm.py (unchanged) --model Qwen3.5-4B --data val.jsonl --caps 256..8192 --answer_max 1280
# --samples 1 --seed 20261006 --gpu_mem 0.45 --probe_effect 0.05, --gate_stats on gated engines; outputs
# results/mech/{ablate,dose,gate_stats}/. Every input checked by sha256 first; a finished output is kept.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
NEXT=$HERE/..
WT=$(cd "$HERE/../../.." && pwd)
VPY=${MECH_VPY:-$HOME/.cache/gdn_gate/vllm/venv/bin/python}
M4=${MECH_M4:-${WG_HF_CACHE:-$HOME/.cache/huggingface/hub}/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
G=${WG_RUNS:-$WT/runs}/ml7_4b/grpo
HY=${MECH_HYBRID:-$HOME/.cache/gdn_gate/mech/hybrid}
DOSE=${MECH_DOSE:-$HOME/.cache/gdn_gate/mech/dose}
INPUTS=${MECH_INPUTS:-$HOME/.cache/gdn_gate/mech/inputs.json}
PYLIB=${MECH_PYLIB:-$HOME/.cache/gdn_gate/ml7/pylib}
VAL=${WG_RUNS:-$WT/runs}/ml7/data/val.jsonl
VAL_SHA=8251b0d493e1371f4aece78b1e163708078d8cda35ece92c2d16f7b6892750dd
RES=$WT/results/mech
KIND=${1:?g1|g2|g3}; L=${2:?label}
echo "$VAL_SHA  $VAL" | sha256sum -c --quiet || { echo "REFUSED: val.jsonl sha256"; exit 2; }
export VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONUNBUFFERED=1 WG_ML7_PYLIB="$PYLIB" OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PATH="$PATH:$(dirname "$VPY")"
want_sha() {  # FILE RECEIPT KEY: the sha256 the receipt records for KEY
  "$VPY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["sha256"])' "$1" "$2"
}
case $KIND in
  g1|g2)
    if [ "$KIND" = g1 ]; then
      OUT=$RES/ablate/$L.json
      case $L in
        *_strip)  R=${L%_strip};  CK=$G/${R}_cap512_r_box/ckpt_step150.pt; ARM=${R%_s?}; EXTRA=(--strip); WANT="";;
        *_hybrid) R=${L%_hybrid}; CK=$HY/${R}_warmLoRA_gate150.pt; ARM=${R%_s?}; EXTRA=(--gate_stats)
                  WANT=$(want_sha "$HY/receipt.json" "hybrid/${R}_warmLoRA_gate150.pt");;
        *_full)   R=${L%_full};   CK=$G/${R}_cap512_r_box/ckpt_step150.pt; ARM=${R%_s?}; EXTRA=(--gate_stats); WANT="";;
        warmup)   CK=$G/ctrl_s0_cap512_fmt_box/ckpt_step40.pt; ARM=ctrl; EXTRA=(); WANT="";;
        ctrl_s?)  CK=$G/${L}_cap512_r_box/ckpt_step150.pt; ARM=ctrl; EXTRA=(); WANT="";;
        *) echo "unknown g1 label $L"; exit 2;;
      esac
    else
      OUT=$RES/dose/$L.json
      case $L in wide1_s?_x*) CK=$DOSE/$L.pt; ARM=wide1; EXTRA=(--gate_stats); WANT=$(want_sha "$DOSE/receipt.json" "$L.pt");;
        *) echo "unknown g2 label $L"; exit 2;; esac
    fi
    [ -f "$OUT" ] && { echo "kept $OUT"; exit 0; }
    [ -f "$CK" ] || { echo "REFUSED: no checkpoint $CK"; exit 2; }
    if [ -n "$WANT" ]; then echo "$WANT  $CK" | sha256sum -c --quiet || { echo "REFUSED: $CK sha256 differs from its receipt"; exit 2; }; fi
    if [ -z "$WANT" ] && [ -f "$WT/src/gdn_gate/ml7/fresh/ckpt_expect.json" ]; then
      key=$(case $L in warmup) echo 4b_warmup;; ctrl_s?) echo "4b_ctrl_${L#ctrl_s}";; *) r=${L%_*}; echo "4b_${r%_s?}_${r##*_s}";; esac)
      W2=$(want_sha "$WT/src/gdn_gate/ml7/fresh/ckpt_expect.json" "$key" 2>/dev/null) && \
        { echo "$W2  $CK" | sha256sum -c --quiet || { echo "REFUSED: $CK sha256 differs from the recorded $key"; exit 2; }; }
    fi
    mkdir -p "$(dirname "$OUT")"
    # per-GPU start lock held MECH_STAGGER s after each start: two engines starting at once race for memory
    exec 8> "/tmp/wg_gpu${CUDA_VISIBLE_DEVICES:-x}.start.lock"; flock 8; ( sleep "${MECH_STAGGER:-150}"; flock -u 8 ) &
    export WG_CACHE_DIR=$HOME/.cache/gdn_gate/mech/gen_${KIND}_$L; rm -rf "$WG_CACHE_DIR"; mkdir -p "$WG_CACHE_DIR"
    "$VPY" "$NEXT/ml7/readout_vllm.py" --model "$M4" --data "$VAL" --arm "$ARM" --ckpt "$CK" "${EXTRA[@]}" \
      --caps 256,512,1024,2048,4096,8192 --answer_max 1280 --samples 1 --seed 20261006 --gpu_mem 0.45 \
      --probe_effect 0.05 --out "$OUT.part"
    rc=$?
    [ -f "$OUT.part.parity_fail.json" ] && cp "$OUT.part.parity_fail.json" "$RES/$KIND.$L.parity_fail.json"
    [ $rc = 0 ] || exit $rc
    [ -f "$OUT.part.traces.jsonl.gz" ] && mv "$OUT.part.traces.jsonl.gz" "$OUT.traces.jsonl.gz"
    mv "$OUT.part" "$OUT"; rm -rf "$WG_CACHE_DIR"; echo "done $OUT";;
  g3)
    OUT=$RES/gate_stats; mkdir -p "$OUT"
    J=()
    add() { J+=(--job "$1:$2:$3:$4:fixed"); }
    if [ "$L" = a ]; then
      add warmup wide1 16 "$G/ctrl_s0_cap512_fmt_box/ckpt_step40.pt"
      for a in wide1 wide; do for s in 0 1 2; do add "${a}_s${s}_150" $a 16 "$G/${a}_s${s}_cap512_r_box/ckpt_step150.pt"; done; done
      for s in 0 1 2; do add "ctrl_s${s}_150" wide1 16 "$G/ctrl_s${s}_cap512_r_box/ckpt_step150.pt"; done
      for s in 0 1 2; do add "ctrlpm_s${s}_150" wide1 20 "$G/ctrlpm_s${s}_cap512_r_box/ckpt_step150.pt"; done
      for t in 25 50; do for a in wide1 wide; do for s in 0 1 2; do add "${a}_s${s}_$t" $a 16 "$G/${a}_s${s}_cap512_r_box/ckpt_step$t.pt"; done; done; done
    else
      for t in 75 100 125; do for a in wide1 wide; do for s in 0 1 2; do add "${a}_s${s}_$t" $a 16 "$G/${a}_s${s}_cap512_r_box/ckpt_step$t.pt"; done; done; done
    fi
    PYTHONPATH=$PYLIB "$VPY" "$HERE/gate_stats.py" --model "$M4" --inputs "$INPUTS" --out_dir "$OUT" "${J[@]}";;
  *) echo "unknown kind $KIND"; exit 2;;
esac
