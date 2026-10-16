#!/usr/bin/env bash
# CPU checks of the release. Tier A needs only the uv environment (uv sync --locked) and reproduces every receipt the
# paper reads from released data (RL analyses incl. the fresh held-out set and the 9B AIME 2025 read, RULER pairing,
# mechanism summaries).
# Tier B runs the lab's CPU tests of the trainer, readout and analysis code; it needs
# the GPU environment env/vllm.txt (CPU use only), the model config and tokenizer files of the pinned revisions in
# $WG_HF_CACHE, and the rebuilt data in $WG_RUNS/ml7/data (scripts/rebuild_data.py).
#   bash tests/run_cpu_checks.sh            # tier A
#   TIER_B_PY=.venv-vllm/bin/python bash tests/run_cpu_checks.sh b    # tier A and tier B
set -u
cd "$(dirname "$0")/.."
PY=${PY:-uv run python}
fail=0
step() { echo "== $*"; "$@" || { echo "FAILED: $*"; fail=1; }; }
step $PY paper/figs/check_claims.py
step $PY scripts/collect_env.py --check
step $PY scripts/rerun_analysis.py
step $PY scripts/rerun_ruler_pairing.py
step $PY scripts/rerun_mech.py
tmp=$(mktemp -d)
step $PY src/gdn_gate/olmo/analyze_nll.py --run results/olmo --out "$tmp/nll.json"
step $PY -c "import json,sys; a=json.load(open('$tmp/nll.json')); b=json.load(open('results/olmo/lesion_nll_base_summary.json')); sys.exit(0 if a==b else 1)"
cp paper/data/derived.json "$tmp/derived.json"
step $PY paper/figs/derive.py
step cmp paper/data/derived.json "$tmp/derived.json"
cp paper/data/extra.json "$tmp/extra.json"
step $PY paper/figs/stats_extra.py
step cmp paper/data/extra.json "$tmp/extra.json"
rm -rf "$tmp"
if [ "${1:-}" = b ]; then
  B=${TIER_B_PY:?set TIER_B_PY to the python of the env/vllm.txt environment}
  export CUDA_VISIBLE_DEVICES= TEST_THREADS=${TEST_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
  step bash -c "cd src/gdn_gate/ml7 && \"$B\" test_readout_cpu.py"
  step bash -c "cd src/gdn_gate/ml7 && \"$B\" test_step14_cpu.py"
  step bash -c "cd src/gdn_gate/vllm && \"$B\" test_vcount_cpu.py"
  step bash -c "cd src/gdn_gate/broad && \"$B\" test_cpu.py --out $(mktemp -d)/test_cpu.json"
  step bash -c "cd src/gdn_gate/ml7 && \"$B\" test_grpo_vllm_cpu.py --out $(mktemp -d)/test_grpo_vllm_cpu.json"
  step bash -c "cd src/gdn_gate/ml7 && \"$B\" test_ctrlpm_cpu.py --out $(mktemp -d)/test_ctrlpm_cpu.json"
  step bash -c "cd src/gdn_gate/ml7/fresh && \"$B\" test_effect_visible_cpu.py"
  step bash -c "cd src/gdn_gate/ml7/fresh && \"$B\" test_fresh_cpu.py"
  step bash -c "cd src/gdn_gate/ml7/test9b && \"$B\" test_test9b_cpu.py"
fi
[ $fail = 0 ] && echo "ALL CPU CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }
