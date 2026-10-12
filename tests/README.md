# Tests

`uv run pytest tests/` runs `tests/test_release.py`, the tier A checks below as eight tests.

`bash tests/run_cpu_checks.sh` runs tier A: the claims checker, the environment summary check, the rerun of all 15
frozen RL analyses (including the fresh held-out set) from the released readouts, the four mechanism summaries, the four paired RULER receipts from the released per-item scores, the
Olmo NLL summary from the released per-chunk sums, `derive.py` and `stats_extra.py` (their outputs must equal the committed
`paper/data/derived.json` and `paper/data/extra.json`). Needs only `uv sync --locked`; a few minutes on one CPU.

`TIER_B_PY=<python of env/vllm.txt> bash tests/run_cpu_checks.sh b` adds the lab's CPU tests of the code itself, on
tiny random Qwen3.5 models with the real tokenizer, each with red arms that must fail:

| Test | Covers | Needs |
|---|---|---|
| `src/gdn_gate/ml7/test_readout_cpu.py` | readout bookkeeping, budget prefixes, grader | nothing extra |
| `src/gdn_gate/ml7/test_step14_cpu.py` | decision rule, bootstrap, refusals of the step 12/14 analysis (49 checks) | Qwen3.5-2B tokenizer |
| `src/gdn_gate/vllm/test_vcount_cpu.py` | counting probe jobs, layout, pairing | nothing extra |
| `src/gdn_gate/broad/test_cpu.py` | gate bound, gate reaches the output, cached decode, SFT trainer and resume (22 checks) | Qwen3.5-4B-Base config and tokenizer, eval data, chat SFT parquet |
| `src/gdn_gate/ml7/test_grpo_vllm_cpu.py` | GRPO loss, reward, micro-row invariance, packing, length accounting (27 checks) | rebuilt RL data |
| `src/gdn_gate/ml7/test_ctrlpm_cpu.py` | LoRA-r20 parameter count, widening, analysis on real readouts | Qwen3.5-4B config, 4B warm-up checkpoint |
| `src/gdn_gate/ml7/fresh/test_effect_visible_cpu.py` | the 2026-10-12 self-test amendment: only the two listed (checkpoint, fresh file) pairs get bar 4.0, red arms keep 5.0, agreement checks unaffected, ledger walk to the lock | nothing extra |
| `src/gdn_gate/ml7/fresh/test_fresh_cpu.py` | fresh-set dedup, analysis (H5/H6, refusals, 6-seed rule, per-source blocks), guard preflight and single read, runner plan | rebuilt validation data |

On the author's workstation (2026-10-12, released tree) tier A passed in full, and tier B passed in full with the
pinned tokenizer and config files, the data rebuilt by `scripts/rebuild_data.py` and the 4B warm-up checkpoint
converted from the published adapter (`hf download Avifenesh/gdn-beta-gate-qwen3.5-4b-grpo --include "warmup/*"`,
then `scripts/adapters.py to-pt` to `$WG_RUNS/ml7_4b/grpo/ctrl_s0_cap512_fmt_box/ckpt_step40.pt`): readout checks
PASS, step 14 analysis 49 of 49, counting probe ALL OK, broad 22 of 22, GRPO 27 of 27, LoRA-r20 17 of 17, effect_visible
amendment 9 of 9, fresh-set tooling 26 of 26. The broad test also needs the chat SFT parquet and eval data of
data/MANIFEST.json under `$WG_RUNS/broad`. The code-lock checks (LoRA-r20 P0, fresh F0, `code_ledger.py accept`)
compare the instrument files with the lab's code lock through `results/SCRUB_LEDGER.json`, and the LoRA-r20 P2 check
accepts the published warm-up by a digest of its tensors.

GPU behavior (kernels, engine parity, numerics) is not covered by these tests: every training run and readout
checks engine-trainer parity on its own GPU before it starts, and records the result in its receipt (`parity`).
