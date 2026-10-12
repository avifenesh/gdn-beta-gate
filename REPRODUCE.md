# Reproducing the paper

Paper version, tag and archive: the draft of October 2026 maps to tag `v1.0.0` of this repository. The arXiv
identifier and the archive DOI are added here when they exist. The trained adapters are on the Hugging Face Hub at
these revisions (docs/ADAPTERS.md):

| Hub repository | Revision (commit) |
|---|---|
| `Avifenesh/gdn-beta-gate-qwen3.5-4b-grpo` | `ee1e6c0b031b13f009b50009278cc03f5b1d07ec` |
| `Avifenesh/gdn-beta-gate-qwen3.5-2b-grpo` | `9bb0c56a6b4b43f10a7f6b2cfe50349c2b80eb6f` |
| `Avifenesh/gdn-beta-gate-qwen3.5-4b-base-chat-sft` | `333c4445c97aa603e998f5ddb297db71c769fff2` |
| `Avifenesh/gdn-beta-gate-qwen3.5-4b-base-forgetting` | `20bb1d22ae83042682c8f024ce9e8628dc4f628e` |
| `Avifenesh/gdn-beta-gate-qwen3.5-9b-base-forgetting` | `8be3d70f03f5f6211536a37debbacca7885c728a` |

There are two levels of reproduction:

1. **From the released readouts (CPU, minutes).** Every analysis receipt, table and figure is recomputed from the
   per-item readouts in `results/`, and every number in the paper is checked against its receipt. This is exact.
2. **From scratch (GPU).** Rebuild the data, train the arms, read them out, analyse. The commands below are the ones
   the lab ran; the readouts are sampled, so a rerun reproduces the protocol and, within seed and item noise, the
   numbers, not the bytes.

## 0. Claims map

GPU hours are for one rerun on the stated hardware. "Expected" is the paper value; per-seed values are in the
receipts. The tolerance for a GPU rerun is the decision rule (3 of 3 seeds with one sign and a paired item-bootstrap
95% interval that excludes 0) plus a numeric band of about one observed seed range; for the CPU rerun it is exact.

| Claim | Paper location | Experiment id (receipt) | Command | Hardware | Wall time | Expected | Tolerance |
|---|---|---|---|---|---|---|---|
| Gated arms raise MATH-500 area over LoRA at 4B | Abstract, Results (Qwen3.5-4B), Table 1, Fig. 1 | `results/ml7_4b/step14_ctrlpm/analysis_math500.json` | sections 2.2 to 2.5 | 1 RTX PRO 6000 (96 GB) per run or readout | ~21 h training + ~45 h readouts (GPU-h) | LoRA 0.786, Gate(0,1) 0.852, Gate(0,2) 0.849; Gate(0,2) - LoRA +0.063 [+0.048, +0.078] | rule holds; seed-mean area within 0.02 |
| Gated arms shorten GSM8K reasoning (H2) | Results (Qwen3.5-4B) | `results/ml7_4b/step14_ctrlpm/analysis_gsm8k.json` | same | same | same | LoRA 1,666, Gate(0,1) 297, Gate(0,2) 261 tokens; H2 -1,405 | rule holds; lengths within 25% |
| Parameter-matched LoRA-r20 does not reproduce it (H3, H4) | Results (Qwen3.5-4B) | same two files, `preregistered_hypotheses` | sections 2.4, 2.5 | same | ~4 h training + ~8 h readouts | H3 +0.060 [+0.049, +0.072]; H4 -1,000 tokens | rule holds |
| Effect holds on the fresh held-out set at 4B (H5, H6) | Abstract, Results (fresh set), Fig. 2 | `results/fresh/analysis_fresh_4b.json` | section 2.10 | 1 RTX PRO 6000 per readout | ~1.3 h per readout, 14 readouts (~18 GPU-h) | H5 +0.110 [+0.089, +0.132]; H6 -1,329 [-1,585, -1,081] tokens; both supported | rule holds; seed-mean area within 0.03 |
| 2B replication on the fresh set (secondary) | Results (fresh set) | `results/fresh/analysis_fresh_2b.json` | section 2.10 | same | ~1 h per readout, 19 readouts | H5-2B +0.012, H6-2B -126 tokens; not supported | verdicts hold |
| Effect at 2B, read 1 and fresh seeds | Results (Qwen3.5-2B) | `results/ml7/step14/`, `results/ml7/step14_ext/` | section 2.6 | GB10 or RTX PRO 6000 per run | ~4.8 h per 2B run on GB10, ~0.9 h on RTX PRO 6000 | +0.027 [+0.019, +0.035]; +0.025 [+0.016, +0.034] | rule holds |
| Range adds nothing in RL | Results (range) | `effects/wide - wide1` in the four test analyses | as above | as above | as above | +0.006, +0.003, -0.002, all no effect | no effect |
| The trained gate carries the shortening (G1) | Mechanism, Fig. 3a | `results/mech/g1g2_summary.json` (`g1_condition_means`) | section 2.11 | 1 RTX PRO 6000 per readout | 22 readouts, 0.03 to 0.4 h each | Gate(0,1) length: trained 307, gate removed 665, gate only 970; LoRA 1,600, warm-up 2,916 | order holds per seed |
| Length falls with the gate scale (G2) | Mechanism, Fig. 3b | `results/mech/g1g2_summary.json` (`g2_dose`), `results/mech/dose_receipt.json` | section 2.11 | same | 12 readouts | monotone in all 3 seeds; area 0.949 to 0.963 | monotone per seed |
| Gate moves beta more than LoRA (G3) | Mechanism, Fig. 3c | `results/mech/gate_stats_summary/gate_stats_summary_fixed.json` | section 2.11 | 1 GPU (teacher forcing) | about 1 h for 43 checkpoints | mean abs change of beta at step 150: Gate(0,1) 0.018 to 0.021, Gate(0,2) 0.038 to 0.041 | seed range |
| Where the tokens go, item-matched gap | Mechanism (text and training dynamics) | `results/mech/text_stats.json`, `item_match.json`, `traj.json`, `traj_extra.json` | section 2.11 | CPU | minutes | e.g. MATH-500: 94% of the saving is after the answer | exact (from released files where released) |
| Range gives back half the SFT gain | Appendix (SFT and forgetting), Fig. 7 left | `paper/data/derived.json` `step6_sft` (from `results/run_logs/broad/step6/*/val_log.jsonl`) | section 2.7 | seed 0: 1 GB10 per run; seeds 1 to 5: 1 RTX PRO 6000 per run | seed 0: about 8 h per run; seeds 1 to 5: about 1.5 h per run (18 runs) | Gate(0,1) - LoRA -0.0061; Gate(0,2) - Gate(0,1) +0.0032, 6 of 6 | signs per seed |
| Gate reduces forgetting | Appendix (SFT and forgetting), Fig. 7 right | `results/rescore/`, `results/w1main/` | section 2.8 | 1 RTX PRO 6000 (training), 1 GPU 24 GB+ (scoring) | training about 1.9 h per 4B run and 2.5 h per 9B run on one RTX PRO 6000 (`train_seconds` in the run configs); scoring not recorded | HumanEval LoRA 0.152, Gate(0,2) 0.264, Gate(0,1) 0.409 (4B seed means) | order holds per seed |
| Clamp costs counting in Olmo-Hybrid-7B | Appendix (Olmo-Hybrid-7B), Fig. 8 | `results/olmo/vllm2/counting_paired.json`, `counting_localization.json`, `ruler_paired.json` | section 2.9 | 1 GPU, 24 GB+ (RTX 5090 Laptop in the lab) | ~2 h per arm for the 500-item cells (estimate from the battery timing) | CWE -3.8 / -6.3 / -6.0 points at 4k / 8k / 16k | sign and p < 0.05 per cell |
| Clamp leaves the broad battery | Appendix (Olmo-Hybrid-7B) | `results/olmo/lesion_nll_base_summary.json`, `results/olmo/vllm2/battery/*/lmeval_lm.json` | section 2.9 | same | ~3 h per arm for the vLLM battery (receipt `timing_s`) | NLL +0.020 [0.017, 0.023]; GSM8K 0.747 vs 0.745 | within 2 points |
| Every number in the paper | all | `paper/CLAIMS.md` (1044 rows) | `uv run python paper/figs/check_claims.py` | CPU | seconds | 0 failed | per row |

## 1. Getting started (CPU, about 10 minutes)

    uv sync --locked
    uv run python paper/figs/check_claims.py      # prints "1044 rows checked, 0 failed"
    uv run python scripts/rerun_analysis.py       # 15 lines, each "OK"; exit 0
    uv run python scripts/rerun_mech.py           # 4 lines, each "OK" (G1/G2, G3, item match, training windows)
    uv run python paper/figs/derive.py            # rewrites paper/data/derived.json from results/ (git diff: empty)
    uv run python paper/figs/stats_extra.py       # rewrites paper/data/extra.json: intervals, censoring-aware lengths (git diff: empty)
    SOURCE_DATE_EPOCH=1760227200 uv run python paper/figs/make_figs.py   # rewrites paper/figs/*.pdf byte for byte
    uv run pytest tests/                          # 8 passed: the checks above as tests (tests/README.md)
    bash tests/run_cpu_checks.sh                  # the same, plus tier B with the GPU environment (see tests/README.md)

Two paper scripts need lab files that are not released and are kept as records of how their outputs were made:
`paper/figs/compute_hours.py` (sums the timing logs of the run trees into `paper/data/compute_hours.json`) and
`paper/figs/example_traces.py` (decodes the paired example traces of the implementation appendix into
`paper/data/example_traces.txt` from the raw trace files).

Expected output of `rerun_analysis.py` on any machine:

    OK   results/ml7/step14/analysis_gsm8k.json (released file predates the keys length_effects and preregistered_hypotheses; every released key is equal)
    OK   results/ml7/step14/analysis_math500.json (released file predates ...)
    OK   results/ml7/step14_ext/analysis_gsm8k.json
    ...
    OK   results/ml7_4b/step12_ctrlpm/analysis_val.json

## 2. Full reproduction

Set the roots once (paths are examples):

    export WG_RUNS=$PWD/runs                         # data, checkpoints, run logs
    export WG_HF_CACHE=~/.cache/huggingface/hub      # model snapshots by revision (data/MANIFEST.json)
    export WG_ML7_PYLIB=$PWD/runs/pylib              # flash-linear-attention 0.5.2 for the GRPO trainer forward
    huggingface-cli download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    huggingface-cli download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc
    M4=$WG_HF_CACHE/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

### 2.1 Data

    uv run python scripts/rebuild_data.py            # writes $WG_RUNS/ml7/data/{rl_pool,val,test_gsm8k,test_math500,test_fresh}.jsonl
                                                     # and checks each sha256 against ml_pipeline/split_manifest.json
                                                     # and ml_pipeline_4b/fresh_test/split_manifest.json (byte for byte)

The split rules, the sibling removal (8 training problems) and the grader self-check are in `ml_pipeline/02_prep.py`
(a marimo notebook; `python ml_pipeline/02_prep.py` regenerates its artifacts) and `ml_pipeline/PIPELINE.md`. The
other frozen inputs (chat SFT mixture, replay streams, held-out NLL text, battery items) are listed with their
builder and sha256 in `data/MANIFEST.json`.

### 2.2 Shared warm-up (4B record)

GRPO with the correctness reward alone, 40 steps, from the untouched model. Environment: `env/vllm.txt` plus
`WG_ML7_PYLIB`. One GPU per run (`CUDA_VISIBLE_DEVICES`), 96 GB.

    export GRPO_MODEL=$M4 GRPO_MANIFEST=$PWD/ml_pipeline_4b/split_manifest.json GPU_MEM=0.3 MICRO=8192
    export GRPO_VPY=$PWD/.venv-vllm/bin/python GRPO_PYLIB=$WG_ML7_PYLIB
    bash src/gdn_gate/ml7/run_grpo.sh ctrl 0 512 fmt_box --length_on reasoning --alpha 0.0 --steps 40 \
         --answer_max 1280 --probe_effect 0.05

### 2.3 Arm runs (4B: LoRA, Gate(0,1), Gate(0,2), seeds 0 to 2)

    W=$WG_RUNS/ml7/grpo/ctrl_s0_cap512_fmt_box/ckpt_step40.pt
    for ARM in ctrl wide1 wide; do for S in 0 1 2; do
      bash src/gdn_gate/ml7/run_grpo.sh $ARM $S 512 r_box --length_on reasoning --alpha 0.2 --steps 150 \
           --answer_max 1280 --probe_effect 0.05 --init_from $W
    done; done

The exact arguments of every paper run, with its startup parity self-test, are in
`results/run_configs/{2B_rl,4B_rl}/<run>/meta.json`. The training cap 512 and answer cap 1,280 come from the
validation rules in `ml_pipeline_4b/PIPELINE.md` (receipts `results/ml7_4b/CAP`, `results/ml7_4b/answer_cap.json`,
`results/ml7_4b/pilots/`).

### 2.4 LoRA-r20 (parameter-matched control)

Widen the warm-up from rank 16 to rank 20 (first 16 components copied, new A rows by PEFT's default init, new B
columns zero), then train with the trainer's `--rank 20`:

    for S in 0 1 2; do
      python src/gdn_gate/ml7/pad_rank.py --src $W --rank 20 --seed $S \
             --out $WG_RUNS/ml7/grpo/ctrlpm_init/ctrlpm_s${S}_r20_init.pt
      bash src/gdn_gate/ml7/run_grpo.sh ctrlpm $S 512 r_box --length_on reasoning --alpha 0.2 --steps 150 \
           --answer_max 1280 --probe_effect 0.05 --rank 20 --init_from $WG_RUNS/ml7/grpo/ctrlpm_init/ctrlpm_s${S}_r20_init.pt
    done

The padding receipts are `results/run_configs/4B_rl/ctrlpm_init/*.pad.json`; `python src/gdn_gate/ml7/pad_rank.py
--help` lists the flags.

### 2.5 Test readouts and analysis (4B)

One readout per (checkpoint, test set): 2 samples per item, budgets 256 to 8,192, vendor sampling, the engine-trainer
parity self-test first. The argv of every released readout is inside it (`zcat results/ml7_4b/step14/readout_*.json.gz
| python -c "import json,sys; print(json.load(sys.stdin)['argv'])"`). For one arm checkpoint:

    D=$WG_RUNS/ml7/data
    python src/gdn_gate/ml7/readout_vllm.py --model $M4 --data $D/test_math500.jsonl --arm wide1 \
           --ckpt $WG_RUNS/ml7/grpo/wide1_s0_cap512_r_box/ckpt_step150.pt --caps 256,512,1024,2048,4096,8192 \
           --answer_max 1280 --samples 2 --seed 20261006 --gpu_mem 0.45 --probe_effect 0.05 \
           --out results/runs/readout_math500_wide1_0.json
    # untouched model: --arm stock (no --ckpt); warm-up: --arm ctrl --ckpt <warm-up ckpt_step40.pt>;
    # LoRA-r20: --arm ctrl --rank 20 --ckpt <ctrlpm run ckpt_step150.pt>; GSM8K: --data $D/test_gsm8k.jsonl

Then the analysis, with the slots named `ARM_SEED=readout`:

    python src/gdn_gate/ml7/ctrlpm_analysis.py --split test --seeds 0,1,2 --pm_rank 20 \
           --manifest ml_pipeline_4b/split_manifest.json --touch1 results/ml7_4b/step14/analysis_math500.json \
           --baseline <untouched> --warmup <warm-up> ctrl_0=... wide_0=... wide1_0=... ctrlpm_0=... \
           --out results/runs/analysis_math500.json

`step14_analysis.py` has the same interface for the three-arm records. In the lab a guard (`step14_guard.py`,
`ml_pipeline/guard.py`) counted each read of the test set and the code was frozen by sha256 before each read
(`results/**/code.sha256`, amendments in `src/gdn_gate/ml7/amendments/ledger.json`); the GPU scheduling scripts that
queued these readouts two per GPU are not released.

### 2.6 2B record

Same steps with Qwen3.5-2B (revision 15852e8c), `ml_pipeline/split_manifest.json`, `--answer_max 1792`,
`--probe_effect 0.2` (the default), `MICRO=16384`; seeds 0 to 2 trained on GB10 (run tag `r_gb10`), seeds 3 to 5 on
RTX PRO 6000 (`r_box`); the 2B warm-up and pilots also trained on GB10. Run configs: `results/run_configs/2B_rl/`.
Receipts: `results/ml7/step14` (read 1) and `results/ml7/step14_ext` (read 2, `--seeds 3,4,5`).

### 2.7 Chat SFT (appendix, held-out loss)

Environment `env/sft-fast.txt` (Unsloth stack; seed 0 on GB10, seeds 1 to 5 on RTX PRO 6000); the plain trainer
`src/gdn_gate/broad/train_sft.py` runs the same recipe with the `env/train.txt` environment, and
`src/gdn_gate/unsloth/check_fast.py` compares the two (receipts in `results/unsloth/checks/`). For each arm in ctrl,
wide1, wide and seed 0 to 5:

    B=$WG_HF_CACHE/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b
    O=$WG_RUNS/broad/step6/${ARM}_s${SEED}
    python src/gdn_gate/unsloth/dump_init.py $B $ARM $SEED $O/init_${ARM}_s${SEED}.pt
    python src/gdn_gate/unsloth/train_sft_fast.py --stack unsloth --model $B \
           --data $WG_RUNS/broad/data/tulu3_permissive.parquet --arm $ARM --seed $SEED --out $O \
           --ckpt_steps 300,700,1200 --resume --init_from $O/init_${ARM}_s${SEED}.pt

The held-out loss is the last line of `$O/val_log.jsonl` (step 1,200); `paper/figs/derive.py` pairs the seeds.

### 2.8 Forgetting recipe (appendix)

`src/retrofit/recipe_main_4b.txt` lists one run per line as `gpu name arm lr w_lr steps seed [flags]`; each line
ran as

    python src/retrofit/train.py --model $B --arm $ARM --seed $SEED --steps 1200 --lr 1e-4 --w_lr 1e-4 \
           --eval_every 100 --out $WG_RUNS/retrofit/box/main_${ARM}_s${SEED} \
           --replay $WG_RUNS/retrofit/replay_wikitext103.pt \
           --task_counts parity:16,swap:64,codeswap:64 --sched constant --replay_batch 12 --replay_every 1 \
           --replay_micro 4 --micro_tokens 24576

with `wide1` added the same way (run configs in `results/run_configs/forgetting/`). Scoring and comparison:

    python src/gdn_gate/rescore/rescore_4b.py --run_dir <run dir> --name main_<arm>_s<seed> --out_dir results/runs/rescore4b
    python src/gdn_gate/rescore/rescore_9b.py --model <Qwen3.5-9B-Base dir> --replay_wikitext <p> --replay_clean <p> -- <rescore_4b args>
    python src/gdn_gate/w1main/compare.py --help

HumanEval for the chat SFT seed-0 trio: `src/gdn_gate/vllm/vscore_4b.py` (argv in `results/vllm/step6_s0/*.json`).

### 2.9 Olmo-Hybrid-7B lesions (appendix)

Model `allenai/Olmo-Hybrid-7B` revision 4f1cc566. Environments: `env/train.txt` (HF lesions, NLL), `env/vllm.txt`
(vLLM battery, RULER).

    O7=$WG_HF_CACHE/models--allenai--Olmo-Hybrid-7B/snapshots/4f1cc566f9fdf3ce68da2ab6a788a83d89896dcf
    python src/gdn_gate/olmo/prep_nll.py --help                                    # held-out NLL text
    python src/gdn_gate/olmo/lesion_nll.py --model $O7 --data $WG_RUNS/olmo/nll_heldout_olmo.pt \
           --out $WG_RUNS/olmo/lesion_nll_base --arms stock,half,erase --n_chunks 256 --batch 4
    python src/gdn_gate/olmo/analyze_nll.py --help
    for ARM in stock erase half; do                                                 # broad battery and RULER, 4k to 16k
      python src/gdn_gate/olmo/vbattery_small_gpu.py --model $O7 --gpu_mem 0.90 --out $WG_RUNS/olmo/battery_vllm2 \
             --arm $ARM --groups ruler --ruler_lengths 16384; done
    for ARM in stock erase; do                                                      # counting cells, 500 items
      python src/gdn_gate/olmo/vbattery_small_gpu.py --model $O7 --gpu_mem 0.90 --out $WG_RUNS/olmo/battery_vllm2_cwe \
             --arm $ARM --groups ruler --ruler_tasks ruler_cwe,ruler_fwe --ruler_lengths 16384 --ruler_limit 500; done
    for ARM in stock erase_g0 erase_g1 erase_g2 erase_g3; do                        # depth quarters, CWE at 8k
      python src/gdn_gate/olmo/vbattery_small_gpu.py --model $O7 --gpu_mem 0.90 --out $WG_RUNS/olmo/battery_vllm2_loc \
             --arm $ARM --groups ruler --ruler_tasks ruler_cwe --ruler_lengths 8192 --ruler_limit 500; done
    python src/gdn_gate/olmo/ruler_paired.py $WG_RUNS/olmo/battery_vllm2_cwe results/runs/counting_paired.json --arms erase
    python src/gdn_gate/olmo/usage_probe.py --model $O7 --out results/runs/usage_by_task_8k.json

The paired RULER receipts and the NLL summary also rerun from released per-item data, without a GPU:

    uv run python scripts/rerun_ruler_pairing.py                       # 4 x OK (Olmo RULER, counting, depth, 32k pointer)
    uv run python src/gdn_gate/olmo/analyze_nll.py --run results/olmo --out /tmp/nll.json   # = lesion_nll_base_summary.json

The retrofitted-range counting pointer (Olmo appendix, last paragraph) is `src/gdn_gate/vllm/vcount.py` (argv in the
receipt `results/vcount/step6_32k_paired.json` inputs; per-item scores in `results/vcount/ruler_items/`).

### 2.10 Fresh held-out set (Phase 5)

Pre-registered in `ml_pipeline_4b/PIPELINE.md` ("Fresh test set") before any read: 130 problems from AIME 2024,
AIME 2025, AIME 2026 and AMC 2023 (12A/12B), deduplicated against the RL pool and the validation slice with the 2B
record's sibling scan (0 hits), hypotheses H5 (Gate(0,1) - LoRA area > 0) and H6 (length < 0) at 4B, the 2B rows as a
secondary replication. `scripts/rebuild_data.py` rebuilds `test_fresh.jsonl` byte for byte from the public sources;
the lab built it with `src/gdn_gate/ml7/fresh/build_fresh.py --raw RAW --data $WG_RUNS/ml7/data --guard
ml_pipeline_4b/fresh_test --download`, which also wrote the set's own guard state (`ml_pipeline_4b/fresh_test/`).

One readout per checkpoint, same instrument as the test reads with 8 samples per item; the checkpoints are the step-150
runs (warm-up: step 40) whose sha256 `src/gdn_gate/ml7/fresh/ckpt_expect.json` lists. 4B (14 readouts: untouched,
warm-up, LoRA, LoRA-r20, Gate(0,1), Gate(0,2) x seeds 0 to 2):

    M2=$WG_HF_CACHE/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
    python src/gdn_gate/ml7/readout_vllm.py --model $M4 --data $D/test_fresh.jsonl --arm ctrl \
           --ckpt $WG_RUNS/ml7_4b/grpo/ctrlpm_s0_cap512_r_box/ckpt_step150.pt --caps 256,512,1024,2048,4096,8192 \
           --answer_max 1280 --samples 8 --seed 20261006 --gpu_mem 0.45 --probe_effect 0.05 --rank 20 \
           --out results/runs/fresh/readout_fresh_4b_ctrlpm_0.json
    # 2B (19 readouts: untouched, LoRA, Gate(0,1), Gate(0,2) x seeds 0 to 5): --model $M2 --answer_max 1792 --probe_effect 0.2

The argv of every released readout is inside it (`results/fresh/readout_fresh_*.json.gz`). In the lab every fresh
readout ran alone on its GPU (amendment 06:00 in the record) and `src/gdn_gate/ml7/fresh/fresh_test_box.sh` queued
them under one `fresh_guard.py final` call, which counted the single read of the set. The pre-readout self-test of
two 4B readouts used the visibility bar 4.0 instead of 5.0 (owner-approved amendment, ledger entry and CPU test
`src/gdn_gate/ml7/fresh/test_effect_visible_cpu.py`); scored generation is unchanged. Analysis:

    python src/gdn_gate/ml7/fresh/fresh_analysis.py --record 4b --baseline <untouched> --warmup <warm-up> \
           ctrl_0=... ctrlpm_0=... wide1_0=... wide_0=... --manifest ml_pipeline_4b/fresh_test/split_manifest.json \
           --out results/runs/fresh/analysis_fresh_4b.json
    python src/gdn_gate/ml7/fresh/fresh_table.py results/fresh          # the tables of results/fresh/fresh_table.md

`fresh_analysis.py` imports the frozen `step14_analysis.py` and `ctrlpm_analysis.py` unchanged and sets only their
module constants (test file, id prefix, H5 and H6). The released receipts rerun exactly from the released readouts
(`scripts/rerun_analysis.py`).

### 2.11 Mechanism (validation slice and saved traces)

The causal tests read the GSM8K validation slice only (500 items, 1 sample, the 4B record's readout settings), never
a test set. Inputs, then one job per readout with `src/gdn_gate/mech/mech_box.sh` (it checks every input's sha256):

    python src/gdn_gate/mech/hybrid_ckpts.py $WG_RUNS/mech/hybrid         # G1 "gate only": warm-up LoRA + trained gate
    python src/gdn_gate/mech/dose_ckpts.py $WG_RUNS/mech/dose             # G2: Gate(0,1) gate tensors x 0, 0.5, 1.5, 2
    MECH_HYBRID=$WG_RUNS/mech/hybrid MECH_DOSE=$WG_RUNS/mech/dose \
      bash src/gdn_gate/mech/mech_box.sh g1 wide1_s0_strip                 # labels: <arm>_s<k>_{full,strip,hybrid}, warmup, ctrl_s<k>
    bash src/gdn_gate/mech/mech_box.sh g2 wide1_s0_x0.5                    # labels: wide1_s<k>_x{0,0.5,1.5,2}
    python src/gdn_gate/mech/g1g2_summary.py results/mech                  # g1g2_summary.json (CPU)

G1 conditions: `full` is the trained checkpoint, `strip` the same LoRA without the gate (readout `--strip`, stock
engine), `hybrid` the warm-up's LoRA with the trained gate. The released `results/mech/hybrid_receipt.json` and
`dose_receipt.json` give the sha256 of every input checkpoint. G3 (teacher-forced beta statistics, 43 checkpoints) runs
`mech_box.sh g3 a` and `g3 b`, that is `src/gdn_gate/mech/gate_stats.py` on 64 warm-up validation traces x 2,048
tokens (`build_inputs.py` builds `inputs.json` from the saved traces, which are not released), then
`gs_summary.py results/mech/gate_stats results/mech/gate_stats_summary fixed` on CPU.

CPU analyses of saved data: `text_stats.py` (where the reasoning tokens go; needs the generation traces, not released),
`item_match.py` (item-matched accuracy from the released 4B test readouts), `traj.py` and `traj_extra.py` (training
dynamics from the run logs), `w_stats.py` and `param_drift.py` (gate and LoRA weights; need the checkpoints, published as
adapters). `scripts/rerun_mech.py` reruns `g1g2_summary.py`, `gs_summary.py`, `item_match.py` and `traj_extra.py` from
released files and compares their output with the receipts.

## 3. Check your results

- Analysis rerun on your readouts: put them in one directory and run the analysis script with the slots as in 2.5;
  compare the `effects`, `length_effects` and `preregistered_hypotheses` blocks with the released receipts.
- Numbers in the paper: `check_claims.py` reads the paths in `paper/CLAIMS.md`; point a row at your file to compare.
- A pass on the reference hardware (one RTX PRO 6000 per run, vLLM 0.31.0): the four RL-record hypotheses keep
  their verdicts, H5 and H6 stay supported at 4B, and the seed means fall inside the observed seed ranges. On other GPUs, compare orderings and verdicts,
  not values.

## 4. What differs or cannot be shipped

- **Code edits for publication.** Internal names and absolute paths were replaced, path defaults moved to `WG_RUNS`
  and `WG_HF_CACHE`, and the gate tensor was renamed `gate_w`. The sha256 values in `code.sha256` files and readout
  env blocks refer to the lab files; `results/SCRUB_LEDGER.json` maps each lab file hash to its published hash.
- **Receipts.** Paths and host names inside the receipts were replaced the same way; values are untouched.
  `rerun_analysis.py` shows the analyses still reproduce from the published readouts.
- **Not released:** the raw generation traces (`*.traces.jsonl.gz`, about 830 MB; planned as a separate archive), the
  GPU scheduling, provisioning and transfer scripts of the rented servers (including `box/orch.py`, which
  `fresh_test_box.sh` and `mech_box.sh` call), and the lab notebook. The released
  `run_configs`, readout argv and receipts carry every argument those scripts passed.
- **Weights.** The 65 adapters the paper reads are on the Hugging Face Hub as safetensors, revisions at the top of
  this file (docs/ADAPTERS.md); the intermediate checkpoints are not published.
- **Fresh set and mechanism.** The fresh set's problem text is not redistributed (rebuild it with
  `scripts/rebuild_data.py`). `results/mech/` omits the lab's one paired example trace (generated text). Two outputs
  the lab computed inline have release scripts that reproduce them: `src/gdn_gate/mech/hybrid_ckpts.py` (same sha256
  as the lab's hybrid checkpoints) and `traj_extra.py` (same JSON). The lab's driver for running G1 on the GB10 units
  (`ablate_spark.sh`) is not released; the G1 readouts in `results/mech/` ran on the rented server through
  `mech_box.sh`.
- **Records.** Rental prices in `ml_pipeline_4b/PIPELINE.md` are replaced by `<price>`; docs/PREREGISTRATION.md
  lists every edited line.
- **Nondeterminism.** Readouts and rollouts sample; two readouts shared each GPU in the lab. vLLM reproduces outputs
  only on the same hardware and version. The analysis bootstrap uses a fixed seed (20261009) and is exact.
- **Hardware.** Training of the 2B warm-up, pilots and seeds 0 to 2, the chat SFT seed 0 runs and the 4B forgetting
  LoRA seeds 4 and 5 used DGX Spark (GB10) units; the 9B forgetting runs, the 4B forgetting runs of seeds 0 to 2 and
  chat SFT seeds 1 to 5 trained on rented RTX PRO 6000 GPUs. The Olmo lesions, the 2B validation readouts and the
  HumanEval scoring of the 4B LoRA and Gate(0,2) forgetting runs used an RTX 5090 Laptop GPU (24 GB); the 9B
  forgetting runs were scored on GB10 (LoRA seed 1 on an RTX PRO 6000). Hardware per receipt is in its env block
  (`results/ENVIRONMENT.json` for the RL readouts).
- **Quick mode.** There is no shrunk GPU workload with its own reference values. `readout_vllm.py --limit N
  --stride K` reads a subset for a smoke test; the CPU level above is the fast path.
