# Claims ledger

Every number in the paper maps to a receipt. `python3 paper/figs/check_claims.py` parses the tables below and checks
each `json` row (value at the path, times the scale, within the tolerance of the expected value), each `text` row (the
quoted string occurs in the file) and each `textfull` row (the whole file equals the string). It then scans every number
in the body of `paper/main.tex` and fails on any number that matches no verified row value and no entry of its
allowlist; the paper source is not in this repository, so that scan is skipped here. Paths are relative to the repository root unless absolute; JSON paths use `/` between
keys and list indices. `D` = `paper/data/derived.json`, written by `paper/figs/derive.py` from run logs and receipts
(each value in it names its own source file).

Arm names: paper `LoRA` = receipt `ctrl`; `LoRA-r20` = `ctrlpm`; `Gate(0,1)` = `wide1`; `Gate(0,2)` = `wide`.
Shorthands for files:

- `A4M` = results/ml7_4b/step14_ctrlpm/analysis_math500.json (4B test, all arms; reproduces step14/analysis_math500.json on ctrl/wide/wide1, `ctrlpm/touch1_check/equal_on_original_arms` = true)
- `A4G` = results/ml7_4b/step14_ctrlpm/analysis_gsm8k.json
- `A2M`, `A2G` = results/ml7/step14/analysis_{math500,gsm8k}.json (2B read 1, seeds 0-2)
- `AXM`, `AXG` = results/ml7/step14_ext/analysis_{math500,gsm8k}.json (2B read 2, seeds 3-5)
- `OCP` = results/olmo/vllm2/counting_paired.json; `OLOC` = .../counting_localization.json; `ORP` = .../ruler_paired.json; `ONLL` = results/olmo/lesion_nll_base_summary.json
- `VC32` = results/vcount/step6_32k_paired.json; `W1` = results/w1main/compare.json
- `F4`, `F2` = results/fresh/analysis_fresh_{4b,2b}.json (fresh held-out set, read once); `FFIN` = results/fresh/final.json; `PL4` = ml_pipeline_4b/PIPELINE.md (fresh-set pre-registration and amendments)
- `X` = paper/data/extra.json (paper/figs/stats_extra.py: seed-level t and hierarchical intervals, censoring-aware length, item splits, G1 split, ratios, Olmo MDE, all from per-sample readout rows); `CH` = paper/data/compute_hours.json (paper/figs/compute_hours.py); `PL2` = ml_pipeline/PIPELINE.md; `LEDG` = src/gdn_gate/ml7/amendments/ledger.json; `FIND` = results/FINDINGS.md
- `G12` = results/mech/g1g2_summary.json (G1 ablation, G2 dose); `GS` = results/mech/gate_stats_summary/gate_stats_summary_fixed.json (G3); `TS` = results/mech/text_stats.json; `IM` = .../mech/item_match.json; `TJ` = .../mech/traj.json; `WS` = .../mech/w_stats.json
- Mechanism and fresh-set values that combine several fields are computed by `paper/figs/derive.py` into `D` under `mech/` and `fresh/` (seed ranges, ratios, window means)

## Parameters, data, protocol

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| P1 | LoRA 8,675,328 params | json | A4M | ctrlpm/trainable_params/ctrl | 1 | 8675328 | 0 |
| P2 | Gate(0,1)/(0,2) 10,641,408 | json | A4M | ctrlpm/trainable_params/wide1 | 1 | 10641408 | 0 |
| P3 | LoRA-r20 10,844,160 | json | A4M | ctrlpm/trainable_params/ctrlpm | 1 | 10844160 | 0 |
| P4 | rank 19 -3.2%, rank 21 +7.0%, +1.91% | text | ml_pipeline_4b/PIPELINE.md | rank 19 gives -3.19%, rank 21 +7.0% | | | |
| P5 | 9,284 prompts render identically | json | results/ml7_4b/template_identity.json | 4B_identical_thinking_prompts | 1 | 9284 | 0 |
| P6 | 6,965 RL prompts | json | ml_pipeline/split_manifest.json | counts/rl_pool.jsonl | 1 | 6965 | 0 |
| P7 | 500 validation items | json | ml_pipeline/split_manifest.json | counts/val.jsonl | 1 | 500 | 0 |
| P8 | GSM8K test 1,319 | json | ml_pipeline/split_manifest.json | counts/test_gsm8k.jsonl | 1 | 1319 | 0 |
| P9 | 8 template siblings removed (4 + 4) | text | ml_pipeline/PIPELINE.md | 4 (train 1314, 5162, 20, 3726 | | | |
| P10 | 84,077 grader checks | json | ml_pipeline/grader_check.json | n_checks | 1 | 84077 | 0 |
| P11 | grader accepts every correct form | json | ml_pipeline/grader_check.json | accept_variants_math_verify | 1 | 1.0 | 0 |
| P12 | grader rejects every wrong form | json | ml_pipeline/grader_check.json | red_accept_math_verify | 1 | 0.0 | 0 |
| P13 | answer cap 1,792 at 2B | json | results/ml7/answer_cap.json | answer_max | 1 | 1792 | 0 |
| P14 | answer cap 1,280 at 4B | json | results/ml7_4b/answer_cap.json | answer_max | 1 | 1280 | 0 |
| P15 | training cap 512 at 4B (the whole CAP file is 512) | textfull | results/ml7_4b/CAP | 512 | | | |
| P16 | training cap 512 at 2B | json | results/ml7/pilot_pick_phase2.json | cap | 1 | 512 | 0 |
| P17 | 2B amendment: 2.03x vs 2.0, raised to 2.5 | text | results/ml7/step14/code_lock_amendment.txt | decode-path check 2.03x floor vs 2.0x limit | | | |
| P18 | probe 0.05 for every 4B run | text | ml_pipeline_4b/PIPELINE.md | Parity probe 0.05 for this record | | | |
| P19 | test touches 2 in each record | json | ml_pipeline_4b/.guard_state.json | test_touches | 1 | 2 | 0 |
| P20 | test touches 2 (2B record) | json | ml_pipeline/.guard_state.json | test_touches | 1 | 2 | 0 |
| P21 | read-2 analysis reproduces read 1 | json | A4M | ctrlpm/touch1_check/equal_on_original_arms | 1 | 1 | 0 |
| P22 | bootstrap 10,000 draws, items resampled, seeds paired | text | A4M | 10000 draws, numpy default_rng(20261009), items resampled with replacement, seeds paired by index | | | |
| P23 | reward 1 - 0.2 sigmoid(z) over correct samples | text | src/gdn_gate/broad/rl_common.py | r = 1[correct] * (1 - alpha * sigmoid((len - mean_c) / std_c)) | | | |
| P24 | recipe: LoRA r16 alpha 32, lr 2e-5, 16x8, 150 steps, Dr. GRPO, no KL | text | ml_pipeline_4b/PIPELINE.md | LoRA r16 (alpha 32) on the GDN projections | | | |
| P25 | warm-up 40 steps, correctness only | text | ml_pipeline/PIPELINE.md | correctness reward only (alpha 0), cap 512, answer_max 1792, 40 steps | | | |
| P26 | gate formula and zero init | text | src/gdn_gate/broad/arms.py | W zero-initialised (fp32, hidden -> num_v_heads) | | | |
| P27 | LoRA-r20 widened init, step 0 = warm-up | text | ml_pipeline_4b/PIPELINE.md | the 4 new B columns exactly zero | | | |
| P28 | vendor sampling T 1.0, top-p 0.95, top-k 20, presence 1.5 | text | ml_pipeline_4b/PIPELINE.md | temperature 1.0, top_p 0.95, top_k 20, presence 1.5 | | | |
| P29 | vLLM 0.31.0 in every readout | text | results/ENVIRONMENT.json | "vllm": "0.31.0" | | | |
| P29b | torch 2.13.0+cu130 in the readouts | text | results/ENVIRONMENT.json | "torch": "2.13.0+cu130" | | | |
| P29c | transformers 5.17.0 | text | results/ENVIRONMENT.json | "transformers": "5.17.0" | | | |
| P30 | 4B readouts on RTX PRO 6000 Server Edition, analysis refuses mixed GPUs | text | ml_pipeline_4b/PIPELINE.md | the analysis refuses mixed GPUs | | | |
| P31 | rented compute ~173 GPU-h: 8 GPUs 9.5 h | text | docs/COMPUTE.md | 8 x RTX PRO 6000, 9.5 h | | | |
| P31b | ... preemptible leg 01:40-05:15 | text | docs/COMPUTE.md | preemptible server from 01:40 to 05:15 | | | |
| P32 | 8 GPUs 10.3 h | text | docs/COMPUTE.md | 8 x RTX PRO 6000, 10.3 h | | | |
| P33 | 4 GPUs 3.7 h | text | docs/COMPUTE.md | 4 x RTX PRO 6000, 3.7 h | | | |

## 4B RL (Table 2, Section 5.1)

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| R1 | MATH area untouched 0.699 | json | A4M | baseline/norm_auc_log2 | 1 | 0.699 | 0.0005 |
| R2 | MATH area warm-up 0.729 | json | A4M | warmup/norm_auc_log2 | 1 | 0.729 | 0.0005 |
| R3 | MATH area LoRA 0.786 | json | A4M | arms/ctrl/norm_auc_log2/seed_mean | 1 | 0.786 | 0.0005 |
| R4 | MATH area LoRA-r20 0.792 | json | A4M | arms/ctrlpm/norm_auc_log2/seed_mean | 1 | 0.792 | 0.0005 |
| R5 | MATH area Gate(0,1) 0.852 | json | A4M | arms/wide1/norm_auc_log2/seed_mean | 1 | 0.852 | 0.0005 |
| R6 | MATH area Gate(0,2) 0.849 | json | A4M | arms/wide/norm_auc_log2/seed_mean | 1 | 0.849 | 0.0005 |
| R7 | MATH acc@256 untouched 0.596 | json | A4M | baseline/acc/256 | 1 | 0.596 | 0.0005 |
| R8 | MATH acc@256 LoRA 0.695 | json | A4M | arms/ctrl/acc/256/seed_mean | 1 | 0.695 | 0.0005 |
| R9 | MATH acc@256 LoRA-r20 0.710 | json | A4M | arms/ctrlpm/acc/256/seed_mean | 1 | 0.710 | 0.0005 |
| R10 | MATH acc@256 Gate(0,1) 0.776 | json | A4M | arms/wide1/acc/256/seed_mean | 1 | 0.776 | 0.0005 |
| R11 | MATH acc@256 Gate(0,2) 0.778 | json | A4M | arms/wide/acc/256/seed_mean | 1 | 0.778 | 0.0005 |
| R12 | MATH acc@8k untouched 0.824 | json | A4M | baseline/acc/8192 | 1 | 0.824 | 0.0005 |
| R13 | MATH acc@8k warm-up 0.853 | json | A4M | warmup/acc/8192 | 1 | 0.853 | 0.0005 |
| R14 | MATH acc@8k LoRA 0.889 | json | A4M | arms/ctrl/acc/8192/seed_mean | 1 | 0.889 | 0.0005 |
| R15 | MATH acc@8k LoRA-r20 0.895 | json | A4M | arms/ctrlpm/acc/8192/seed_mean | 1 | 0.895 | 0.0005 |
| R16 | MATH acc@8k Gate(0,1) 0.937 | json | A4M | arms/wide1/acc/8192/seed_mean | 1 | 0.937 | 0.0005 |
| R17 | MATH acc@8k Gate(0,2) 0.928 | json | A4M | arms/wide/acc/8192/seed_mean | 1 | 0.928 | 0.0005 |
| R18 | MATH length untouched 5,282 | json | A4M | baseline/natural_len | 1 | 5282 | 0.5 |
| R19 | MATH length warm-up 5,022 | json | A4M | warmup/natural_len | 1 | 5022 | 0.5 |
| R20 | MATH length LoRA 4,223 | json | A4M | arms/ctrl/natural_len/seed_mean | 1 | 4223 | 0.5 |
| R21 | MATH length LoRA-r20 4,003 | json | A4M | arms/ctrlpm/natural_len/seed_mean | 1 | 4003 | 0.5 |
| R22 | MATH length Gate(0,1) 2,202 | json | A4M | arms/wide1/natural_len/seed_mean | 1 | 2202 | 0.5 |
| R23 | MATH length Gate(0,2) 2,153 | json | A4M | arms/wide/natural_len/seed_mean | 1 | 2153 | 0.5 |
| R24 | MATH correct-only untouched 4,676 | json | A4M | baseline/cond_length/natural/correct/mean | 1 | 4676 | 0.5 |
| R25 | MATH correct-only warm-up 4,490 | json | A4M | warmup/cond_length/natural/correct/mean | 1 | 4490 | 0.5 |
| R26 | MATH correct-only LoRA 3,752 | json | D | cond_length/4B_math500/rows/ctrl/correct_mean_seed_mean | 1 | 3752 | 0.5 |
| R27 | MATH correct-only LoRA-r20 3,533 | json | D | cond_length/4B_math500/rows/ctrlpm/correct_mean_seed_mean | 1 | 3533 | 0.5 |
| R28 | MATH correct-only Gate(0,1) 1,901 | json | D | cond_length/4B_math500/rows/wide1/correct_mean_seed_mean | 1 | 1901 | 0.5 |
| R29 | MATH correct-only Gate(0,2) 1,849 | json | D | cond_length/4B_math500/rows/wide/correct_mean_seed_mean | 1 | 1849 | 0.5 |
| R30 | MATH open untouched 0.363 | json | A4M | baseline/cut_off_share_at_max | 1 | 0.363 | 0.0005 |
| R31 | MATH open warm-up 0.330 | json | A4M | warmup/cut_off_share_at_max | 1 | 0.330 | 0.0005 |
| R32 | MATH open LoRA 0.249 | json | A4M | arms/ctrl/cut_off_share_at_max/seed_mean | 1 | 0.249 | 0.0005 |
| R33 | MATH open LoRA-r20 0.232 | json | A4M | arms/ctrlpm/cut_off_share_at_max/seed_mean | 1 | 0.232 | 0.0005 |
| R34 | MATH open Gate(0,1) 0.098 | json | A4M | arms/wide1/cut_off_share_at_max/seed_mean | 1 | 0.098 | 0.0005 |
| R35 | MATH open Gate(0,2) 0.103 | json | A4M | arms/wide/cut_off_share_at_max/seed_mean | 1 | 0.103 | 0.0005 |
| R36 | GSM8K area untouched 0.912 | json | A4G | baseline/norm_auc_log2 | 1 | 0.912 | 0.0006 |
| R37 | GSM8K area warm-up 0.921 | json | A4G | warmup/norm_auc_log2 | 1 | 0.921 | 0.0005 |
| R38 | GSM8K area LoRA 0.933 | json | A4G | arms/ctrl/norm_auc_log2/seed_mean | 1 | 0.933 | 0.0005 |
| R39 | GSM8K area LoRA-r20 0.939 | json | A4G | arms/ctrlpm/norm_auc_log2/seed_mean | 1 | 0.939 | 0.0005 |
| R40 | GSM8K area Gate(0,1) 0.940 | json | A4G | arms/wide1/norm_auc_log2/seed_mean | 1 | 0.940 | 0.0006 |
| R41 | GSM8K area Gate(0,2) 0.941 | json | A4G | arms/wide/norm_auc_log2/seed_mean | 1 | 0.941 | 0.0005 |
| R42 | GSM8K acc@256 untouched 0.880 | json | A4G | baseline/acc/256 | 1 | 0.880 | 0.0005 |
| R43 | GSM8K acc@256 warm-up 0.900 | json | A4G | warmup/acc/256 | 1 | 0.900 | 0.0005 |
| R44 | GSM8K acc@256 LoRA 0.923 | json | A4G | arms/ctrl/acc/256/seed_mean | 1 | 0.923 | 0.0005 |
| R45 | GSM8K acc@256 LoRA-r20 0.928 | json | A4G | arms/ctrlpm/acc/256/seed_mean | 1 | 0.928 | 0.0005 |
| R46 | GSM8K acc@256 Gate(0,1) 0.929 | json | A4G | arms/wide1/acc/256/seed_mean | 1 | 0.929 | 0.0005 |
| R47 | GSM8K acc@256 Gate(0,2) 0.930 | json | A4G | arms/wide/acc/256/seed_mean | 1 | 0.930 | 0.0005 |
| R48 | GSM8K acc@8k untouched 0.933 | json | A4G | baseline/acc/8192 | 1 | 0.933 | 0.0005 |
| R49 | GSM8K acc@8k warm-up 0.938 | json | A4G | warmup/acc/8192 | 1 | 0.938 | 0.0005 |
| R50 | GSM8K acc@8k LoRA 0.941 | json | A4G | arms/ctrl/acc/8192/seed_mean | 1 | 0.941 | 0.0005 |
| R51 | GSM8K acc@8k LoRA-r20 0.945 | json | A4G | arms/ctrlpm/acc/8192/seed_mean | 1 | 0.945 | 0.0005 |
| R52 | GSM8K acc@8k Gate(0,1) 0.943 | json | A4G | arms/wide1/acc/8192/seed_mean | 1 | 0.943 | 0.0005 |
| R53 | GSM8K acc@8k Gate(0,2) 0.944 | json | A4G | arms/wide/acc/8192/seed_mean | 1 | 0.944 | 0.0005 |
| R54 | GSM8K length untouched 3,271 | json | A4G | baseline/natural_len | 1 | 3271 | 0.5 |
| R55 | GSM8K length warm-up 2,858 | json | A4G | warmup/natural_len | 1 | 2858 | 0.5 |
| R56 | GSM8K length LoRA 1,666 | json | A4G | arms/ctrl/natural_len/seed_mean | 1 | 1666 | 0.5 |
| R57 | GSM8K length LoRA-r20 1,296 | json | A4G | arms/ctrlpm/natural_len/seed_mean | 1 | 1296 | 0.5 |
| R58 | GSM8K length Gate(0,1) 297 | json | A4G | arms/wide1/natural_len/seed_mean | 1 | 297 | 0.5 |
| R59 | GSM8K length Gate(0,2) 261 | json | A4G | arms/wide/natural_len/seed_mean | 1 | 261 | 0.5 |
| R60 | GSM8K correct-only untouched 3,064 | json | A4G | baseline/cond_length/natural/correct/mean | 1 | 3064 | 0.5 |
| R61 | GSM8K correct-only warm-up 2,679 | json | A4G | warmup/cond_length/natural/correct/mean | 1 | 2679 | 0.5 |
| R62 | GSM8K correct-only LoRA 1,535 | json | D | cond_length/4B_gsm8k/rows/ctrl/correct_mean_seed_mean | 1 | 1535 | 0.5 |
| R63 | GSM8K correct-only LoRA-r20 1,202 | json | D | cond_length/4B_gsm8k/rows/ctrlpm/correct_mean_seed_mean | 1 | 1202 | 0.5 |
| R64 | GSM8K correct-only Gate(0,1) 271 | json | D | cond_length/4B_gsm8k/rows/wide1/correct_mean_seed_mean | 1 | 271 | 0.5 |
| R65 | GSM8K correct-only Gate(0,2) 232 | json | D | cond_length/4B_gsm8k/rows/wide/correct_mean_seed_mean | 1 | 232 | 0.5 |
| R66 | GSM8K open untouched 0.095 | json | A4G | baseline/cut_off_share_at_max | 1 | 0.095 | 0.0005 |
| R67 | GSM8K open warm-up 0.063 | json | A4G | warmup/cut_off_share_at_max | 1 | 0.063 | 0.0005 |
| R68 | GSM8K open LoRA 0.016 (1.6%) | json | A4G | arms/ctrl/cut_off_share_at_max/seed_mean | 1 | 0.016 | 0.0005 |
| R69 | GSM8K open LoRA-r20 0.008 | json | A4G | arms/ctrlpm/cut_off_share_at_max/seed_mean | 1 | 0.008 | 0.0005 |
| R70 | GSM8K open Gate(0,1) 0.000 | json | A4G | arms/wide1/cut_off_share_at_max/seed_mean | 1 | 0.000 | 0.0005 |
| R71 | GSM8K open Gate(0,2) 0.001 | json | A4G | arms/wide/cut_off_share_at_max/seed_mean | 1 | 0.001 | 0.0005 |
| R72 | budget-to-match target MATH 0.824 | json | A4M | arms/ctrl/tokens_to_base_max_acc/target | 1 | 0.824 | 0.0005 |
| R73 | budget-to-match target GSM8K 0.933 | json | A4G | arms/ctrl/tokens_to_base_max_acc/target | 1 | 0.933 | 0.0005 |
| R74 | MATH budget LoRA 2,590 | json | A4M | arms/ctrl/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 2590 | 0.5 |
| R75 | MATH budget LoRA-r20 2,600 | json | A4M | arms/ctrlpm/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 2600 | 0.5 |
| R76 | MATH budget Gate(0,1) 1,024 | json | A4M | arms/wide1/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 1024 | 0.5 |
| R77 | MATH budget Gate(0,2) 882 | json | A4M | arms/wide/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 882 | 0.5 |
| R78 | GSM8K budget LoRA 1,390 | json | A4G | arms/ctrl/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 1390 | 0.5 |
| R79 | GSM8K budget LoRA-r20 494 | json | A4G | arms/ctrlpm/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 494 | 0.5 |
| R80 | GSM8K budget Gate(0,1) 390 | json | A4G | arms/wide1/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 390 | 0.5 |
| R81 | GSM8K budget Gate(0,2) 323 | json | A4G | arms/wide/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 323 | 0.5 |
| R82 | Gate(0,2)-LoRA MATH area +0.063 | json | A4M | effects/wide - ctrl/seed_mean_diff | 1 | 0.063 | 0.0005 |
| R83 | ... interval lower +0.048 | json | A4M | effects/wide - ctrl/ci95/0 | 1 | 0.048 | 0.0005 |
| R84 | ... interval upper +0.078 | json | A4M | effects/wide - ctrl/ci95/1 | 1 | 0.078 | 0.0005 |
| R85 | ... seeds +0.080 | json | A4M | effects/wide - ctrl/per_seed_diff/0 | 1 | 0.080 | 0.0005 |
| R86 | ... seeds +0.045 | json | A4M | effects/wide - ctrl/per_seed_diff/1 | 1 | 0.045 | 0.0005 |
| R87 | ... seeds +0.064 | json | A4M | effects/wide - ctrl/per_seed_diff/2 | 1 | 0.064 | 0.0005 |
| R88 | Gate(0,1)-LoRA MATH area +0.065 | json | A4M | effects/wide1 - ctrl/seed_mean_diff | 1 | 0.065 | 0.0005 |
| R89 | ... interval [+0.053, +0.078] | json | A4M | effects/wide1 - ctrl/ci95/0 | 1 | 0.053 | 0.0005 |
| R90 | ... upper +0.078 | json | A4M | effects/wide1 - ctrl/ci95/1 | 1 | 0.078 | 0.0005 |
| R91 | H1 supported (4B) | text | A4M | "verdict": "supported" | | | |
| R92 | H2: Gate(0,2)-LoRA GSM8K length -1,405 | json | A4G | length_effects/wide - ctrl/seed_mean_diff | 1 | -1405 | 0.5 |
| R93 | ... interval lower -1,468 | json | A4G | length_effects/wide - ctrl/ci95/0 | 1 | -1468 | 0.5 |
| R94 | ... interval upper -1,344 | json | A4G | length_effects/wide - ctrl/ci95/1 | 1 | -1344 | 0.5 |
| R95 | ... seeds -1,731 / -1,282 / -1,202 | json | A4G | length_effects/wide - ctrl/per_seed_diff/0 | 1 | -1731 | 0.5 |
| R96 | ... seed 1 -1,282 | json | A4G | length_effects/wide - ctrl/per_seed_diff/1 | 1 | -1282 | 0.5 |
| R97 | ... seed 2 -1,202 | json | A4G | length_effects/wide - ctrl/per_seed_diff/2 | 1 | -1202 | 0.5 |
| R98 | H3: Gate(0,1)-LoRA-r20 MATH area +0.060 | json | A4M | preregistered_hypotheses/H3/seed_mean_diff | 1 | 0.060 | 0.0005 |
| R99 | ... interval [+0.049, +0.072] | json | A4M | preregistered_hypotheses/H3/ci95/0 | 1 | 0.049 | 0.0005 |
| R100 | ... upper | json | A4M | preregistered_hypotheses/H3/ci95/1 | 1 | 0.072 | 0.0005 |
| R101 | ... seeds +0.085 / +0.056 / +0.040 | json | A4M | preregistered_hypotheses/H3/per_seed_diff/0 | 1 | 0.085 | 0.0005 |
| R102 | ... seed 1 | json | A4M | preregistered_hypotheses/H3/per_seed_diff/1 | 1 | 0.056 | 0.0005 |
| R103 | ... seed 2 | json | A4M | preregistered_hypotheses/H3/per_seed_diff/2 | 1 | 0.040 | 0.0005 |
| R104 | H4: Gate(0,1)-LoRA-r20 GSM8K length -1,000 | json | A4G | length_effects/wide1 - ctrlpm/seed_mean_diff | 1 | -1000 | 0.5 |
| R105 | ... interval [-1,043, -958] | json | A4G | length_effects/wide1 - ctrlpm/ci95/0 | 1 | -1043 | 0.5 |
| R106 | ... upper | json | A4G | length_effects/wide1 - ctrlpm/ci95/1 | 1 | -958 | 0.5 |
| R107 | ... seeds -1,757 / -725 / -516 | json | A4G | length_effects/wide1 - ctrlpm/per_seed_diff/0 | 1 | -1757 | 0.5 |
| R108 | ... seed 1 | json | A4G | length_effects/wide1 - ctrlpm/per_seed_diff/1 | 1 | -725 | 0.5 |
| R109 | ... seed 2 | json | A4G | length_effects/wide1 - ctrlpm/per_seed_diff/2 | 1 | -516 | 0.5 |
| R110 | LoRA-r20 - LoRA GSM8K area +0.006, effect | json | A4G | effects/ctrlpm - ctrl/seed_mean_diff | 1 | 0.006 | 0.0005 |
| R111 | ... is an effect | json | A4G | effects/ctrlpm - ctrl/effect | 1 | 1 | 0 |
| R112 | LoRA-r20 - LoRA MATH area: no effect | json | A4M | effects/ctrlpm - ctrl/effect | 1 | 0 | 0 |
| R113 | LoRA-r20 - LoRA GSM8K length: no effect | json | A4G | length_effects/ctrlpm - ctrl/effect | 1 | 0 | 0 |
| R114 | unseen 450: Gate(0,1)-LoRA-r20 +0.060 | json | A4M | math500_unseen450/effects/wide1 - ctrlpm/seed_mean_diff | 1 | 0.060 | 0.0005 |
| R115 | ... interval [+0.048, +0.073] | json | A4M | math500_unseen450/effects/wide1 - ctrlpm/ci95/0 | 1 | 0.048 | 0.0005 |
| R116 | ... upper | json | A4M | math500_unseen450/effects/wide1 - ctrlpm/ci95/1 | 1 | 0.073 | 0.0005 |
| R117 | Gate(0,2)-Gate(0,1) MATH area -0.002 no effect | json | A4M | effects/wide - wide1/seed_mean_diff | 1 | -0.002 | 0.0005 |
| R118 | Gate(0,2)-Gate(0,1) GSM8K length -35 | json | A4G | length_effects/wide - wide1/seed_mean_diff | 1 | -35 | 0.5 |
| R119 | ... seeds -95 / -45 / +35 | json | A4G | length_effects/wide - wide1/per_seed_diff/0 | 1 | -95 | 0.5 |
| R120 | ... seed 2 +35 | json | A4G | length_effects/wide - wide1/per_seed_diff/2 | 1 | 35 | 0.5 |
| R121 | training: LoRA hits cap on 80% to 94% | json | D | train_signature_last20/4B/ctrl_s2/forced_frac | 1 | 0.80 | 0.005 |
| R122 | ... LoRA s0 94% | json | D | train_signature_last20/4B/ctrl_s0/forced_frac | 1 | 0.94 | 0.005 |
| R123 | ... LoRA-r20 57% | json | D | train_signature_last20/4B/ctrlpm_s2/forced_frac | 1 | 0.57 | 0.005 |
| R124 | ... LoRA-r20 96% | json | D | train_signature_last20/4B/ctrlpm_s0/forced_frac | 1 | 0.96 | 0.005 |
| R125 | ... gated 5% (Gate(0,2) s0) | json | D | train_signature_last20/4B/wide_s0/forced_frac | 1 | 0.05 | 0.005 |
| R126 | ... gated 14% (Gate(0,1) s1) | json | D | train_signature_last20/4B/wide1_s1/forced_frac | 1 | 0.14 | 0.005 |
| R127 | ... training reasoning 181 tokens (Gate(0,2) s0) | json | D | train_signature_last20/4B/wide_s0/reasoning_mean | 1 | 181 | 0.5 |
| R128 | ... training reasoning 306 tokens (Gate(0,1) s1) | json | D | train_signature_last20/4B/wide1_s1/reasoning_mean | 1 | 306 | 0.5 |

## 2B RL (Section 5.2, 5.3)

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| B1 | read 1 Gate(0,2)-LoRA MATH area +0.027 | json | A2M | effects/wide - ctrl/seed_mean_diff | 1 | 0.027 | 0.0005 |
| B2 | ... interval [+0.019, +0.035] | json | A2M | effects/wide - ctrl/ci95/0 | 1 | 0.019 | 0.0005 |
| B3 | ... upper | json | A2M | effects/wide - ctrl/ci95/1 | 1 | 0.035 | 0.0005 |
| B4 | ... seeds +0.025 / +0.005 / +0.051 | json | A2M | effects/wide - ctrl/per_seed_diff/1 | 1 | 0.005 | 0.0005 |
| B5 | read 1 Gate(0,1)-LoRA +0.021, seed 0 -0.001, no effect | json | A2M | effects/wide1 - ctrl/per_seed_diff/0 | 1 | -0.001 | 0.0005 |
| B6 | ... mean +0.021 | json | A2M | effects/wide1 - ctrl/seed_mean_diff | 1 | 0.021 | 0.0005 |
| B7 | ... interval [+0.013, +0.028] | json | A2M | effects/wide1 - ctrl/ci95/0 | 1 | 0.013 | 0.0005 |
| B8 | MATH length LoRA 4,830 | json | A2M | arms/ctrl/natural_len/seed_mean | 1 | 4830 | 0.5 |
| B9 | MATH length Gate(0,1) 4,392 | json | A2M | arms/wide1/natural_len/seed_mean | 1 | 4392 | 0.5 |
| B10 | MATH length Gate(0,2) 3,879 | json | A2M | arms/wide/natural_len/seed_mean | 1 | 3879 | 0.5 |
| B11 | GSM8K length LoRA 2,022 | json | A2G | arms/ctrl/natural_len/seed_mean | 1 | 2022 | 0.5 |
| B12 | GSM8K length Gate(0,1) 1,476 | json | A2G | arms/wide1/natural_len/seed_mean | 1 | 1476 | 0.5 |
| B13 | GSM8K length Gate(0,2) 1,025 | json | A2G | arms/wide/natural_len/seed_mean | 1 | 1025 | 0.5 |
| B14 | GSM8K area does not move (no effect) | json | A2G | effects/wide - ctrl/effect | 1 | 0 | 0 |
| B15 | read 2 H1 +0.025 | json | AXM | effects/wide - ctrl/seed_mean_diff | 1 | 0.025 | 0.0005 |
| B16 | ... interval [+0.016, +0.034] | json | AXM | effects/wide - ctrl/ci95/0 | 1 | 0.016 | 0.0005 |
| B17 | ... upper | json | AXM | effects/wide - ctrl/ci95/1 | 1 | 0.034 | 0.0005 |
| B18 | read 2 H2 -997 | json | AXG | length_effects/wide - ctrl/seed_mean_diff | 1 | -997 | 0.5 |
| B19 | ... interval [-1,043, -953] | json | AXG | length_effects/wide - ctrl/ci95/0 | 1 | -1043 | 0.5 |
| B20 | ... upper | json | AXG | length_effects/wide - ctrl/ci95/1 | 1 | -953 | 0.5 |
| B21 | read 2 Gate(0,1)-LoRA MATH +0.021 effect | json | AXM | effects/wide1 - ctrl/seed_mean_diff | 1 | 0.021 | 0.0005 |
| B22 | read 2 Gate(0,1)-LoRA GSM8K length -676 | json | AXG | length_effects/wide1 - ctrl/seed_mean_diff | 1 | -676 | 0.5 |
| B23 | range: 2B read 1 Gate(0,2)-Gate(0,1) MATH +0.006 | json | A2M | effects/wide - wide1/seed_mean_diff | 1 | 0.006 | 0.0005 |
| B24 | range: 2B read 2 +0.003 | json | AXM | effects/wide - wide1/seed_mean_diff | 1 | 0.003 | 0.0005 |
| B25 | range: 2B read 2 GSM8K length -321 | json | AXG | length_effects/wide - wide1/seed_mean_diff | 1 | -321 | 0.5 |
| B26 | ... seeds +113 / +12 / -1,088 | json | AXG | length_effects/wide - wide1/per_seed_diff/2 | 1 | -1088 | 0.5 |
| B27 | 4B Gate(0,1)-LoRA GSM8K length -1,370 | json | A4G | length_effects/wide1 - ctrl/seed_mean_diff | 1 | -1370 | 0.5 |

Derived ratios (now also checked by rows V239-V242 on paper/data/extra.json): 1,025 / 2,022 = 0.507, so 49% shorter; 4B 261 / 1,666 = 0.157, so 84% shorter; 4B Gate(0,1) 297 / 1,666, 82%;
MATH-500 2,202 / 4,223 = 0.521 (48%) and 2,153 / 4,223 = 0.510 (49%). These ratios are computed from rows R20-R23,
R56-R59, B11, B13. The abstract's CWE relative losses (5%, 12%, 20%) are O24/O25, O26/O27 and O28 over the 16k stock
0.305 (OCP rows/4/stock). The 6.8-point sum of the quarter costs is O35+O36+0.68+O38 (OLOC macro/2/diff = -0.0068).

## SFT and forgetting (Section 7)

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| S1 | Gate(0,1)-LoRA held-out loss -0.0061 | json | D | step6_sft/paired/wide1 - ctrl/mean | 1 | -0.0061 | 0.00005 |
| S2 | ... SD 0.0011 | json | D | step6_sft/paired/wide1 - ctrl/sd | 1 | 0.0011 | 0.00005 |
| S3 | ... 6 of 6 | json | D | step6_sft/paired/wide1 - ctrl/n_negative | 1 | 6 | 0 |
| S4 | Gate(0,2)-LoRA -0.0029 | json | D | step6_sft/paired/wide - ctrl/mean | 1 | -0.0029 | 0.00005 |
| S5 | ... SD 0.0018 | json | D | step6_sft/paired/wide - ctrl/sd | 1 | 0.0018 | 0.00005 |
| S6 | ... 5 of 6 | json | D | step6_sft/paired/wide - ctrl/n_negative | 1 | 5 | 0 |
| S7 | Gate(0,2)-Gate(0,1) +0.0032 | json | D | step6_sft/paired/wide - wide1/mean | 1 | 0.0032 | 0.00005 |
| S8 | ... SD 0.0014 | json | D | step6_sft/paired/wide - wide1/sd | 1 | 0.0014 | 0.00005 |
| S9 | beta > 1 share 5.2% (low end) | json | D | step6_sft/rows/wide_s0/gate_stat_mean | 1 | 0.052 | 0.0005 |
| S10 | beta > 1 share 5.5% (high end) | json | D | step6_sft/rows/wide_s2/gate_stat_mean | 1 | 0.055 | 0.0005 |
| S11 | held-out set 26,312 tokens | text | results/run_logs/broad/step6/ctrl_s0/val_log.jsonl | "val_tokens": 26312 |  |  |  |
| S12 | 1,200 steps of about 33,000 tokens, lr 1e-4 | text | src/gdn_gate/broad/train_sft.py | ap.add_argument("--step_tokens", type=int, default=33472) | | | |
| F1 | 4B untouched HumanEval 0.500 | json | D | forgetting_humaneval/4B_base_untouched/pass1 | 1 | 0.500 | 0.0005 |
| F2 | 4B LoRA 0.177 | json | D | forgetting_humaneval/4B_main_ctrl/s0/pass1 | 1 | 0.177 | 0.0005 |
| F3 | 4B LoRA 0.091 | json | D | forgetting_humaneval/4B_main_ctrl/s1/pass1 | 1 | 0.091 | 0.0005 |
| F4 | 4B LoRA 0.189 | json | D | forgetting_humaneval/4B_main_ctrl/s2/pass1 | 1 | 0.189 | 0.0005 |
| F5 | 4B Gate(0,2) 0.207 | json | D | forgetting_humaneval/4B_main_wide/s0/pass1 | 1 | 0.207 | 0.0005 |
| F6 | 4B Gate(0,2) 0.274 | json | D | forgetting_humaneval/4B_main_wide/s1/pass1 | 1 | 0.274 | 0.0005 |
| F7 | 4B Gate(0,2) 0.311 | json | D | forgetting_humaneval/4B_main_wide/s2/pass1 | 1 | 0.311 | 0.0005 |
| F8 | 4B Gate(0,1) 0.451 | json | D | forgetting_humaneval/4B_main_wide1/s0/pass1 | 1 | 0.451 | 0.0005 |
| F9 | 4B Gate(0,1) 0.366 | json | D | forgetting_humaneval/4B_main_wide1/s1/pass1 | 1 | 0.366 | 0.0005 |
| F10 | 4B Gate(0,1) 0.409 | json | D | forgetting_humaneval/4B_main_wide1/s2/pass1 | 1 | 0.409 | 0.0005 |
| F11 | McNemar p <= 4e-7 per seed (largest) | json | V245 | Table 3 4b_math500_area_wide-ctrl item lo 0.048 | json | X | contrasts/4b_math500_area_wide-ctrl/receipt_ci95_item/0 | 1 | 0.048 | 0.0005 |
| V246 | Table 3 4b_math500_area_wide-ctrl item hi 0.078 | json | X | contrasts/4b_math500_area_wide-ctrl/receipt_ci95_item/1 | 1 | 0.078 | 0.0005 |
| V247 | Table 3 4b_math500_area_wide1-ctrl item lo 0.053 | json | X | contrasts/4b_math500_area_wide1-ctrl/receipt_ci95_item/0 | 1 | 0.053 | 0.0005 |
| V248 | Table 3 4b_math500_area_wide1-ctrl item hi 0.078 | json | X | contrasts/4b_math500_area_wide1-ctrl/receipt_ci95_item/1 | 1 | 0.078 | 0.0005 |
| V249 | Table 3 4b_math500_area_wide1-ctrlpm item lo 0.049 | json | X | contrasts/4b_math500_area_wide1-ctrlpm/receipt_ci95_item/0 | 1 | 0.049 | 0.0005 |
| V250 | Table 3 4b_math500_area_wide1-ctrlpm item hi 0.072 | json | X | contrasts/4b_math500_area_wide1-ctrlpm/receipt_ci95_item/1 | 1 | 0.072 | 0.0005 |
| V251 | Table 3 4b_math500_area_wide-wide1 item lo -0.011 | json | X | contrasts/4b_math500_area_wide-wide1/receipt_ci95_item/0 | 1 | -0.011 | 0.0005 |
| V252 | Table 3 4b_math500_area_wide-wide1 item hi 0.006 | json | X | contrasts/4b_math500_area_wide-wide1/receipt_ci95_item/1 | 1 | 0.006 | 0.0005 |
| V253 | Table 3 4b_math500_area_ctrlpm-ctrl item lo -0.0 | json | X | contrasts/4b_math500_area_ctrlpm-ctrl/receipt_ci95_item/0 | 1 | -0.0 | 0.0005 |
| V254 | Table 3 4b_math500_area_ctrlpm-ctrl item hi 0.01 | json | X | contrasts/4b_math500_area_ctrlpm-ctrl/receipt_ci95_item/1 | 1 | 0.01 | 0.0005 |
| V255 | Table 3 4b_math500_len_wide1-ctrl item lo -2144 | json | X | contrasts/4b_math500_len_wide1-ctrl/receipt_ci95_item/0 | 1 | -2144 | 0.5 |
| V256 | Table 3 4b_math500_len_wide1-ctrl item hi -1905 | json | X | contrasts/4b_math500_len_wide1-ctrl/receipt_ci95_item/1 | 1 | -1905 | 0.5 |
| V257 | Table 3 4b_math500_len_wide-wide1 item lo -88 | json | X | contrasts/4b_math500_len_wide-wide1/receipt_ci95_item/0 | 1 | -88 | 0.5 |
| V258 | Table 3 4b_math500_len_wide-wide1 item hi -10 | json | X | contrasts/4b_math500_len_wide-wide1/receipt_ci95_item/1 | 1 | -10 | 0.5 |
| V259 | Table 3 4b_gsm8k_len_wide-ctrl item lo -1468 | json | X | contrasts/4b_gsm8k_len_wide-ctrl/receipt_ci95_item/0 | 1 | -1468 | 0.5 |
| V260 | Table 3 4b_gsm8k_len_wide-ctrl item hi -1344 | json | X | contrasts/4b_gsm8k_len_wide-ctrl/receipt_ci95_item/1 | 1 | -1344 | 0.5 |
| V261 | Table 3 4b_gsm8k_len_wide1-ctrl item lo -1433 | json | X | contrasts/4b_gsm8k_len_wide1-ctrl/receipt_ci95_item/0 | 1 | -1433 | 0.5 |
| V262 | Table 3 4b_gsm8k_len_wide1-ctrl item hi -1308 | json | X | contrasts/4b_gsm8k_len_wide1-ctrl/receipt_ci95_item/1 | 1 | -1308 | 0.5 |
| V263 | Table 3 4b_gsm8k_len_wide1-ctrlpm item lo -1043 | json | X | contrasts/4b_gsm8k_len_wide1-ctrlpm/receipt_ci95_item/0 | 1 | -1043 | 0.5 |
| V264 | Table 3 4b_gsm8k_len_wide1-ctrlpm item hi -958 | json | X | contrasts/4b_gsm8k_len_wide1-ctrlpm/receipt_ci95_item/1 | 1 | -958 | 0.5 |
| V265 | Table 3 4b_gsm8k_len_wide-wide1 item lo -44 | json | X | contrasts/4b_gsm8k_len_wide-wide1/receipt_ci95_item/0 | 1 | -44 | 0.5 |
| V266 | Table 3 4b_gsm8k_len_wide-wide1 item hi -27 | json | X | contrasts/4b_gsm8k_len_wide-wide1/receipt_ci95_item/1 | 1 | -27 | 0.5 |
| V267 | Table 3 4b_gsm8k_area_ctrlpm-ctrl item lo 0.003 | json | X | contrasts/4b_gsm8k_area_ctrlpm-ctrl/receipt_ci95_item/0 | 1 | 0.003 | 0.0005 |
| V268 | Table 3 4b_gsm8k_area_ctrlpm-ctrl item hi 0.01 | json | X | contrasts/4b_gsm8k_area_ctrlpm-ctrl/receipt_ci95_item/1 | 1 | 0.01 | 0.0005 |
| V269 | Table 3 4b_fresh_area_wide1-ctrl item lo 0.089 | json | X | contrasts/4b_fresh_area_wide1-ctrl/receipt_ci95_item/0 | 1 | 0.089 | 0.0005 |
| V270 | Table 3 4b_fresh_area_wide1-ctrl item hi 0.132 | json | X | contrasts/4b_fresh_area_wide1-ctrl/receipt_ci95_item/1 | 1 | 0.132 | 0.0005 |
| V271 | Table 3 4b_fresh_len_wide1-ctrl item lo -1585 | json | X | contrasts/4b_fresh_len_wide1-ctrl/receipt_ci95_item/0 | 1 | -1585 | 0.5 |
| V272 | Table 3 4b_fresh_len_wide1-ctrl item hi -1081 | json | X | contrasts/4b_fresh_len_wide1-ctrl/receipt_ci95_item/1 | 1 | -1081 | 0.5 |
| V273 | Table 3 4b_fresh_area_wide1-ctrlpm item lo 0.078 | json | X | contrasts/4b_fresh_area_wide1-ctrlpm/receipt_ci95_item/0 | 1 | 0.078 | 0.0005 |
| V274 | Table 3 4b_fresh_area_wide1-ctrlpm item hi 0.119 | json | X | contrasts/4b_fresh_area_wide1-ctrlpm/receipt_ci95_item/1 | 1 | 0.119 | 0.0005 |
| V275 | Table 3 4b_fresh_len_wide1-ctrlpm item lo -1467 | json | X | contrasts/4b_fresh_len_wide1-ctrlpm/receipt_ci95_item/0 | 1 | -1467 | 0.5 |
| V276 | Table 3 4b_fresh_len_wide1-ctrlpm item hi -1004 | json | X | contrasts/4b_fresh_len_wide1-ctrlpm/receipt_ci95_item/1 | 1 | -1004 | 0.5 |
| V277 | Table 3 4b_fresh_area_wide-wide1 item lo 0.004 | json | X | contrasts/4b_fresh_area_wide-wide1/receipt_ci95_item/0 | 1 | 0.004 | 0.0005 |
| V278 | Table 3 4b_fresh_area_wide-wide1 item hi 0.02 | json | X | contrasts/4b_fresh_area_wide-wide1/receipt_ci95_item/1 | 1 | 0.02 | 0.0005 |
| V279 | Table 3 4b_fresh_len_wide-wide1 item lo -169 | json | X | contrasts/4b_fresh_len_wide-wide1/receipt_ci95_item/0 | 1 | -169 | 0.5 |
| V280 | Table 3 4b_fresh_len_wide-wide1 item hi -61 | json | X | contrasts/4b_fresh_len_wide-wide1/receipt_ci95_item/1 | 1 | -61 | 0.5 |
| V281 | Table 3 2b_math500_area_wide-ctrl_r1 item lo 0.019 | json | X | contrasts/2b_math500_area_wide-ctrl_r1/receipt_ci95_item/0 | 1 | 0.019 | 0.0005 |
| V282 | Table 3 2b_math500_area_wide-ctrl_r1 item hi 0.035 | json | X | contrasts/2b_math500_area_wide-ctrl_r1/receipt_ci95_item/1 | 1 | 0.035 | 0.0005 |
| V283 | Table 3 2b_math500_area_wide1-ctrl_r1 item lo 0.013 | json | X | contrasts/2b_math500_area_wide1-ctrl_r1/receipt_ci95_item/0 | 1 | 0.013 | 0.0005 |
| V284 | Table 3 2b_math500_area_wide1-ctrl_r1 item hi 0.028 | json | X | contrasts/2b_math500_area_wide1-ctrl_r1/receipt_ci95_item/1 | 1 | 0.028 | 0.0005 |
| V285 | Table 3 2b_math500_area_wide-ctrl_r2 item lo 0.016 | json | X | contrasts/2b_math500_area_wide-ctrl_r2/receipt_ci95_item/0 | 1 | 0.016 | 0.0005 |
| V286 | Table 3 2b_math500_area_wide-ctrl_r2 item hi 0.034 | json | X | contrasts/2b_math500_area_wide-ctrl_r2/receipt_ci95_item/1 | 1 | 0.034 | 0.0005 |
| V287 | Table 3 2b_math500_area_wide1-ctrl_r2 item lo 0.013 | json | X | contrasts/2b_math500_area_wide1-ctrl_r2/receipt_ci95_item/0 | 1 | 0.013 | 0.0005 |
| V288 | Table 3 2b_math500_area_wide1-ctrl_r2 item hi 0.03 | json | X | contrasts/2b_math500_area_wide1-ctrl_r2/receipt_ci95_item/1 | 1 | 0.03 | 0.0005 |
| V289 | Table 3 2b_gsm8k_len_wide-ctrl_r2 item lo -1043 | json | X | contrasts/2b_gsm8k_len_wide-ctrl_r2/receipt_ci95_item/0 | 1 | -1043 | 0.5 |
| V290 | Table 3 2b_gsm8k_len_wide-ctrl_r2 item hi -953 | json | X | contrasts/2b_gsm8k_len_wide-ctrl_r2/receipt_ci95_item/1 | 1 | -953 | 0.5 |
| V291 | Table 3 2b_fresh_area_wide1-ctrl item lo 0.006 | json | X | contrasts/2b_fresh_area_wide1-ctrl/receipt_ci95_item/0 | 1 | 0.006 | 0.0005 |
| V292 | Table 3 2b_fresh_area_wide1-ctrl item hi 0.018 | json | X | contrasts/2b_fresh_area_wide1-ctrl/receipt_ci95_item/1 | 1 | 0.018 | 0.0005 |
| V293 | Table 3 2b_fresh_area_wide-wide1 item lo 0.006 | json | X | contrasts/2b_fresh_area_wide-wide1/receipt_ci95_item/0 | 1 | 0.006 | 0.0005 |
| V294 | Table 3 2b_fresh_area_wide-wide1 item hi 0.016 | json | X | contrasts/2b_fresh_area_wide-wide1/receipt_ci95_item/1 | 1 | 0.016 | 0.0005 |
| V295 | Table 3 2b_fresh_len_wide-ctrl item lo -404 | json | X | contrasts/2b_fresh_len_wide-ctrl/receipt_ci95_item/0 | 1 | -404 | 0.5 |
| V296 | Table 3 2b_fresh_len_wide-ctrl item hi -219 | json | X | contrasts/2b_fresh_len_wide-ctrl/receipt_ci95_item/1 | 1 | -219 | 0.5 |
| W1 | contrasts/wide1 - ctrl/he/per_seed/2/mcnemar_p | 1e7 | 4.04 | 0.01 |
| F12 | 9B LoRA 0.305 | json | D | forgetting_humaneval/9B_main_ctrl/s0/pass1 | 1 | 0.305 | 0.0005 |
| F13 | 9B LoRA 0.427 | json | D | forgetting_humaneval/9B_main_ctrl/s1/pass1 | 1 | 0.427 | 0.0005 |
| F14 | 9B LoRA 0.335 | json | D | forgetting_humaneval/9B_main_ctrl/s2/pass1 | 1 | 0.335 | 0.0005 |
| F15 | 9B Gate(0,2) 0.512 | json | D | forgetting_humaneval/9B_main_wide/s0/pass1 | 1 | 0.512 | 0.0005 |
| F16 | 9B Gate(0,2) 0.457 | json | D | forgetting_humaneval/9B_main_wide/s1/pass1 | 1 | 0.457 | 0.0005 |
| F17 | 9B Gate(0,2) 0.488 | json | D | forgetting_humaneval/9B_main_wide/s2/pass1 | 1 | 0.488 | 0.0005 |
| F18 | scorer check 4B 0.171 | json | D | forgetting_humaneval/4B_scorer_check/ctrl_s0_box/pass1 | 1 | 0.171 | 0.0005 |
| F19 | scorer check 4B 0.195 | json | D | forgetting_humaneval/4B_scorer_check/wide_s0_box/pass1 | 1 | 0.195 | 0.0005 |
| F20 | scorer check 9B 0.299 | json | D | forgetting_humaneval/9B_scorer_check/ctrl_s0_box/pass1 | 1 | 0.299 | 0.0005 |
| F21 | same-hardware trio LoRA 0.079 | json | D | forgetting_humaneval/4B_same_hw_s0/ctrl/pass1 | 1 | 0.079 | 0.0005 |
| F22 | same-hardware trio Gate(0,2) 0.262 | json | D | forgetting_humaneval/4B_same_hw_s0/wide/pass1 | 1 | 0.262 | 0.0005 |
| F23 | chat SFT seed 0: LoRA 0.530 | json | D | forgetting_humaneval/4B_step6_s0_vllm/ctrl_s0/pass1 | 1 | 0.530 | 0.0005 |
| F24 | chat SFT seed 0: gated 0.537 | json | D | forgetting_humaneval/4B_step6_s0_vllm/wide_s0/pass1 | 1 | 0.537 | 0.0005 |
| F25 | chat SFT seed 0: Gate(0,1) 0.537 | json | D | forgetting_humaneval/4B_step6_s0_vllm/wide1_s0/pass1 | 1 | 0.537 | 0.0005 |
| F26 | forgetting recipe: parity/swap/codeswap, lr 1e-4, 1,200 steps, WikiText replay | text | src/retrofit/recipe_main_4b.txt | --task_counts parity:16,swap:64,codeswap:64 | | | |

## Olmo-Hybrid-7B (Section 8)

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| O1 | clamp NLL +0.020 | json | ONLL | arms/erase/all/delta | 1 | 0.020 | 0.0005 |
| O2 | ... interval 0.017 | json | ONLL | arms/erase/all/ci/0 | 1 | 0.017 | 0.0005 |
| O3 | ... interval 0.023 | json | ONLL | arms/erase/all/ci/1 | 1 | 0.023 | 0.0005 |
| O4 | halve NLL +0.190 | json | ONLL | arms/half/all/delta | 1 | 0.190 | 0.0005 |
| O5 | 256 chunks | json | ONLL | arms/erase/n_chunks | 1 | 256 | 0 |
| O6 | MMLU-Pro stock 0.228 | json | D | olmo/mmlu_cloze/stock/cloze_acc | 1 | 0.228 | 0.0005 |
| O7 | MMLU-Pro clamp 0.222 | json | D | olmo/mmlu_cloze/erase/cloze_acc | 1 | 0.222 | 0.0005 |
| O8 | MMLU-Pro halve 0.192 | json | D | olmo/mmlu_cloze/half/cloze_acc | 1 | 0.192 | 0.0005 |
| O9 | 12,032 items | json | D | olmo/mmlu_cloze/stock/n | 1 | 12032 | 0 |
| O10 | GSM8K stock 0.747 | json | D | olmo/lm/stock/gsm8k_flexible | 1 | 0.747 | 0.0005 |
| O11 | GSM8K clamp 0.745 | json | D | olmo/lm/erase/gsm8k_flexible | 1 | 0.745 | 0.0005 |
| O12 | halve GSM8K -26 pt | json | D | olmo/lm/half/gsm8k_flexible | 1 | 0.488 | 0.0005 |
| O13 | halve HumanEval -20 pt (0.451 -> 0.256) | json | D | olmo/lm/half/humaneval | 1 | 0.256 | 0.0005 |
| O14 | halve MBPP+ -32 pt (0.712 -> 0.389) | json | D | olmo/lm/half/mbpp_plus | 1 | 0.389 | 0.0005 |
| O15 | clamp HumanEval within 4 pt (0.415) | json | D | olmo/lm/erase/humaneval | 1 | 0.415 | 0.0005 |
| O16 | clamp MBPP+ within 4 pt (0.675) | json | D | olmo/lm/erase/mbpp_plus | 1 | 0.675 | 0.0005 |
| O17 | clamp SQuAD within 2 pt (0.629 vs 0.609) | json | D | olmo/lm/erase/squad_completion | 1 | 0.629 | 0.0005 |
| O18 | RULER clamp macro 4k -0.5 | json | ORP | macro/1/diff | 100 | -0.46 | 0.005 |
| O19 | RULER clamp macro 8k -1.5 | json | ORP | macro/3/diff | 100 | -1.47 | 0.005 |
| O20 | RULER clamp macro 16k -1.1 | json | ORP | macro/5/diff | 100 | -1.12 | 0.005 |
| O21 | halve macro 4k -6.9 (7 to 8 pt) | json | ORP | macro/0/diff | 100 | -6.92 | 0.005 |
| O22 | halve macro 16k -8.0 | json | ORP | macro/4/diff | 100 | -7.95 | 0.005 |
| O23 | non-CWE clamp rows within 4 pt | json | D | olmo/ruler_nonCWE_erase/max_abs_diff | 1 | 0.04 | 0.00005 |
| O23b | ... with permutation p >= 0.06 | json | D | olmo/ruler_nonCWE_erase/min_p | 1 | 0.0608 | 0.00005 |
| O24 | CWE 4k 0.765 -> -3.8 | json | OCP | rows/0/diff | 100 | -3.84 | 0.005 |
| O25 | CWE 4k stock 0.765 | json | OCP | rows/0/stock | 1 | 0.765 | 0.0005 |
| O26 | CWE 8k -6.3 (from 0.511) | json | OCP | rows/2/diff | 100 | -6.26 | 0.005 |
| O27 | CWE 8k stock 0.511 | json | OCP | rows/2/stock | 1 | 0.511 | 0.0005 |
| O28 | CWE 16k -6.0 (from 0.305) | json | OCP | rows/4/diff | 100 | -6.00 | 0.005 |
| O29 | FWE 4k -2.1 | json | OCP | rows/1/diff | 100 | -2.07 | 0.005 |
| O30 | FWE 8k -3.7 | json | OCP | rows/3/diff | 100 | -3.73 | 0.005 |
| O31 | FWE 16k -2.9 | json | OCP | rows/5/diff | 100 | -2.93 | 0.005 |
| O32 | worse/better ratio low end 1.6 (CWE 4k 190/122) | json | OCP | rows/0/worse | 1 | 190 | 0 |
| O33 | ratio high end 4 (FWE 8k 53/13) | json | OCP | rows/3/worse | 1 | 53 | 0 |
| O34 | every cell p < 0.02 (largest FWE 4k) | json | OCP | rows/1/p | 1 | 0.0129 | 0.00005 |
| O35 | quarter Q1 -1.8 | json | OLOC | macro/0/diff | 100 | -1.80 | 0.005 |
| O36 | quarter Q2 -2.6 | json | OLOC | macro/1/diff | 100 | -2.60 | 0.005 |
| O37 | quarter Q3 -0.7 (interval includes 0) | json | OLOC | macro/2/ci95/1 | 1 | 0.0086 | 0.00005 |
| O38 | quarter Q4 -1.7 | json | OLOC | macro/3/diff | 100 | -1.68 | 0.005 |
| O39 | usage CWE 35% | json | D | olmo/usage_8k/tasks/ruler_cwe/gt1_frac | 1 | 0.349 | 0.0005 |
| O40 | usage single-needle 28% | json | D | olmo/usage_8k/tasks/niah_single_1/gt1_frac | 1 | 0.280 | 0.0005 |
| O41 | usage QA 46% | json | D | olmo/usage_8k/tasks/ruler_qa_squad/gt1_frac | 1 | 0.456 | 0.0005 |
| O42 | usage multikey (most use) 42% | json | D | olmo/usage_8k/tasks/niah_multikey_1/gt1_frac | 1 | 0.423 | 0.0005 |
| O43 | 40 items per task | json | D | olmo/usage_8k/n_items | 1 | 40 | 0 |
| O44 | retrofit 32k: Gate(0,2) -1.4 vs LoRA | json | VC32 | rows/0/diff | 100 | -1.40 | 0.005 |
| O45 | retrofit 32k: Gate(0,1) -3.9 vs LoRA | json | VC32 | rows/1/diff | 100 | -3.92 | 0.005 |

## Prior work numbers

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| C1 | ThinkingCap -31% GSM8K, -19% MATH-500, one seed | text | paper/data/thinkingcap_card_excerpt.txt | 31.2% | | | |

## Fresh held-out set (Sections 4, 5.2, 5.4, Reproducibility)

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| N1 | fresh set: AIME 2024/2025/2026 30 each, AMC 2023 40 (12A/12B) | text | PL4 | the 40 rows of 2023 AMC 12A/12B | | | |
| N2 | fresh set 130 items | json | F4 | n_items | 1 | 130 | 0 |
| N3 | 8 samples per item | json | F4 | samples_per_item | 1 | 8 | 0 |
| N4 | AIME 2026 30 items | json | F4 | per_source_descriptive/aime2026/n_items | 1 | 30 | 0 |
| N165 | AIME 2024 30 items | json | F4 | per_source_descriptive/aime2024/n_items | 1 | 30 | 0 |
| N166 | AIME 2025 30 items | json | F4 | per_source_descriptive/aime2025/n_items | 1 | 30 | 0 |
| N5 | AMC 2023 40 items | json | F4 | per_source_descriptive/amc2023/n_items | 1 | 40 | 0 |
| N6 | near-duplicate scan: no match | text | PL4 | 0 hits, 0 dropped | | | |
| N7 | read once (guard test touches 1) | json | ml_pipeline_4b/fresh_test/.guard_state.json | test_touches | 1 | 1 | 0 |
| N8 | 33 readouts, every row read by all models | json | FFIN | models | 1 | 33 | 0 |
| N9 | ... fresh rows read by all models 1.0 | json | FFIN | score | 1 | 1.0 | 0 |
| N10 | six budgets 256..8,192 (largest) | json | F4 | caps/5 | 1 | 8192 | 0 |
| N11 | H5 supported | json | F4 | preregistered_hypotheses/H5/supported | 1 | 1 | 0 |
| N12 | H6 supported | json | F4 | preregistered_hypotheses/H6/supported | 1 | 1 | 0 |
| N13 | H5-2B not supported | json | F2 | preregistered_hypotheses/H5/supported | 1 | 0 | 0 |
| N14 | H6-2B not supported | json | F2 | preregistered_hypotheses/H6/supported | 1 | 0 | 0 |
| N15 | analysis imports frozen code, constants only | text | src/gdn_gate/ml7/fresh/fresh_analysis.py | Nothing of the frozen scripts is edited: step14_analysis.py (S) and ctrlpm_analysis.py (P) are imported and only these | | | |
| N16 | deviation 1: plan two per GPU; every readout alone, pre-registered args | text | PL4 | every fresh readout runs ALONE on its GPU | | | |
| N17 | deviation 1: retries 4 | text | PL4 | Retries per readout 4 | | | |
| N18 | deviation 2: LoRA effect 0.0405 vs bar 0.0419 | text | PL4 | lora effect 0.0405 vs bar 0.0419 | | | |
| N19 | deviation 2: 0.0433 vs 0.0436, gate 0.0398 | text | PL4 | wide_2: lora 0.0433 vs 0.0436, gate 0.0398 vs 0.0436 | | | |
| N20 | deviation 2: bar 4.0 instead of 5.0 for exactly two readouts | text | PL4 | uses effect_visible 4.0 instead of 5.0 | | | |
| N21 | deviation 2: only 4b_ctrl_1 and 4b_wide_2 | text | src/gdn_gate/ml7/amendments/ledger.json | (ml7_4b ctrl_s1_cap512_r_box step 150, test_fresh.jsonl) and (ml7_4b wide_s2_cap512_r_box step 150, test_fresh.jsonl) | | | |
| N22 | eviction: four unfinished readouts restarted (2 4B + 2 2B) | text | docs/COMPUTE.md | four unfinished readouts (two 4B, two 2B) restarted | | | |
| N23 | all fresh readouts on RTX PRO 6000 Server Edition | text | results/fresh/env.txt | 7, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB | | | |
| N24 | mechanism readouts on RTX PRO 6000 Server Edition | text | results/mech/g1g2_summary.txt | gpus ['NVIDIA RTX PRO 6000 Blackwell Server Edition'] | | | |
| N25 | 4B table untouched area 0.199 | json | F4 | baseline/norm_auc_log2 | 1 | 0.199 | 0.0005 |
| N26 | 4B table untouched acc@1k 0.149 | json | F4 | baseline/acc/1024 | 1 | 0.149 | 0.0005 |
| N27 | 4B table untouched acc@8k 0.355 | json | F4 | baseline/acc/8192 | 1 | 0.355 | 0.0005 |
| N28 | 4B table untouched length 7741 | json | F4 | baseline/natural_len | 1 | 7741 | 0.5 |
| N29 | 4B table untouched open 0.87 | json | F4 | baseline/cut_off_share_at_max | 1 | 0.87 | 0.0005 |
| N30 | 4B table warm-up area 0.225 | json | F4 | warmup/norm_auc_log2 | 1 | 0.225 | 0.0005 |
| N31 | 4B table warm-up acc@1k 0.174 | json | F4 | warmup/acc/1024 | 1 | 0.174 | 0.0005 |
| N32 | 4B table warm-up acc@8k 0.387 | json | F4 | warmup/acc/8192 | 1 | 0.387 | 0.0005 |
| N33 | 4B table warm-up length 7663 | json | F4 | warmup/natural_len | 1 | 7663 | 0.5 |
| N34 | 4B table warm-up open 0.855 | json | F4 | warmup/cut_off_share_at_max | 1 | 0.855 | 0.0005 |
| N35 | 4B table LoRA area 0.285 | json | F4 | arms/ctrl/norm_auc_log2/seed_mean | 1 | 0.285 | 0.0005 |
| N36 | 4B table LoRA acc@1k 0.236 | json | F4 | arms/ctrl/acc/1024/seed_mean | 1 | 0.236 | 0.0005 |
| N37 | 4B table LoRA acc@8k 0.475 | json | F4 | arms/ctrl/acc/8192/seed_mean | 1 | 0.475 | 0.0005 |
| N38 | 4B table LoRA length 7424 | json | F4 | arms/ctrl/natural_len/seed_mean | 1 | 7424 | 0.5 |
| N39 | 4B table LoRA open 0.786 | json | F4 | arms/ctrl/cut_off_share_at_max/seed_mean | 1 | 0.786 | 0.0005 |
| N40 | 4B table LoRA-r20 area 0.297 | json | F4 | arms/ctrlpm/norm_auc_log2/seed_mean | 1 | 0.297 | 0.0005 |
| N41 | 4B table LoRA-r20 acc@1k 0.245 | json | F4 | arms/ctrlpm/acc/1024/seed_mean | 1 | 0.245 | 0.0005 |
| N42 | 4B table LoRA-r20 acc@8k 0.488 | json | F4 | arms/ctrlpm/acc/8192/seed_mean | 1 | 0.488 | 0.0005 |
| N43 | 4B table LoRA-r20 length 7325 | json | F4 | arms/ctrlpm/natural_len/seed_mean | 1 | 7325 | 0.5 |
| N44 | 4B table LoRA-r20 open 0.759 | json | F4 | arms/ctrlpm/cut_off_share_at_max/seed_mean | 1 | 0.759 | 0.0005 |
| N45 | 4B table Gate(0,1) area 0.395 | json | F4 | arms/wide1/norm_auc_log2/seed_mean | 1 | 0.395 | 0.0005 |
| N46 | 4B table Gate(0,1) acc@1k 0.334 | json | F4 | arms/wide1/acc/1024/seed_mean | 1 | 0.334 | 0.0005 |
| N47 | 4B table Gate(0,1) acc@8k 0.608 | json | F4 | arms/wide1/acc/8192/seed_mean | 1 | 0.608 | 0.0005 |
| N48 | 4B table Gate(0,1) length 6094 | json | F4 | arms/wide1/natural_len/seed_mean | 1 | 6094 | 0.5 |
| N49 | 4B table Gate(0,1) open 0.525 | json | F4 | arms/wide1/cut_off_share_at_max/seed_mean | 1 | 0.525 | 0.0005 |
| N50 | 4B table Gate(0,2) area 0.407 | json | F4 | arms/wide/norm_auc_log2/seed_mean | 1 | 0.407 | 0.0005 |
| N51 | 4B table Gate(0,2) acc@1k 0.348 | json | F4 | arms/wide/acc/1024/seed_mean | 1 | 0.348 | 0.0005 |
| N52 | 4B table Gate(0,2) acc@8k 0.621 | json | F4 | arms/wide/acc/8192/seed_mean | 1 | 0.621 | 0.0005 |
| N53 | 4B table Gate(0,2) length 5981 | json | F4 | arms/wide/natural_len/seed_mean | 1 | 5981 | 0.5 |
| N54 | 4B table Gate(0,2) open 0.515 | json | F4 | arms/wide/cut_off_share_at_max/seed_mean | 1 | 0.515 | 0.0005 |
| N55 | 4B budget to match: warm-up 5,881 | json | F4 | warmup/tokens_to_base_max_acc | 1 | 5881 | 0.5 |
| N56 | 4B budget to match LoRA 3045 | json | F4 | arms/ctrl/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 3045 | 0.5 |
| N57 | 4B budget to match LoRA-r20 2784 | json | F4 | arms/ctrlpm/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 2784 | 0.5 |
| N58 | 4B budget to match Gate(0,1) 1205 | json | F4 | arms/wide1/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 1205 | 0.5 |
| N59 | 4B budget to match Gate(0,2) 1081 | json | F4 | arms/wide/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 1081 | 0.5 |
| N60 | 4B target 0.355 | json | F4 | arms/ctrl/tokens_to_base_max_acc/target | 1 | 0.355 | 0.0005 |
| N61 | 2B table untouched area 0.067 | json | F2 | baseline/norm_auc_log2 | 1 | 0.067 | 0.0005 |
| N62 | 2B table untouched acc@1k 0.049 | json | F2 | baseline/acc/1024 | 1 | 0.049 | 0.0005 |
| N63 | 2B table untouched acc@8k 0.122 | json | F2 | baseline/acc/8192 | 1 | 0.122 | 0.0005 |
| N64 | 2B table untouched length 8083 | json | F2 | baseline/natural_len | 1 | 8083 | 0.5 |
| N65 | 2B table untouched open 0.969 | json | F2 | baseline/cut_off_share_at_max | 1 | 0.969 | 0.0005 |
| N66 | 2B table LoRA area 0.174 | json | F2 | arms/ctrl/norm_auc_log2/seed_mean | 1 | 0.174 | 0.0005 |
| N67 | 2B table LoRA acc@1k 0.133 | json | F2 | arms/ctrl/acc/1024/seed_mean | 1 | 0.133 | 0.0005 |
| N68 | 2B table LoRA acc@8k 0.309 | json | F2 | arms/ctrl/acc/8192/seed_mean | 1 | 0.309 | 0.0005 |
| N69 | 2B table LoRA length 7704 | json | F2 | arms/ctrl/natural_len/seed_mean | 1 | 7704 | 0.5 |
| N70 | 2B table LoRA open 0.868 | json | F2 | arms/ctrl/cut_off_share_at_max/seed_mean | 1 | 0.868 | 0.0005 |
| N71 | 2B table Gate(0,1) area 0.186 | json | F2 | arms/wide1/norm_auc_log2/seed_mean | 1 | 0.186 | 0.0005 |
| N72 | 2B table Gate(0,1) acc@1k 0.154 | json | F2 | arms/wide1/acc/1024/seed_mean | 1 | 0.154 | 0.0005 |
| N73 | 2B table Gate(0,1) acc@8k 0.322 | json | F2 | arms/wide1/acc/8192/seed_mean | 1 | 0.322 | 0.0005 |
| N74 | 2B table Gate(0,1) length 7578 | json | F2 | arms/wide1/natural_len/seed_mean | 1 | 7578 | 0.5 |
| N75 | 2B table Gate(0,1) open 0.833 | json | F2 | arms/wide1/cut_off_share_at_max/seed_mean | 1 | 0.833 | 0.0005 |
| N76 | 2B table Gate(0,2) area 0.196 | json | F2 | arms/wide/norm_auc_log2/seed_mean | 1 | 0.196 | 0.0005 |
| N77 | 2B table Gate(0,2) acc@1k 0.161 | json | F2 | arms/wide/acc/1024/seed_mean | 1 | 0.161 | 0.0005 |
| N78 | 2B table Gate(0,2) acc@8k 0.337 | json | F2 | arms/wide/acc/8192/seed_mean | 1 | 0.337 | 0.0005 |
| N79 | 2B table Gate(0,2) length 7397 | json | F2 | arms/wide/natural_len/seed_mean | 1 | 7397 | 0.5 |
| N80 | 2B table Gate(0,2) open 0.799 | json | F2 | arms/wide/cut_off_share_at_max/seed_mean | 1 | 0.799 | 0.0005 |
| N81 | 2B budget to match LoRA 715 | json | F2 | arms/ctrl/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 715 | 0.5 |
| N82 | 2B budget to match Gate(0,1) 523 | json | F2 | arms/wide1/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 523 | 0.5 |
| N83 | 2B budget to match Gate(0,2) 303 | json | F2 | arms/wide/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 303 | 0.5 |
| N84 | 2B target 0.122 | json | F2 | arms/ctrl/tokens_to_base_max_acc/target | 1 | 0.122 | 0.0005 |
| N85 | 2B 6 seeds | json | F2 | arms/ctrl/seeds/5 | 1 | 5 | 0 |
| N86 | 2B 18 arm runs (3 arms x 6 seeds) | json | F2 | n_arm_runs | 1 | 18 | 0 |
| N87 | H5 Gate(0,1)-LoRA area seed_mean_diff 0.11 | json | F4 | preregistered_hypotheses/H5/seed_mean_diff | 1 | 0.11 | 0.0005 |
| N88 | H5 Gate(0,1)-LoRA area ci95/0 0.089 | json | F4 | preregistered_hypotheses/H5/ci95/0 | 1 | 0.089 | 0.0005 |
| N89 | H5 Gate(0,1)-LoRA area ci95/1 0.132 | json | F4 | preregistered_hypotheses/H5/ci95/1 | 1 | 0.132 | 0.0005 |
| N90 | H5 Gate(0,1)-LoRA area per_seed_diff/0 0.137 | json | F4 | preregistered_hypotheses/H5/per_seed_diff/0 | 1 | 0.137 | 0.0005 |
| N91 | H5 Gate(0,1)-LoRA area per_seed_diff/1 0.082 | json | F4 | preregistered_hypotheses/H5/per_seed_diff/1 | 1 | 0.082 | 0.0005 |
| N92 | H5 Gate(0,1)-LoRA area per_seed_diff/2 0.112 | json | F4 | preregistered_hypotheses/H5/per_seed_diff/2 | 1 | 0.112 | 0.0005 |
| N93 | H6 Gate(0,1)-LoRA length seed_mean_diff -1329 | json | F4 | preregistered_hypotheses/H6/seed_mean_diff | 1 | -1329 | 0.5 |
| N94 | H6 Gate(0,1)-LoRA length ci95/0 -1585 | json | F4 | preregistered_hypotheses/H6/ci95/0 | 1 | -1585 | 0.5 |
| N95 | H6 Gate(0,1)-LoRA length ci95/1 -1081 | json | F4 | preregistered_hypotheses/H6/ci95/1 | 1 | -1081 | 0.5 |
| N96 | H6 Gate(0,1)-LoRA length per_seed_diff/0 -1409 | json | F4 | preregistered_hypotheses/H6/per_seed_diff/0 | 1 | -1409 | 0.5 |
| N97 | H6 Gate(0,1)-LoRA length per_seed_diff/1 -1038 | json | F4 | preregistered_hypotheses/H6/per_seed_diff/1 | 1 | -1038 | 0.5 |
| N98 | H6 Gate(0,1)-LoRA length per_seed_diff/2 -1541 | json | F4 | preregistered_hypotheses/H6/per_seed_diff/2 | 1 | -1541 | 0.5 |
| N99 | Gate(0,1)-LoRA-r20 area seed_mean_diff 0.098 | json | F4 | effects/wide1 - ctrlpm/seed_mean_diff | 1 | 0.098 | 0.0005 |
| N100 | Gate(0,1)-LoRA-r20 area ci95/0 0.078 | json | F4 | effects/wide1 - ctrlpm/ci95/0 | 1 | 0.078 | 0.0005 |
| N101 | Gate(0,1)-LoRA-r20 area ci95/1 0.119 | json | F4 | effects/wide1 - ctrlpm/ci95/1 | 1 | 0.119 | 0.0005 |
| N102 | ... is an effect | json | F4 | effects/wide1 - ctrlpm/effect | 1 | 1 | 0 |
| N103 | Gate(0,1)-LoRA-r20 length seed_mean_diff -1231 | json | F4 | length_effects/wide1 - ctrlpm/seed_mean_diff | 1 | -1231 | 0.5 |
| N104 | Gate(0,1)-LoRA-r20 length ci95/0 -1467 | json | F4 | length_effects/wide1 - ctrlpm/ci95/0 | 1 | -1467 | 0.5 |
| N105 | Gate(0,1)-LoRA-r20 length ci95/1 -1004 | json | F4 | length_effects/wide1 - ctrlpm/ci95/1 | 1 | -1004 | 0.5 |
| N106 | ... is an effect | json | F4 | length_effects/wide1 - ctrlpm/effect | 1 | 1 | 0 |
| N107 | LoRA-r20-LoRA area +0.012 | json | F4 | effects/ctrlpm - ctrl/seed_mean_diff | 1 | 0.012 | 0.0005 |
| N108 | ... seed 0 -0.004 | json | F4 | effects/ctrlpm - ctrl/per_seed_diff/0 | 1 | -0.004 | 0.0005 |
| N109 | ... no effect | json | F4 | effects/ctrlpm - ctrl/effect | 1 | 0 | 0 |
| N110 | LoRA-r20-LoRA length -99 | json | F4 | length_effects/ctrlpm - ctrl/seed_mean_diff | 1 | -99 | 0.5 |
| N111 | ... no effect | json | F4 | length_effects/ctrlpm - ctrl/effect | 1 | 0 | 0 |
| N112 | open 52% gated (Gate(0,1) 0.525) | json | F4 | arms/wide1/cut_off_share_at_max/seed_mean | 1 | 0.52 | 0.005 |
| N113 | open 52% gated (Gate(0,2) 0.515) | json | F4 | arms/wide/cut_off_share_at_max/seed_mean | 1 | 0.52 | 0.005 |
| N114 | open 79% LoRA | json | F4 | arms/ctrl/cut_off_share_at_max/seed_mean | 1 | 0.79 | 0.005 |
| N115 | aime2024 LoRA area 0.18 | json | F4 | per_source_descriptive/aime2024/arms/ctrl/norm_auc_log2/seed_mean | 1 | 0.18 | 0.0005 |
| N116 | aime2024 Gate(0,1) area 0.302 | json | F4 | per_source_descriptive/aime2024/arms/wide1/norm_auc_log2/seed_mean | 1 | 0.302 | 0.0005 |
| N117 | aime2024 Gate(0,2) area 0.305 | json | F4 | per_source_descriptive/aime2024/arms/wide/norm_auc_log2/seed_mean | 1 | 0.305 | 0.0005 |
| N118 | aime2025 LoRA area 0.175 | json | F4 | per_source_descriptive/aime2025/arms/ctrl/norm_auc_log2/seed_mean | 1 | 0.175 | 0.0005 |
| N119 | aime2025 Gate(0,1) area 0.259 | json | F4 | per_source_descriptive/aime2025/arms/wide1/norm_auc_log2/seed_mean | 1 | 0.259 | 0.0005 |
| N120 | aime2025 Gate(0,2) area 0.267 | json | F4 | per_source_descriptive/aime2025/arms/wide/norm_auc_log2/seed_mean | 1 | 0.267 | 0.0005 |
| N121 | aime2026 LoRA area 0.132 | json | F4 | per_source_descriptive/aime2026/arms/ctrl/norm_auc_log2/seed_mean | 1 | 0.132 | 0.0005 |
| N122 | aime2026 Gate(0,1) area 0.242 | json | F4 | per_source_descriptive/aime2026/arms/wide1/norm_auc_log2/seed_mean | 1 | 0.242 | 0.0005 |
| N123 | aime2026 Gate(0,2) area 0.259 | json | F4 | per_source_descriptive/aime2026/arms/wide/norm_auc_log2/seed_mean | 1 | 0.259 | 0.0005 |
| N124 | amc2023 LoRA area 0.561 | json | F4 | per_source_descriptive/amc2023/arms/ctrl/norm_auc_log2/seed_mean | 1 | 0.561 | 0.0005 |
| N125 | amc2023 Gate(0,1) area 0.682 | json | F4 | per_source_descriptive/amc2023/arms/wide1/norm_auc_log2/seed_mean | 1 | 0.682 | 0.0005 |
| N126 | amc2023 Gate(0,2) area 0.7 | json | F4 | per_source_descriptive/amc2023/arms/wide/norm_auc_log2/seed_mean | 1 | 0.7 | 0.0005 |
| N127 | AIME 2026 untouched 0.059 | json | F4 | per_source_descriptive/aime2026/baseline_norm_auc_log2 | 1 | 0.059 | 0.0005 |
| N128 | gain on every source (both gated arms above LoRA) | json | D | fresh/4b/per_source_all_gated_above_ctrl | 1 | 1 | 0 |
| N129 | model card newest benchmark HMMT Nov 2025 | text | PL4 | reports HMMT Nov 2025 as its newest benchmark | | | |
| N130 | AIME 2026 contests 2026-02-05 and 02-11 | text | PL4 | contests 2026-02-05 and 02-11 | | | |
| N131 | weights published 2026-02-27 / 02-28 | text | PL4 | weights were published 2026-02-27 / 02-28 | | | |
| N132 | Gate(0,2)-Gate(0,1) area seed_mean_diff 0.012 | json | F4 | effects/wide - wide1/seed_mean_diff | 1 | 0.012 | 0.0005 |
| N133 | Gate(0,2)-Gate(0,1) area ci95/0 0.004 | json | F4 | effects/wide - wide1/ci95/0 | 1 | 0.004 | 0.0005 |
| N134 | Gate(0,2)-Gate(0,1) area ci95/1 0.02 | json | F4 | effects/wide - wide1/ci95/1 | 1 | 0.02 | 0.0005 |
| N135 | Gate(0,2)-Gate(0,1) area per_seed_diff/0 0.017 | json | F4 | effects/wide - wide1/per_seed_diff/0 | 1 | 0.017 | 0.0005 |
| N136 | Gate(0,2)-Gate(0,1) area per_seed_diff/1 0.013 | json | F4 | effects/wide - wide1/per_seed_diff/1 | 1 | 0.013 | 0.0005 |
| N137 | Gate(0,2)-Gate(0,1) area per_seed_diff/2 0.006 | json | F4 | effects/wide - wide1/per_seed_diff/2 | 1 | 0.006 | 0.0005 |
| N138 | ... an effect | json | F4 | effects/wide - wide1/effect | 1 | 1 | 0 |
| N139 | Gate(0,2)-Gate(0,1) length -113 | json | F4 | length_effects/wide - wide1/seed_mean_diff | 1 | -113 | 0.5 |
| N140 | ... seed 2 +65 | json | F4 | length_effects/wide - wide1/per_seed_diff/2 | 1 | 65 | 0.5 |
| N141 | ... no effect | json | F4 | length_effects/wide - wide1/effect | 1 | 0 | 0 |
| N142 | 2B: never separate on MATH/GSM8K (2B read-1 G02-G01 no effect) | json | A2M | effects/wide - wide1/effect | 1 | 0 | 0 |
| N143 | 2B H5 Gate(0,1)-LoRA area seed_mean_diff 0.012 | json | F2 | preregistered_hypotheses/H5/seed_mean_diff | 1 | 0.012 | 0.0005 |
| N144 | 2B H5 Gate(0,1)-LoRA area ci95/0 0.006 | json | F2 | preregistered_hypotheses/H5/ci95/0 | 1 | 0.006 | 0.0005 |
| N145 | 2B H5 Gate(0,1)-LoRA area ci95/1 0.018 | json | F2 | preregistered_hypotheses/H5/ci95/1 | 1 | 0.018 | 0.0005 |
| N146 | 2B H5 Gate(0,1)-LoRA area per_seed_diff/0 -0.003 | json | F2 | preregistered_hypotheses/H5/per_seed_diff/0 | 1 | -0.003 | 0.0005 |
| N147 | 2B H5 Gate(0,1)-LoRA area per_seed_diff/5 -0.002 | json | F2 | preregistered_hypotheses/H5/per_seed_diff/5 | 1 | -0.002 | 0.0005 |
| N148 | 2B H6 Gate(0,1)-LoRA length seed_mean_diff -126 | json | F2 | preregistered_hypotheses/H6/seed_mean_diff | 1 | -126 | 0.5 |
| N149 | 2B H6 Gate(0,1)-LoRA length ci95/0 -173 | json | F2 | preregistered_hypotheses/H6/ci95/0 | 1 | -173 | 0.5 |
| N150 | 2B H6 Gate(0,1)-LoRA length ci95/1 -83 | json | F2 | preregistered_hypotheses/H6/ci95/1 | 1 | -83 | 0.5 |
| N151 | 2B H6 Gate(0,1)-LoRA length per_seed_diff/5 5 | json | F2 | preregistered_hypotheses/H6/per_seed_diff/5 | 1 | 5 | 0.5 |
| N152 | 2B Gate(0,2)-LoRA length seed_mean_diff -307 | json | F2 | length_effects/wide - ctrl/seed_mean_diff | 1 | -307 | 0.5 |
| N153 | 2B Gate(0,2)-LoRA length ci95/0 -404 | json | F2 | length_effects/wide - ctrl/ci95/0 | 1 | -404 | 0.5 |
| N154 | 2B Gate(0,2)-LoRA length ci95/1 -219 | json | F2 | length_effects/wide - ctrl/ci95/1 | 1 | -219 | 0.5 |
| N155 | ... 6 of 6, an effect | json | F2 | length_effects/wide - ctrl/effect | 1 | 1 | 0 |
| N156 | 2B Gate(0,2)-LoRA area +0.022 | json | F2 | effects/wide - ctrl/seed_mean_diff | 1 | 0.022 | 0.0005 |
| N157 | ... seed 3 -0.005 | json | F2 | effects/wide - ctrl/per_seed_diff/3 | 1 | -0.005 | 0.0005 |
| N158 | ... no effect | json | F2 | effects/wide - ctrl/effect | 1 | 0 | 0 |
| N159 | 2B LoRA 87% open | json | F2 | arms/ctrl/cut_off_share_at_max/seed_mean | 1 | 0.87 | 0.005 |
| N160 | every 2B arm beats untouched (ctrl) | json | F2 | effects/ctrl - untouched/effect | 1 | 1 | 0 |
| N161 | every 2B arm beats untouched (wide1) | json | F2 | effects/wide1 - untouched/effect | 1 | 1 | 0 |
| N162 | every 2B arm beats untouched (wide) | json | F2 | effects/wide - untouched/effect | 1 | 1 | 0 |
| N163 | compute: Phase 5 4 GPUs 4.9 h | text | docs/COMPUTE.md | 4 x RTX PRO 6000, 4.9 h | | | |
| N164 | compute: Phase 5 8 GPUs 2.2 h | text | docs/COMPUTE.md | 8 x RTX PRO 6000, 2.2 h | | | |

## Mechanism (Section 6, validation slice and saved traces)

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| M1 | validation slice, 1 sample, answer cap 1,280, probe 0.05 | text | src/gdn_gate/mech/mech_box.sh | --caps 256,512,1024,2048,4096,8192 --answer_max 1280 --samples 1 --seed 20261006 --gpu_mem 0.45 | | | |
| M2 | 500 items | json | G12 | n_items | 1 | 500 | 0 |
| M3 | decay change at most 0.0016 | json | D | mech/g3/decay_abs_change_max_all_runs | 1 | 0.0016 | 5e-05 |
| M4 | 42 trained checkpoints probed (12 at step 150 + 6 x 5 earlier) | json | D | mech/g3/n_runs | 1 | 42 | 0 |
| M5 | W shape 32 x 2,560 (hidden -> value heads) | text | src/gdn_gate/mech/w_stats.py | W (one 32 x 2560 matrix per GDN | | | |
| M6 | G1 length wide1_full 307 | json | G12 | g1_condition_means/wide1_full/natural_len | 1 | 307 | 0.5 |
| M7 | G1 length wide1_strip 665 | json | G12 | g1_condition_means/wide1_strip/natural_len | 1 | 665 | 0.5 |
| M8 | G1 length wide1_hybrid 970 | json | G12 | g1_condition_means/wide1_hybrid/natural_len | 1 | 970 | 0.5 |
| M9 | G1 length wide_full 254 | json | G12 | g1_condition_means/wide_full/natural_len | 1 | 254 | 0.5 |
| M10 | G1 length wide_strip 917 | json | G12 | g1_condition_means/wide_strip/natural_len | 1 | 917 | 0.5 |
| M11 | G1 length wide_hybrid 690 | json | G12 | g1_condition_means/wide_hybrid/natural_len | 1 | 690 | 0.5 |
| M12 | G1 length warmup 2916 | json | G12 | g1_condition_means/warmup/natural_len | 1 | 2916 | 0.5 |
| M13 | G1 length ctrl 1600 | json | G12 | g1_condition_means/ctrl/natural_len | 1 | 1600 | 0.5 |
| M14 | G1 wide1: full - strip seed_mean -358 | json | G12 | g1_contrasts_descriptive/wide1: full - strip/natural_len/seed_mean | 1 | -358 | 0.5 |
| M15 | G1 wide1: full - strip ci95/0 -411 | json | G12 | g1_contrasts_descriptive/wide1: full - strip/natural_len/ci95/0 | 1 | -411 | 0.5 |
| M16 | G1 wide1: full - strip ci95/1 -310 | json | G12 | g1_contrasts_descriptive/wide1: full - strip/natural_len/ci95/1 | 1 | -310 | 0.5 |
| M17 | G1 wide: full - strip seed_mean -663 | json | G12 | g1_contrasts_descriptive/wide: full - strip/natural_len/seed_mean | 1 | -663 | 0.5 |
| M18 | G1 wide: full - strip ci95/0 -744 | json | G12 | g1_contrasts_descriptive/wide: full - strip/natural_len/ci95/0 | 1 | -744 | 0.5 |
| M19 | G1 wide: full - strip ci95/1 -592 | json | G12 | g1_contrasts_descriptive/wide: full - strip/natural_len/ci95/1 | 1 | -592 | 0.5 |
| M20 | G1 wide1: hybrid - warmup seed_mean -1946 | json | G12 | g1_contrasts_descriptive/wide1: hybrid - warmup/natural_len/seed_mean | 1 | -1946 | 0.5 |
| M21 | G1 wide: hybrid - warmup seed_mean -2226 | json | G12 | g1_contrasts_descriptive/wide: hybrid - warmup/natural_len/seed_mean | 1 | -2226 | 0.5 |
| M22 | G1 wide1: strip - ctrl seed_mean -935 | json | G12 | g1_contrasts_descriptive/wide1: strip - ctrl/natural_len/seed_mean | 1 | -935 | 0.5 |
| M23 | G1 wide: strip - ctrl seed_mean -683 | json | G12 | g1_contrasts_descriptive/wide: strip - ctrl/natural_len/seed_mean | 1 | -683 | 0.5 |
| M24 | every G1 length contrast: same sign 3/3 and interval excludes 0 | json | D | mech/g1/length_contrasts_all_rule_shape | 1 | 1 | 0 |
| M25 | removing the gate lengthens all 6 runs: Gate(0,1) s0 | json | G12 | g1_contrasts_descriptive/wide1: full - strip/natural_len/per_seed/0 | 1 | -352 | 0.5 |
| M26 | removing the gate lengthens all 6 runs: Gate(0,1) s1 | json | G12 | g1_contrasts_descriptive/wide1: full - strip/natural_len/per_seed/1 | 1 | -452 | 0.5 |
| M27 | removing the gate lengthens all 6 runs: Gate(0,1) s2 | json | G12 | g1_contrasts_descriptive/wide1: full - strip/natural_len/per_seed/2 | 1 | -270 | 0.5 |
| M28 | ... Gate(0,2) s0 | json | G12 | g1_contrasts_descriptive/wide: full - strip/natural_len/per_seed/0 | 1 | -568 | 0.5 |
| M29 | ... Gate(0,2) s1 | json | G12 | g1_contrasts_descriptive/wide: full - strip/natural_len/per_seed/1 | 1 | -781 | 0.5 |
| M30 | ... Gate(0,2) s2 | json | G12 | g1_contrasts_descriptive/wide: full - strip/natural_len/per_seed/2 | 1 | -641 | 0.5 |
| M31 | gate alone removes 67% of warm-up (Gate(0,1)) | json | D | mech/g1/wide1/hybrid_cut_of_warmup | 1 | 0.67 | 0.005 |
| M32 | ... 76% (Gate(0,2)) | json | D | mech/g1/wide/hybrid_cut_of_warmup | 1 | 0.76 | 0.005 |
| M33 | LoRA training removes 45% of warm-up | json | D | mech/g1/ctrl_cut_of_warmup | 1 | 0.45 | 0.005 |
| M34 | co-trained LoRA without gate 58% shorter than LoRA arm (Gate(0,1)) | json | D | mech/g1/wide1/strip_cut_vs_ctrl | 1 | 0.58 | 0.005 |
| M35 | ... 43% (Gate(0,2)) | json | D | mech/g1/wide/strip_cut_vs_ctrl | 1 | 0.43 | 0.005 |
| M36 | 21 trained readouts area low 0.949 | json | D | mech/g1/area_range_excl_warmup/min | 1 | 0.949 | 0.0005 |
| M37 | ... high 0.966 | json | D | mech/g1/area_range_excl_warmup/max | 1 | 0.966 | 0.0005 |
| M38 | warm-up area 0.943 | json | G12 | per_readout/warmup/norm_auc_log2 | 1 | 0.943 | 0.0005 |
| M39 | 22 readouts in G1 | json | D | mech/g1/n_readouts | 1 | 22 | 0 |
| M40 | no full/strip/hybrid/LoRA area contrast has rule shape (only those vs warm-up) | json | D | mech/g1/area_rule_shape_only_vs_warmup | 1 | 1 | 0 |
| M41 | G2 s0 scale 0 length 712 | json | G12 | g2_dose/wide1_s0/points/0/natural_len | 1 | 712 | 0.5 |
| M42 | G2 s0 scale 2 length 212 | json | G12 | g2_dose/wide1_s0/points/4/natural_len | 1 | 212 | 0.5 |
| M43 | G2 s1 scale 0 length 823 | json | G12 | g2_dose/wide1_s1/points/0/natural_len | 1 | 823 | 0.5 |
| M44 | G2 s1 scale 2 length 299 | json | G12 | g2_dose/wide1_s1/points/4/natural_len | 1 | 299 | 0.5 |
| M45 | G2 s2 scale 0 length 541 | json | G12 | g2_dose/wide1_s2/points/0/natural_len | 1 | 541 | 0.5 |
| M46 | G2 s2 scale 2 length 179 | json | G12 | g2_dose/wide1_s2/points/4/natural_len | 1 | 179 | 0.5 |
| M47 | G2 scale 0 point is scale 0 | json | G12 | g2_dose/wide1_s0/points/0/scale | 1 | 0 | 0 |
| M48 | G2 scale 2 point is scale 2 | json | G12 | g2_dose/wide1_s0/points/4/scale | 1 | 2 | 0 |
| M49 | G2 monotone in all seeds | json | D | mech/g2/monotone_all_seeds | 1 | 1 | 0 |
| M50 | G2 slope -167 (shallowest) | json | D | mech/g2/slopes/max | 1 | -167 | 0.5 |
| M51 | G2 slope -247 (steepest) | json | D | mech/g2/slopes/min | 1 | -247 | 0.5 |
| M52 | G2 62% of drop in 0 to 0.5 (low) | json | D | mech/g2/share_of_0_to_2_drop_in_0_to_0p5/min | 1 | 0.62 | 0.005 |
| M53 | G2 68% (high) | json | D | mech/g2/share_of_0_to_2_drop_in_0_to_0p5/max | 1 | 0.68 | 0.005 |
| M54 | G2 area low 0.949 | json | D | mech/g2/area_range_all_scales/min | 1 | 0.949 | 0.0005 |
| M55 | G2 area high 0.963 | json | D | mech/g2/area_range_all_scales/max | 1 | 0.963 | 0.0005 |
| M56 | strip on stock engine s0 659 | json | G12 | per_readout/wide1_s0_strip/natural_len | 1 | 659 | 0.5 |
| M57 | strip on stock engine s1 832 | json | G12 | per_readout/wide1_s1_strip/natural_len | 1 | 832 | 0.5 |
| M58 | strip on stock engine s2 505 | json | G12 | per_readout/wide1_s2_strip/natural_len | 1 | 505 | 0.5 |
| M59 | G3 64 traces | json | D | mech/g3/n_seq | 1 | 64 | 0 |
| M60 | G3 99,769 tokens | json | D | mech/g3/n_tokens | 1 | 99769 | 0 |
| M61 | G3 traces of 2,048 tokens | text | src/gdn_gate/mech/mech_box.sh | the 64 fixed validation traces x 2,048 | | | |
| M62 | gate |dbeta| wide1 low 0.018 | json | D | mech/g3/wide1/gate_abs_dbeta_150/min | 1 | 0.018 | 0.0005 |
| M63 | gate |dbeta| wide1 high 0.021 | json | D | mech/g3/wide1/gate_abs_dbeta_150/max | 1 | 0.021 | 0.0005 |
| M64 | gate |dbeta| wide low 0.038 | json | D | mech/g3/wide/gate_abs_dbeta_150/min | 1 | 0.038 | 0.0005 |
| M65 | gate |dbeta| wide high 0.041 | json | D | mech/g3/wide/gate_abs_dbeta_150/max | 1 | 0.041 | 0.0005 |
| M66 | LoRA |dbeta| LoRA/LoRA-r20 low 0.0007 | json | D | mech/g3/ctrl_ctrlpm_lora_abs_dbeta_150/min | 1 | 0.0007 | 5e-05 |
| M67 | ... high 0.0021 | json | D | mech/g3/ctrl_ctrlpm_lora_abs_dbeta_150/max | 1 | 0.0021 | 5e-05 |
| M68 | LoRA |dbeta| gated runs low 0.0023 | json | D | mech/g3/gated_lora_abs_dbeta_150/min | 1 | 0.0023 | 5e-05 |
| M69 | ... high 0.0045 | json | D | mech/g3/gated_lora_abs_dbeta_150/max | 1 | 0.0045 | 5e-05 |
| M70 | 16 times (Gate(0,1) vs LoRA arms) | json | D | mech/g3/ratio_gate_wide1_over_ctrl_lora_means | 1 | 16 | 0.5 |
| M71 | 32 times (Gate(0,2) vs LoRA arms) | json | D | mech/g3/ratio_gate_wide_over_ctrl_lora_means | 1 | 32 | 0.5 |
| M72 | 7 times (Gate(0,1) vs own LoRA) | json | D | mech/g3/ratio_gate_wide1_over_own_lora_means | 1 | 7 | 0.5 |
| M73 | 10 times (Gate(0,2) vs own LoRA) | json | D | mech/g3/ratio_gate_wide_over_own_lora_means | 1 | 10 | 0.5 |
| M74 | LoRA-r20 in same range: max LoRA-r20 0.0021 | json | GS | runs/ctrlpm_s2_150/lora_ds_abs_mean | 1 | 0.0021 | 5e-05 |
| M75 | warm-up mean s 0.41 | json | D | mech/g3/warmup_s_mean | 1 | 0.41 | 0.005 |
| M76 | warm-up s<0.1 17% | json | D | mech/g3/warmup_s_lt_0p1 | 1 | 0.17 | 0.005 |
| M77 | |tau| step 25 0.017 | json | D | mech/g3/t_abs_by_step_mean_6runs/25 | 1 | 0.017 | 0.0005 |
| M78 | |tau| step 150 0.045 | json | D | mech/g3/t_abs_by_step_mean_6runs/150 | 1 | 0.045 | 0.0005 |
| M79 | heads > 0.05 wide1 heads_gt_0p05_25/max 2 | json | D | mech/g3/wide1/heads_gt_0p05_25/max | 1 | 2 | 0 |
| M80 | heads > 0.05 wide1 heads_gt_0p05_150/min 61 | json | D | mech/g3/wide1/heads_gt_0p05_150/min | 1 | 61 | 0 |
| M81 | heads > 0.05 wide1 heads_gt_0p05_150/max 83 | json | D | mech/g3/wide1/heads_gt_0p05_150/max | 1 | 83 | 0 |
| M82 | heads > 0.05 wide heads_gt_0p05_25/min 32 | json | D | mech/g3/wide/heads_gt_0p05_25/min | 1 | 32 | 0 |
| M83 | heads > 0.05 wide heads_gt_0p05_25/max 40 | json | D | mech/g3/wide/heads_gt_0p05_25/max | 1 | 40 | 0 |
| M84 | heads > 0.05 wide heads_gt_0p05_150/min 192 | json | D | mech/g3/wide/heads_gt_0p05_150/min | 1 | 192 | 0 |
| M85 | heads > 0.05 wide heads_gt_0p05_150/max 214 | json | D | mech/g3/wide/heads_gt_0p05_150/max | 1 | 214 | 0 |
| M86 | 768 heads (24 GDN layers x 32) | text | src/gdn_gate/mech/gs_summary.py | LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30] | | | |
| M87 | Gate(0,1) layers 2-10 low 0.005 | json | D | mech/g3/wide1/layers_1_9_range/min | 1 | 0.005 | 0.0005 |
| M88 | ... high 0.015 | json | D | mech/g3/wide1/layers_1_9_range/max | 1 | 0.015 | 0.0005 |
| M89 | Gate(0,1) last 8 low 0.023 | json | D | mech/g3/wide1/last8_range/min | 1 | 0.023 | 0.0005 |
| M90 | ... high 0.032 | json | D | mech/g3/wide1/last8_range/max | 1 | 0.032 | 0.0005 |
| M91 | Gate(0,1) first GDN layer 0.039 | json | D | mech/g3/wide1/first_layer | 1 | 0.039 | 0.0005 |
| M92 | Gate(0,2) 1.5 times Gate(0,1) per layer (low) | json | D | mech/g3/wide_over_wide1_by_layer/min | 1 | 1.5 | 0.05 |
| M93 | ... 2.4 times (high) | json | D | mech/g3/wide_over_wide1_by_layer/max | 1 | 2.4 | 0.05 |
| M94 | ... 2.0 on average | json | D | mech/g3/wide_over_wide1_by_layer/mean | 1 | 2.0 | 0.05 |
| M95 | beta>1 at 150 low 1.5% | json | D | mech/g3/wide/beta_gt1_150/min | 1 | 0.015 | 0.0005 |
| M96 | ... high 1.8% | json | D | mech/g3/wide/beta_gt1_150/max | 1 | 0.018 | 0.0005 |
| M97 | beta>1 at 25 low 0.3% | json | D | mech/g3/wide/beta_gt1_25/min | 1 | 0.003 | 0.0005 |
| M98 | ... high 0.4% | json | D | mech/g3/wide/beta_gt1_25/max | 1 | 0.004 | 0.0005 |
| M99 | beta>1 last six low 2.8% | json | D | mech/g3/wide/beta_gt1_last6_range/min | 1 | 0.028 | 0.0005 |
| M100 | ... high 7.1% | json | D | mech/g3/wide/beta_gt1_last6_range/max | 1 | 0.071 | 0.0005 |
| M101 | beta>1 first layer 6.0% | json | D | mech/g3/wide/beta_gt1_first_layer | 1 | 0.06 | 0.0005 |
| M102 | beta>1 other layers at most 1.2% | json | D | mech/g3/wide/beta_gt1_layers_1_17_max | 1 | 0.012 | 0.0005 |
| M103 | Gate(0,1) never beta>1 | json | D | mech/g3/wide1/beta_gt1_150/max | 1 | 0 | 0 |
| M104 | Gate(0,2) net change low +0.022 | json | D | mech/g3/wide/gate_net_dbeta_150/min | 1 | 0.022 | 0.0005 |
| M105 | ... high +0.026 | json | D | mech/g3/wide/gate_net_dbeta_150/max | 1 | 0.026 | 0.0005 |
| M106 | Gate(0,1) net change low -0.0003 | json | D | mech/g3/wide1/gate_net_dbeta_150/min | 1 | -0.0003 | 5e-05 |
| M107 | ... high +0.0039 | json | D | mech/g3/wide1/gate_net_dbeta_150/max | 1 | 0.0039 | 5e-05 |
| M108 | largest per-token shifts on formatting tokens (descriptive) | text | results/mech/gate_stats_summary.txt | layer 22 up [('):**' | | | |
| M109 | W direction across seeds Gate(0,1) cos 0.166 | json | WS | P2_cosines/wide1_across_seeds/mean | 1 | 0.166 | 0.0005 |
| M110 | ... Gate(0,2) 0.138 | json | WS | P2_cosines/wide_across_seeds/mean | 1 | 0.138 | 0.0005 |
| M111 | null sd 0.020 | json | WS | null_cos_sd | 1 | 0.02 | 0.0005 |
| M112 | Gate(0,2) vs Gate(0,1) same seed 0.153 | json | WS | P2_cosines/wide_vs_wide1_same_seed/mean | 1 | 0.153 | 0.0005 |
| M113 | ... other seed 0.142 | json | WS | P2_cosines/wide_vs_wide1_diff_seed/mean | 1 | 0.142 | 0.0005 |
| M114 | |cos| with pretrained in_proj_b wide1_s1 0.026 | json | WS | P4_alignment/wide1_s1/cos_with_base_in_proj_b/abs_mean | 1 | 0.026 | 0.0005 |
| M115 | |cos| with pretrained in_proj_b wide_s0 0.027 | json | WS | P4_alignment/wide_s0/cos_with_base_in_proj_b/abs_mean | 1 | 0.027 | 0.0005 |
| M116 | effective rank low 19 | json | WS | P3_energy/gate/wide1_s2/eff_rank | 1 | 19 | 0.5 |
| M117 | effective rank high 23 | json | WS | P3_energy/gate/wide_s1/eff_rank | 1 | 23 | 0.5 |
| M118 | top 16 energy low 69% | json | WS | P3_energy/gate/wide_s1/top16 | 1 | 0.69 | 0.005 |
| M119 | top 16 energy high 72% | json | WS | P3_energy/gate/wide1_s2/top16 | 1 | 0.72 | 0.005 |
| M120 | GSM8K ctrl gold at token 430 | json | TS | gsm8k/ctrl/mean/tokens_to_first_gold | 1 | 430 | 0.5 |
| M121 | GSM8K ctrl after gold 1105 | json | TS | gsm8k/ctrl/mean/tokens_after_first_gold | 1 | 1105 | 0.5 |
| M122 | GSM8K ctrl share after gold 0.62 | json | TS | gsm8k/ctrl/mean/share_after_first_gold | 1 | 0.62 | 0.005 |
| M123 | GSM8K ctrlpm gold at token 400 | json | TS | gsm8k/ctrlpm/mean/tokens_to_first_gold | 1 | 400 | 0.5 |
| M124 | GSM8K ctrlpm after gold 821 | json | TS | gsm8k/ctrlpm/mean/tokens_after_first_gold | 1 | 821 | 0.5 |
| M125 | GSM8K ctrlpm share after gold 0.55 | json | TS | gsm8k/ctrlpm/mean/share_after_first_gold | 1 | 0.55 | 0.005 |
| M126 | GSM8K wide1 gold at token 200 | json | TS | gsm8k/wide1/mean/tokens_to_first_gold | 1 | 200 | 0.5 |
| M127 | GSM8K wide1 after gold 84 | json | TS | gsm8k/wide1/mean/tokens_after_first_gold | 1 | 84 | 0.5 |
| M128 | GSM8K wide1 share after gold 0.19 | json | TS | gsm8k/wide1/mean/share_after_first_gold | 1 | 0.19 | 0.005 |
| M129 | GSM8K wide gold at token 174 | json | TS | gsm8k/wide/mean/tokens_to_first_gold | 1 | 174 | 0.5 |
| M130 | GSM8K wide after gold 75 | json | TS | gsm8k/wide/mean/tokens_after_first_gold | 1 | 75 | 0.5 |
| M131 | GSM8K wide share after gold 0.17 | json | TS | gsm8k/wide/mean/share_after_first_gold | 1 | 0.17 | 0.005 |
| M132 | 82% of GSM8K saving after answer | json | D | mech/text/gsm8k/post_answer_share_of_saving_wide1_vs_ctrl | 1 | 0.82 | 0.005 |
| M133 | MATH LoRA after gold 2,198 | json | TS | math500/ctrl/mean/tokens_after_first_gold | 1 | 2198 | 0.5 |
| M134 | ... 74% | json | TS | math500/ctrl/mean/share_after_first_gold | 1 | 0.74 | 0.005 |
| M135 | MATH Gate(0,1) after gold 940 | json | TS | math500/wide1/mean/tokens_after_first_gold | 1 | 940 | 0.5 |
| M136 | ... 50% | json | TS | math500/wide1/mean/share_after_first_gold | 1 | 0.5 | 0.005 |
| M137 | 94% of MATH saving after answer | json | D | mech/text/math500/post_answer_share_of_saving_wide1_vs_ctrl | 1 | 0.94 | 0.005 |
| M138 | GSM8K wait per trace ctrl 1.81 | json | TS | gsm8k/ctrl/mean/wait_per_trace | 1 | 1.81 | 0.005 |
| M139 | GSM8K check/verify per 1k ctrl 2.58 | json | TS | gsm8k/ctrl/mean/check_verify_per_1k | 1 | 2.58 | 0.005 |
| M140 | GSM8K wait per trace ctrlpm 1.21 | json | TS | gsm8k/ctrlpm/mean/wait_per_trace | 1 | 1.21 | 0.005 |
| M141 | GSM8K check/verify per 1k ctrlpm 2.44 | json | TS | gsm8k/ctrlpm/mean/check_verify_per_1k | 1 | 2.44 | 0.005 |
| M142 | GSM8K wait per trace wide1 0.18 | json | TS | gsm8k/wide1/mean/wait_per_trace | 1 | 0.18 | 0.005 |
| M143 | GSM8K check/verify per 1k wide1 1.13 | json | TS | gsm8k/wide1/mean/check_verify_per_1k | 1 | 1.13 | 0.005 |
| M144 | GSM8K wait per trace wide 0.17 | json | TS | gsm8k/wide/mean/wait_per_trace | 1 | 0.17 | 0.005 |
| M145 | GSM8K check/verify per 1k wide 0.97 | json | TS | gsm8k/wide/mean/check_verify_per_1k | 1 | 0.97 | 0.005 |
| M146 | MATH closed share ctrl 0.75 | json | TS | math500/ctrl/mean/closed_share | 1 | 0.75 | 0.005 |
| M147 | MATH closed share ctrlpm 0.77 | json | TS | math500/ctrlpm/mean/closed_share | 1 | 0.77 | 0.005 |
| M148 | MATH closed share wide1 0.9 | json | TS | math500/wide1/mean/closed_share | 1 | 0.9 | 0.005 |
| M149 | MATH closed share wide 0.9 | json | TS | math500/wide/mean/closed_share | 1 | 0.9 | 0.005 |
| M150 | on LoRA-open samples Gate(0,1) 0.790 | json | IM | math500/ctrl_cut/wide1/acc | 1 | 0.79 | 0.0005 |
| M151 | ... LoRA 0.578 | json | IM | math500/ctrl_cut/ctrl/acc | 1 | 0.578 | 0.0005 |
| M152 | ... LoRA-r20 0.620 | json | IM | math500/ctrl_cut/ctrlpm/acc | 1 | 0.62 | 0.0005 |
| M153 | on LoRA-closed samples Gate(0,1) 0.985 | json | IM | math500/ctrl_closed/wide1/acc | 1 | 0.985 | 0.0005 |
| M154 | ... LoRA 0.993 | json | IM | math500/ctrl_closed/ctrl/acc | 1 | 0.993 | 0.0005 |
| M155 | easy third ctrl 0.996 | json | IM | math500/tertile_by_untouched_length/ctrl/0/acc | 1 | 0.996 | 0.0005 |
| M156 | middle third ctrl 0.993 | json | IM | math500/tertile_by_untouched_length/ctrl/1/acc | 1 | 0.993 | 0.0005 |
| M157 | easy third wide1 0.998 | json | IM | math500/tertile_by_untouched_length/wide1/0/acc | 1 | 0.998 | 0.0005 |
| M158 | middle third wide1 0.981 | json | IM | math500/tertile_by_untouched_length/wide1/1/acc | 1 | 0.981 | 0.0005 |
| M159 | hardest third ctrl 0.68 | json | IM | math500/tertile_by_untouched_length/ctrl/2/acc | 1 | 0.68 | 0.0005 |
| M160 | hardest third ctrlpm 0.693 | json | IM | math500/tertile_by_untouched_length/ctrlpm/2/acc | 1 | 0.693 | 0.0005 |
| M161 | hardest third wide1 0.831 | json | IM | math500/tertile_by_untouched_length/wide1/2/acc | 1 | 0.831 | 0.0005 |
| M162 | hardest third wide 0.829 | json | IM | math500/tertile_by_untouched_length/wide/2/acc | 1 | 0.829 | 0.0005 |
| M163 | hardest third Gate(0,1) reasoning 5,042 | json | IM | math500/tertile_by_untouched_length/wide1/2/reasoning | 1 | 5042 | 0.5 |
| M164 | hardest third LoRA reasoning 7,454 | json | IM | math500/tertile_by_untouched_length/ctrl/2/reasoning | 1 | 7454 | 0.5 |
| M165 | same prompt order across arms (seed 0) | json | TJ | same_prompt_order_across_arms_per_seed/0 | 1 | 1 | 0 |
| M166 | steps 0-9 natural close ctrl 0.01 (range ends) | json | D | mech/traj/natural_close_by_window_seedmean/0-10/ctrl | 1 | 0.01 | 0.0005 |
| M167 | steps 0-9 natural close wide 0.014 (range ends) | json | D | mech/traj/natural_close_by_window_seedmean/0-10/wide | 1 | 0.014 | 0.0005 |
| M168 | steps 10-25 natural close ctrl 0.01 | json | D | mech/traj/natural_close_by_window_seedmean/10-25/ctrl | 1 | 0.01 | 0.0005 |
| M169 | steps 10-25 natural close ctrlpm 0.009 | json | D | mech/traj/natural_close_by_window_seedmean/10-25/ctrlpm | 1 | 0.009 | 0.0005 |
| M170 | steps 10-25 natural close wide1 0.02 | json | D | mech/traj/natural_close_by_window_seedmean/10-25/wide1 | 1 | 0.02 | 0.0005 |
| M171 | steps 10-25 natural close wide 0.018 | json | D | mech/traj/natural_close_by_window_seedmean/10-25/wide | 1 | 0.018 | 0.0005 |
| M172 | steps 25-50 natural close ctrl 0.015 | json | D | mech/traj/natural_close_by_window_seedmean/25-50/ctrl | 1 | 0.015 | 0.0005 |
| M173 | steps 25-50 natural close ctrlpm 0.015 | json | D | mech/traj/natural_close_by_window_seedmean/25-50/ctrlpm | 1 | 0.015 | 0.0005 |
| M174 | steps 25-50 natural close wide1 0.063 | json | D | mech/traj/natural_close_by_window_seedmean/25-50/wide1 | 1 | 0.063 | 0.0005 |
| M175 | steps 25-50 natural close wide 0.079 | json | D | mech/traj/natural_close_by_window_seedmean/25-50/wide | 1 | 0.079 | 0.0005 |
| M176 | steps 50-75 natural close ctrl 0.023 | json | D | mech/traj/natural_close_by_window_seedmean/50-75/ctrl | 1 | 0.023 | 0.0005 |
| M177 | steps 50-75 natural close ctrlpm 0.036 | json | D | mech/traj/natural_close_by_window_seedmean/50-75/ctrlpm | 1 | 0.036 | 0.0005 |
| M178 | steps 50-75 natural close wide1 0.216 | json | D | mech/traj/natural_close_by_window_seedmean/50-75/wide1 | 1 | 0.216 | 0.0005 |
| M179 | steps 50-75 natural close wide 0.272 | json | D | mech/traj/natural_close_by_window_seedmean/50-75/wide | 1 | 0.272 | 0.0005 |
| M180 | groups to 10% close: gated low 108 | json | D | mech/traj/cum_groups_to_0p1_gated/min | 1 | 108 | 0 |
| M181 | ... gated high 186 | json | D | mech/traj/cum_groups_to_0p1_gated/max | 1 | 186 | 0 |
| M182 | groups to 10% close ctrl s1 251 | json | D | mech/traj/cum_groups_to_0p1_ctrl/1 | 1 | 251 | 0 |
| M183 | groups to 10% close ctrl s2 252 | json | D | mech/traj/cum_groups_to_0p1_ctrl/2 | 1 | 252 | 0 |
| M184 | groups to 10% close ctrlpm s1 269 | json | D | mech/traj/cum_groups_to_0p1_ctrlpm/1 | 1 | 269 | 0 |
| M185 | groups to 10% close ctrlpm s2 238 | json | D | mech/traj/cum_groups_to_0p1_ctrlpm/2 | 1 | 238 | 0 |
| M186 | seed 0 of LoRA and LoRA-r20 never reach 10% | json | D | mech/traj/s0_never_reach_0p1_ctrl_and_ctrlpm | 1 | 1 | 0 |
| M187 | rollout accuracy last 20 low 0.92 | json | D | mech/traj/acc_last20_range_all/min | 1 | 0.92 | 0.005 |
| M188 | ... high 0.96 | json | D | mech/traj/acc_last20_range_all/max | 1 | 0.96 | 0.005 |

Derived: the compute total, about 210 GPU-hours of box time, is 8 x 9.47 h + 8 x 10.3 h + 4 x 3.66 h + 4 x 4.9 h + 8 x 2.2 h (rows P31-P33, N163, N164; row W30 checks the sum in CH). The fresh-set 4B length cut 1,329 / 7,424 = 18% is `D` fresh/4b/wide1_vs_ctrl_len_cut (not printed in the paper).

## Claims without a receipt file

- None of the numbers in the paper lacks a receipt. These statements rest on prose records or on the cited papers
  rather than on a JSON field:
  - the compute hours (P31 to P33, N163, N164): the rental log, summarized in docs/COMPUTE.md;
  - "four from-scratch comparisons ... found likelihood within about 1%" (Background): read from the four cited
    papers;
  - "no paper does either" (Related work): a census of forward citations and recent preprints through 2026-10-11,
    kept in the author's notes, not released.

## Revision rows (2026-10-12): new intervals, censoring-aware length, splits, ratios, rounding, records

V rows are generated from paper/data/extra.json (one row per number shown in Tables 3 and 5, the item splits, the G1 split and the ratios). W rows are written by hand.

| id | paper text | kind | file | path or string | scale | expect | tol |
|---|---|---|---|---|---|---|---|
| V1 | Table 3 4b_math500_area_wide-ctrl diff 0.063 | json | X | contrasts/4b_math500_area_wide-ctrl/seed_mean_diff | 1 | 0.063 | 0.0005 |
| V2 | Table 3 4b_math500_area_wide-ctrl hier lo 0.04 | json | X | contrasts/4b_math500_area_wide-ctrl/hier_ci95/0 | 1 | 0.04 | 0.0005 |
| V3 | Table 3 4b_math500_area_wide-ctrl hier hi 0.086 | json | X | contrasts/4b_math500_area_wide-ctrl/hier_ci95/1 | 1 | 0.086 | 0.0005 |
| V4 | Table 3 4b_math500_area_wide-ctrl t lo 0.019 | json | X | contrasts/4b_math500_area_wide-ctrl/t_ci95/0 | 1 | 0.019 | 0.0005 |
| V5 | Table 3 4b_math500_area_wide-ctrl t hi 0.107 | json | X | contrasts/4b_math500_area_wide-ctrl/t_ci95/1 | 1 | 0.107 | 0.0005 |
| V6 | 4b_math500_area_wide-ctrl positive seeds | json | X | contrasts/4b_math500_area_wide-ctrl/n_pos | 1 | 3 | 0 |
| V7 | Table 3 4b_math500_area_wide1-ctrl diff 0.065 | json | X | contrasts/4b_math500_area_wide1-ctrl/seed_mean_diff | 1 | 0.065 | 0.0005 |
| V8 | Table 3 4b_math500_area_wide1-ctrl hier lo 0.049 | json | X | contrasts/4b_math500_area_wide1-ctrl/hier_ci95/0 | 1 | 0.049 | 0.0005 |
| V9 | Table 3 4b_math500_area_wide1-ctrl hier hi 0.083 | json | X | contrasts/4b_math500_area_wide1-ctrl/hier_ci95/1 | 1 | 0.083 | 0.0005 |
| V10 | Table 3 4b_math500_area_wide1-ctrl t lo 0.037 | json | X | contrasts/4b_math500_area_wide1-ctrl/t_ci95/0 | 1 | 0.037 | 0.0005 |
| V11 | Table 3 4b_math500_area_wide1-ctrl t hi 0.094 | json | X | contrasts/4b_math500_area_wide1-ctrl/t_ci95/1 | 1 | 0.094 | 0.0005 |
| V12 | 4b_math500_area_wide1-ctrl positive seeds | json | X | contrasts/4b_math500_area_wide1-ctrl/n_pos | 1 | 3 | 0 |
| V13 | Table 3 4b_math500_area_wide1-ctrlpm diff 0.06 | json | X | contrasts/4b_math500_area_wide1-ctrlpm/seed_mean_diff | 1 | 0.06 | 0.0005 |
| V14 | Table 3 4b_math500_area_wide1-ctrlpm hier lo 0.037 | json | X | contrasts/4b_math500_area_wide1-ctrlpm/hier_ci95/0 | 1 | 0.037 | 0.0005 |
| V15 | Table 3 4b_math500_area_wide1-ctrlpm hier hi 0.086 | json | X | contrasts/4b_math500_area_wide1-ctrlpm/hier_ci95/1 | 1 | 0.086 | 0.0005 |
| V16 | Table 3 4b_math500_area_wide1-ctrlpm t lo 0.003 | json | X | contrasts/4b_math500_area_wide1-ctrlpm/t_ci95/0 | 1 | 0.003 | 0.0005 |
| V17 | Table 3 4b_math500_area_wide1-ctrlpm t hi 0.117 | json | X | contrasts/4b_math500_area_wide1-ctrlpm/t_ci95/1 | 1 | 0.117 | 0.0005 |
| V18 | 4b_math500_area_wide1-ctrlpm positive seeds | json | X | contrasts/4b_math500_area_wide1-ctrlpm/n_pos | 1 | 3 | 0 |
| V19 | Table 3 4b_math500_area_wide-wide1 diff -0.002 | json | X | contrasts/4b_math500_area_wide-wide1/seed_mean_diff | 1 | -0.002 | 0.0005 |
| V20 | Table 3 4b_math500_area_wide-wide1 hier lo -0.017 | json | X | contrasts/4b_math500_area_wide-wide1/hier_ci95/0 | 1 | -0.017 | 0.0005 |
| V21 | Table 3 4b_math500_area_wide-wide1 hier hi 0.011 | json | X | contrasts/4b_math500_area_wide-wide1/hier_ci95/1 | 1 | 0.011 | 0.0005 |
| V22 | Table 3 4b_math500_area_wide-wide1 t lo -0.027 | json | X | contrasts/4b_math500_area_wide-wide1/t_ci95/0 | 1 | -0.027 | 0.0005 |
| V23 | Table 3 4b_math500_area_wide-wide1 t hi 0.022 | json | X | contrasts/4b_math500_area_wide-wide1/t_ci95/1 | 1 | 0.022 | 0.0005 |
| V24 | 4b_math500_area_wide-wide1 positive seeds | json | X | contrasts/4b_math500_area_wide-wide1/n_pos | 1 | 2 | 0 |
| V25 | Table 3 4b_math500_area_ctrlpm-ctrl diff 0.005 | json | X | contrasts/4b_math500_area_ctrlpm-ctrl/seed_mean_diff | 1 | 0.005 | 0.0005 |
| V26 | Table 3 4b_math500_area_ctrlpm-ctrl hier lo -0.009 | json | X | contrasts/4b_math500_area_ctrlpm-ctrl/hier_ci95/0 | 1 | -0.009 | 0.0005 |
| V27 | Table 3 4b_math500_area_ctrlpm-ctrl hier hi 0.02 | json | X | contrasts/4b_math500_area_ctrlpm-ctrl/hier_ci95/1 | 1 | 0.02 | 0.0005 |
| V28 | Table 3 4b_math500_area_ctrlpm-ctrl t lo -0.028 | json | X | contrasts/4b_math500_area_ctrlpm-ctrl/t_ci95/0 | 1 | -0.028 | 0.0005 |
| V29 | Table 3 4b_math500_area_ctrlpm-ctrl t hi 0.038 | json | X | contrasts/4b_math500_area_ctrlpm-ctrl/t_ci95/1 | 1 | 0.038 | 0.0005 |
| V30 | 4b_math500_area_ctrlpm-ctrl positive seeds | json | X | contrasts/4b_math500_area_ctrlpm-ctrl/n_pos | 1 | 2 | 0 |
| V31 | Table 3 4b_math500_len_wide1-ctrl diff -2021 | json | X | contrasts/4b_math500_len_wide1-ctrl/seed_mean_diff | 1 | -2021 | 0.5 |
| V32 | Table 3 4b_math500_len_wide1-ctrl hier lo -2257 | json | X | contrasts/4b_math500_len_wide1-ctrl/hier_ci95/0 | 1 | -2257 | 0.5 |
| V33 | Table 3 4b_math500_len_wide1-ctrl hier hi -1755 | json | X | contrasts/4b_math500_len_wide1-ctrl/hier_ci95/1 | 1 | -1755 | 0.5 |
| V34 | Table 3 4b_math500_len_wide1-ctrl t lo -2626 | json | X | contrasts/4b_math500_len_wide1-ctrl/t_ci95/0 | 1 | -2626 | 0.5 |
| V35 | Table 3 4b_math500_len_wide1-ctrl t hi -1417 | json | X | contrasts/4b_math500_len_wide1-ctrl/t_ci95/1 | 1 | -1417 | 0.5 |
| V36 | 4b_math500_len_wide1-ctrl positive seeds | json | X | contrasts/4b_math500_len_wide1-ctrl/n_pos | 1 | 0 | 0 |
| V37 | Table 3 4b_math500_len_wide-wide1 diff -49 | json | X | contrasts/4b_math500_len_wide-wide1/seed_mean_diff | 1 | -49 | 0.5 |
| V38 | Table 3 4b_math500_len_wide-wide1 hier lo -215 | json | X | contrasts/4b_math500_len_wide-wide1/hier_ci95/0 | 1 | -215 | 0.5 |
| V39 | Table 3 4b_math500_len_wide-wide1 hier hi 116 | json | X | contrasts/4b_math500_len_wide-wide1/hier_ci95/1 | 1 | 116 | 0.5 |
| V40 | Table 3 4b_math500_len_wide-wide1 t lo -476 | json | X | contrasts/4b_math500_len_wide-wide1/t_ci95/0 | 1 | -476 | 0.5 |
| V41 | Table 3 4b_math500_len_wide-wide1 t hi 378 | json | X | contrasts/4b_math500_len_wide-wide1/t_ci95/1 | 1 | 378 | 0.5 |
| V42 | 4b_math500_len_wide-wide1 positive seeds | json | X | contrasts/4b_math500_len_wide-wide1/n_pos | 1 | 1 | 0 |
| V43 | Table 3 4b_gsm8k_len_wide-ctrl diff -1405 | json | X | contrasts/4b_gsm8k_len_wide-ctrl/seed_mean_diff | 1 | -1405 | 0.5 |
| V44 | Table 3 4b_gsm8k_len_wide-ctrl hier lo -1707 | json | X | contrasts/4b_gsm8k_len_wide-ctrl/hier_ci95/0 | 1 | -1707 | 0.5 |
| V45 | Table 3 4b_gsm8k_len_wide-ctrl hier hi -1189 | json | X | contrasts/4b_gsm8k_len_wide-ctrl/hier_ci95/1 | 1 | -1189 | 0.5 |
| V46 | Table 3 4b_gsm8k_len_wide-ctrl t lo -2114 | json | X | contrasts/4b_gsm8k_len_wide-ctrl/t_ci95/0 | 1 | -2114 | 0.5 |
| V47 | Table 3 4b_gsm8k_len_wide-ctrl t hi -697 | json | X | contrasts/4b_gsm8k_len_wide-ctrl/t_ci95/1 | 1 | -697 | 0.5 |
| V48 | 4b_gsm8k_len_wide-ctrl positive seeds | json | X | contrasts/4b_gsm8k_len_wide-ctrl/n_pos | 1 | 0 | 0 |
| V49 | Table 3 4b_gsm8k_len_wide1-ctrl diff -1370 | json | X | contrasts/4b_gsm8k_len_wide1-ctrl/seed_mean_diff | 1 | -1370 | 0.5 |
| V50 | Table 3 4b_gsm8k_len_wide1-ctrl hier lo -1612 | json | X | contrasts/4b_gsm8k_len_wide1-ctrl/hier_ci95/0 | 1 | -1612 | 0.5 |
| V51 | Table 3 4b_gsm8k_len_wide1-ctrl hier hi -1191 | json | X | contrasts/4b_gsm8k_len_wide1-ctrl/hier_ci95/1 | 1 | -1191 | 0.5 |
| V52 | Table 3 4b_gsm8k_len_wide1-ctrl t lo -1942 | json | X | contrasts/4b_gsm8k_len_wide1-ctrl/t_ci95/0 | 1 | -1942 | 0.5 |
| V53 | Table 3 4b_gsm8k_len_wide1-ctrl t hi -798 | json | X | contrasts/4b_gsm8k_len_wide1-ctrl/t_ci95/1 | 1 | -798 | 0.5 |
| V54 | 4b_gsm8k_len_wide1-ctrl positive seeds | json | X | contrasts/4b_gsm8k_len_wide1-ctrl/n_pos | 1 | 0 | 0 |
| V55 | Table 3 4b_gsm8k_len_wide1-ctrlpm diff -1000 | json | X | contrasts/4b_gsm8k_len_wide1-ctrlpm/seed_mean_diff | 1 | -1000 | 0.5 |
| V56 | Table 3 4b_gsm8k_len_wide1-ctrlpm hier lo -1731 | json | X | contrasts/4b_gsm8k_len_wide1-ctrlpm/hier_ci95/0 | 1 | -1731 | 0.5 |
| V57 | Table 3 4b_gsm8k_len_wide1-ctrlpm hier hi -527 | json | X | contrasts/4b_gsm8k_len_wide1-ctrlpm/hier_ci95/1 | 1 | -527 | 0.5 |
| V58 | Table 3 4b_gsm8k_len_wide1-ctrlpm t lo -2650 | json | X | contrasts/4b_gsm8k_len_wide1-ctrlpm/t_ci95/0 | 1 | -2650 | 0.5 |
| V59 | Table 3 4b_gsm8k_len_wide1-ctrlpm t hi 651 | json | X | contrasts/4b_gsm8k_len_wide1-ctrlpm/t_ci95/1 | 1 | 651 | 0.5 |
| V60 | 4b_gsm8k_len_wide1-ctrlpm positive seeds | json | X | contrasts/4b_gsm8k_len_wide1-ctrlpm/n_pos | 1 | 0 | 0 |
| V61 | Table 3 4b_gsm8k_len_wide-wide1 diff -35 | json | X | contrasts/4b_gsm8k_len_wide-wide1/seed_mean_diff | 1 | -35 | 0.5 |
| V62 | Table 3 4b_gsm8k_len_wide-wide1 hier lo -92 | json | X | contrasts/4b_gsm8k_len_wide-wide1/hier_ci95/0 | 1 | -92 | 0.5 |
| V63 | Table 3 4b_gsm8k_len_wide-wide1 hier hi 30 | json | X | contrasts/4b_gsm8k_len_wide-wide1/hier_ci95/1 | 1 | 30 | 0.5 |
| V64 | Table 3 4b_gsm8k_len_wide-wide1 t lo -198 | json | X | contrasts/4b_gsm8k_len_wide-wide1/t_ci95/0 | 1 | -198 | 0.5 |
| V65 | Table 3 4b_gsm8k_len_wide-wide1 t hi 127 | json | X | contrasts/4b_gsm8k_len_wide-wide1/t_ci95/1 | 1 | 127 | 0.5 |
| V66 | 4b_gsm8k_len_wide-wide1 positive seeds | json | X | contrasts/4b_gsm8k_len_wide-wide1/n_pos | 1 | 1 | 0 |
| V67 | Table 3 4b_gsm8k_area_ctrlpm-ctrl diff 0.006 | json | X | contrasts/4b_gsm8k_area_ctrlpm-ctrl/seed_mean_diff | 1 | 0.006 | 0.0005 |
| V68 | Table 3 4b_gsm8k_area_ctrlpm-ctrl hier lo 0.001 | json | X | contrasts/4b_gsm8k_area_ctrlpm-ctrl/hier_ci95/0 | 1 | 0.001 | 0.0005 |
| V69 | Table 3 4b_gsm8k_area_ctrlpm-ctrl hier hi 0.012 | json | X | contrasts/4b_gsm8k_area_ctrlpm-ctrl/hier_ci95/1 | 1 | 0.012 | 0.0005 |
| V70 | Table 3 4b_gsm8k_area_ctrlpm-ctrl t lo -0.003 | json | X | contrasts/4b_gsm8k_area_ctrlpm-ctrl/t_ci95/0 | 1 | -0.003 | 0.0005 |
| V71 | Table 3 4b_gsm8k_area_ctrlpm-ctrl t hi 0.015 | json | X | contrasts/4b_gsm8k_area_ctrlpm-ctrl/t_ci95/1 | 1 | 0.015 | 0.0005 |
| V72 | 4b_gsm8k_area_ctrlpm-ctrl positive seeds | json | X | contrasts/4b_gsm8k_area_ctrlpm-ctrl/n_pos | 1 | 3 | 0 |
| V73 | Table 3 4b_fresh_area_wide1-ctrl diff 0.11 | json | X | contrasts/4b_fresh_area_wide1-ctrl/seed_mean_diff | 1 | 0.11 | 0.0005 |
| V74 | Table 3 4b_fresh_area_wide1-ctrl hier lo 0.077 | json | X | contrasts/4b_fresh_area_wide1-ctrl/hier_ci95/0 | 1 | 0.077 | 0.0005 |
| V75 | Table 3 4b_fresh_area_wide1-ctrl hier hi 0.145 | json | X | contrasts/4b_fresh_area_wide1-ctrl/hier_ci95/1 | 1 | 0.145 | 0.0005 |
| V76 | Table 3 4b_fresh_area_wide1-ctrl t lo 0.043 | json | X | contrasts/4b_fresh_area_wide1-ctrl/t_ci95/0 | 1 | 0.043 | 0.0005 |
| V77 | Table 3 4b_fresh_area_wide1-ctrl t hi 0.178 | json | X | contrasts/4b_fresh_area_wide1-ctrl/t_ci95/1 | 1 | 0.178 | 0.0005 |
| V78 | 4b_fresh_area_wide1-ctrl positive seeds | json | X | contrasts/4b_fresh_area_wide1-ctrl/n_pos | 1 | 3 | 0 |
| V79 | Table 3 4b_fresh_len_wide1-ctrl diff -1329 | json | X | contrasts/4b_fresh_len_wide1-ctrl/seed_mean_diff | 1 | -1329 | 0.5 |
| V80 | Table 3 4b_fresh_len_wide1-ctrl hier lo -1693 | json | X | contrasts/4b_fresh_len_wide1-ctrl/hier_ci95/0 | 1 | -1693 | 0.5 |
| V81 | Table 3 4b_fresh_len_wide1-ctrl hier hi -989 | json | X | contrasts/4b_fresh_len_wide1-ctrl/hier_ci95/1 | 1 | -989 | 0.5 |
| V82 | Table 3 4b_fresh_len_wide1-ctrl t lo -1977 | json | X | contrasts/4b_fresh_len_wide1-ctrl/t_ci95/0 | 1 | -1977 | 0.5 |
| V83 | Table 3 4b_fresh_len_wide1-ctrl t hi -682 | json | X | contrasts/4b_fresh_len_wide1-ctrl/t_ci95/1 | 1 | -682 | 0.5 |
| V84 | 4b_fresh_len_wide1-ctrl positive seeds | json | X | contrasts/4b_fresh_len_wide1-ctrl/n_pos | 1 | 0 | 0 |
| V85 | Table 3 4b_fresh_area_wide1-ctrlpm diff 0.098 | json | X | contrasts/4b_fresh_area_wide1-ctrlpm/seed_mean_diff | 1 | 0.098 | 0.0005 |
| V86 | Table 3 4b_fresh_area_wide1-ctrlpm hier lo 0.063 | json | X | contrasts/4b_fresh_area_wide1-ctrlpm/hier_ci95/0 | 1 | 0.063 | 0.0005 |
| V87 | Table 3 4b_fresh_area_wide1-ctrlpm hier hi 0.142 | json | X | contrasts/4b_fresh_area_wide1-ctrlpm/hier_ci95/1 | 1 | 0.142 | 0.0005 |
| V88 | Table 3 4b_fresh_area_wide1-ctrlpm t lo 0.007 | json | X | contrasts/4b_fresh_area_wide1-ctrlpm/t_ci95/0 | 1 | 0.007 | 0.0005 |
| V89 | Table 3 4b_fresh_area_wide1-ctrlpm t hi 0.19 | json | X | contrasts/4b_fresh_area_wide1-ctrlpm/t_ci95/1 | 1 | 0.19 | 0.0005 |
| V90 | 4b_fresh_area_wide1-ctrlpm positive seeds | json | X | contrasts/4b_fresh_area_wide1-ctrlpm/n_pos | 1 | 3 | 0 |
| V91 | Table 3 4b_fresh_len_wide1-ctrlpm diff -1231 | json | X | contrasts/4b_fresh_len_wide1-ctrlpm/seed_mean_diff | 1 | -1231 | 0.5 |
| V92 | Table 3 4b_fresh_len_wide1-ctrlpm hier lo -1588 | json | X | contrasts/4b_fresh_len_wide1-ctrlpm/hier_ci95/0 | 1 | -1588 | 0.5 |
| V93 | Table 3 4b_fresh_len_wide1-ctrlpm hier hi -881 | json | X | contrasts/4b_fresh_len_wide1-ctrlpm/hier_ci95/1 | 1 | -881 | 0.5 |
| V94 | Table 3 4b_fresh_len_wide1-ctrlpm t lo -1927 | json | X | contrasts/4b_fresh_len_wide1-ctrlpm/t_ci95/0 | 1 | -1927 | 0.5 |
| V95 | Table 3 4b_fresh_len_wide1-ctrlpm t hi -535 | json | X | contrasts/4b_fresh_len_wide1-ctrlpm/t_ci95/1 | 1 | -535 | 0.5 |
| V96 | 4b_fresh_len_wide1-ctrlpm positive seeds | json | X | contrasts/4b_fresh_len_wide1-ctrlpm/n_pos | 1 | 0 | 0 |
| V97 | Table 3 4b_fresh_area_wide-wide1 diff 0.012 | json | X | contrasts/4b_fresh_area_wide-wide1/seed_mean_diff | 1 | 0.012 | 0.0005 |
| V98 | Table 3 4b_fresh_area_wide-wide1 hier lo 0.001 | json | X | contrasts/4b_fresh_area_wide-wide1/hier_ci95/0 | 1 | 0.001 | 0.0005 |
| V99 | Table 3 4b_fresh_area_wide-wide1 hier hi 0.023 | json | X | contrasts/4b_fresh_area_wide-wide1/hier_ci95/1 | 1 | 0.023 | 0.0005 |
| V100 | Table 3 4b_fresh_area_wide-wide1 t lo -0.001 | json | X | contrasts/4b_fresh_area_wide-wide1/t_ci95/0 | 1 | -0.001 | 0.0005 |
| V101 | Table 3 4b_fresh_area_wide-wide1 t hi 0.025 | json | X | contrasts/4b_fresh_area_wide-wide1/t_ci95/1 | 1 | 0.025 | 0.0005 |
| V102 | 4b_fresh_area_wide-wide1 positive seeds | json | X | contrasts/4b_fresh_area_wide-wide1/n_pos | 1 | 3 | 0 |
| V103 | Table 3 4b_fresh_len_wide-wide1 diff -113 | json | X | contrasts/4b_fresh_len_wide-wide1/seed_mean_diff | 1 | -113 | 0.5 |
| V104 | Table 3 4b_fresh_len_wide-wide1 hier lo -252 | json | X | contrasts/4b_fresh_len_wide-wide1/hier_ci95/0 | 1 | -252 | 0.5 |
| V105 | Table 3 4b_fresh_len_wide-wide1 hier hi 55 | json | X | contrasts/4b_fresh_len_wide-wide1/hier_ci95/1 | 1 | 55 | 0.5 |
| V106 | Table 3 4b_fresh_len_wide-wide1 t lo -496 | json | X | contrasts/4b_fresh_len_wide-wide1/t_ci95/0 | 1 | -496 | 0.5 |
| V107 | Table 3 4b_fresh_len_wide-wide1 t hi 270 | json | X | contrasts/4b_fresh_len_wide-wide1/t_ci95/1 | 1 | 270 | 0.5 |
| V108 | 4b_fresh_len_wide-wide1 positive seeds | json | X | contrasts/4b_fresh_len_wide-wide1/n_pos | 1 | 1 | 0 |
| V109 | Table 3 2b_math500_area_wide-ctrl_r1 diff 0.027 | json | X | contrasts/2b_math500_area_wide-ctrl_r1/seed_mean_diff | 1 | 0.027 | 0.0005 |
| V110 | Table 3 2b_math500_area_wide-ctrl_r1 hier lo 0.004 | json | X | contrasts/2b_math500_area_wide-ctrl_r1/hier_ci95/0 | 1 | 0.004 | 0.0005 |
| V111 | Table 3 2b_math500_area_wide-ctrl_r1 hier hi 0.051 | json | X | contrasts/2b_math500_area_wide-ctrl_r1/hier_ci95/1 | 1 | 0.051 | 0.0005 |
| V112 | Table 3 2b_math500_area_wide-ctrl_r1 t lo -0.03 | json | X | contrasts/2b_math500_area_wide-ctrl_r1/t_ci95/0 | 1 | -0.03 | 0.0005 |
| V113 | Table 3 2b_math500_area_wide-ctrl_r1 t hi 0.084 | json | X | contrasts/2b_math500_area_wide-ctrl_r1/t_ci95/1 | 1 | 0.084 | 0.0005 |
| V114 | 2b_math500_area_wide-ctrl_r1 positive seeds | json | X | contrasts/2b_math500_area_wide-ctrl_r1/n_pos | 1 | 3 | 0 |
| V115 | Table 3 2b_math500_area_wide1-ctrl_r1 diff 0.021 | json | X | contrasts/2b_math500_area_wide1-ctrl_r1/seed_mean_diff | 1 | 0.021 | 0.0005 |
| V116 | Table 3 2b_math500_area_wide1-ctrl_r1 hier lo 0.0 | json | X | contrasts/2b_math500_area_wide1-ctrl_r1/hier_ci95/0 | 1 | 0.0 | 0.0005 |
| V117 | Table 3 2b_math500_area_wide1-ctrl_r1 hier hi 0.039 | json | X | contrasts/2b_math500_area_wide1-ctrl_r1/hier_ci95/1 | 1 | 0.039 | 0.0005 |
| V118 | Table 3 2b_math500_area_wide1-ctrl_r1 t lo -0.026 | json | X | contrasts/2b_math500_area_wide1-ctrl_r1/t_ci95/0 | 1 | -0.026 | 0.0005 |
| V119 | Table 3 2b_math500_area_wide1-ctrl_r1 t hi 0.067 | json | X | contrasts/2b_math500_area_wide1-ctrl_r1/t_ci95/1 | 1 | 0.067 | 0.0005 |
| V120 | 2b_math500_area_wide1-ctrl_r1 positive seeds | json | X | contrasts/2b_math500_area_wide1-ctrl_r1/n_pos | 1 | 2 | 0 |
| V121 | Table 3 2b_math500_area_wide-ctrl_r2 diff 0.025 | json | X | contrasts/2b_math500_area_wide-ctrl_r2/seed_mean_diff | 1 | 0.025 | 0.0005 |
| V122 | Table 3 2b_math500_area_wide-ctrl_r2 hier lo 0.001 | json | X | contrasts/2b_math500_area_wide-ctrl_r2/hier_ci95/0 | 1 | 0.001 | 0.0005 |
| V123 | Table 3 2b_math500_area_wide-ctrl_r2 hier hi 0.048 | json | X | contrasts/2b_math500_area_wide-ctrl_r2/hier_ci95/1 | 1 | 0.048 | 0.0005 |
| V124 | Table 3 2b_math500_area_wide-ctrl_r2 t lo -0.034 | json | X | contrasts/2b_math500_area_wide-ctrl_r2/t_ci95/0 | 1 | -0.034 | 0.0005 |
| V125 | Table 3 2b_math500_area_wide-ctrl_r2 t hi 0.084 | json | X | contrasts/2b_math500_area_wide-ctrl_r2/t_ci95/1 | 1 | 0.084 | 0.0005 |
| V126 | 2b_math500_area_wide-ctrl_r2 positive seeds | json | X | contrasts/2b_math500_area_wide-ctrl_r2/n_pos | 1 | 3 | 0 |
| V127 | Table 3 2b_math500_area_wide1-ctrl_r2 diff 0.021 | json | X | contrasts/2b_math500_area_wide1-ctrl_r2/seed_mean_diff | 1 | 0.021 | 0.0005 |
| V128 | Table 3 2b_math500_area_wide1-ctrl_r2 hier lo 0.001 | json | X | contrasts/2b_math500_area_wide1-ctrl_r2/hier_ci95/0 | 1 | 0.001 | 0.0005 |
| V129 | Table 3 2b_math500_area_wide1-ctrl_r2 hier hi 0.04 | json | X | contrasts/2b_math500_area_wide1-ctrl_r2/hier_ci95/1 | 1 | 0.04 | 0.0005 |
| V130 | Table 3 2b_math500_area_wide1-ctrl_r2 t lo -0.025 | json | X | contrasts/2b_math500_area_wide1-ctrl_r2/t_ci95/0 | 1 | -0.025 | 0.0005 |
| V131 | Table 3 2b_math500_area_wide1-ctrl_r2 t hi 0.067 | json | X | contrasts/2b_math500_area_wide1-ctrl_r2/t_ci95/1 | 1 | 0.067 | 0.0005 |
| V132 | 2b_math500_area_wide1-ctrl_r2 positive seeds | json | X | contrasts/2b_math500_area_wide1-ctrl_r2/n_pos | 1 | 3 | 0 |
| V133 | Table 3 2b_gsm8k_len_wide-ctrl_r2 diff -997 | json | X | contrasts/2b_gsm8k_len_wide-ctrl_r2/seed_mean_diff | 1 | -997 | 0.5 |
| V134 | Table 3 2b_gsm8k_len_wide-ctrl_r2 hier lo -1294 | json | X | contrasts/2b_gsm8k_len_wide-ctrl_r2/hier_ci95/0 | 1 | -1294 | 0.5 |
| V135 | Table 3 2b_gsm8k_len_wide-ctrl_r2 hier hi -565 | json | X | contrasts/2b_gsm8k_len_wide-ctrl_r2/hier_ci95/1 | 1 | -565 | 0.5 |
| V136 | Table 3 2b_gsm8k_len_wide-ctrl_r2 t lo -1987 | json | X | contrasts/2b_gsm8k_len_wide-ctrl_r2/t_ci95/0 | 1 | -1987 | 0.5 |
| V137 | Table 3 2b_gsm8k_len_wide-ctrl_r2 t hi -8 | json | X | contrasts/2b_gsm8k_len_wide-ctrl_r2/t_ci95/1 | 1 | -8 | 0.5 |
| V138 | 2b_gsm8k_len_wide-ctrl_r2 positive seeds | json | X | contrasts/2b_gsm8k_len_wide-ctrl_r2/n_pos | 1 | 0 | 0 |
| V139 | Table 3 2b_fresh_area_wide1-ctrl diff 0.012 | json | X | contrasts/2b_fresh_area_wide1-ctrl/seed_mean_diff | 1 | 0.012 | 0.0005 |
| V140 | Table 3 2b_fresh_area_wide1-ctrl hier lo 0.001 | json | X | contrasts/2b_fresh_area_wide1-ctrl/hier_ci95/0 | 1 | 0.001 | 0.0005 |
| V141 | Table 3 2b_fresh_area_wide1-ctrl hier hi 0.024 | json | X | contrasts/2b_fresh_area_wide1-ctrl/hier_ci95/1 | 1 | 0.024 | 0.0005 |
| V142 | Table 3 2b_fresh_area_wide1-ctrl t lo -0.002 | json | X | contrasts/2b_fresh_area_wide1-ctrl/t_ci95/0 | 1 | -0.002 | 0.0005 |
| V143 | Table 3 2b_fresh_area_wide1-ctrl t hi 0.025 | json | X | contrasts/2b_fresh_area_wide1-ctrl/t_ci95/1 | 1 | 0.025 | 0.0005 |
| V144 | 2b_fresh_area_wide1-ctrl positive seeds | json | X | contrasts/2b_fresh_area_wide1-ctrl/n_pos | 1 | 4 | 0 |
| V145 | Table 3 2b_fresh_area_wide-wide1 diff 0.011 | json | X | contrasts/2b_fresh_area_wide-wide1/seed_mean_diff | 1 | 0.011 | 0.0005 |
| V146 | Table 3 2b_fresh_area_wide-wide1 hier lo -0.007 | json | X | contrasts/2b_fresh_area_wide-wide1/hier_ci95/0 | 1 | -0.007 | 0.0005 |
| V147 | Table 3 2b_fresh_area_wide-wide1 hier hi 0.028 | json | X | contrasts/2b_fresh_area_wide-wide1/hier_ci95/1 | 1 | 0.028 | 0.0005 |
| V148 | Table 3 2b_fresh_area_wide-wide1 t lo -0.013 | json | X | contrasts/2b_fresh_area_wide-wide1/t_ci95/0 | 1 | -0.013 | 0.0005 |
| V149 | Table 3 2b_fresh_area_wide-wide1 t hi 0.034 | json | X | contrasts/2b_fresh_area_wide-wide1/t_ci95/1 | 1 | 0.034 | 0.0005 |
| V150 | 2b_fresh_area_wide-wide1 positive seeds | json | X | contrasts/2b_fresh_area_wide-wide1/n_pos | 1 | 4 | 0 |
| V151 | Table 3 2b_fresh_len_wide-ctrl diff -307 | json | X | contrasts/2b_fresh_len_wide-ctrl/seed_mean_diff | 1 | -307 | 0.5 |
| V152 | Table 3 2b_fresh_len_wide-ctrl hier lo -477 | json | X | contrasts/2b_fresh_len_wide-ctrl/hier_ci95/0 | 1 | -477 | 0.5 |
| V153 | Table 3 2b_fresh_len_wide-ctrl hier hi -155 | json | X | contrasts/2b_fresh_len_wide-ctrl/hier_ci95/1 | 1 | -155 | 0.5 |
| V154 | Table 3 2b_fresh_len_wide-ctrl t lo -488 | json | X | contrasts/2b_fresh_len_wide-ctrl/t_ci95/0 | 1 | -488 | 0.5 |
| V155 | Table 3 2b_fresh_len_wide-ctrl t hi -126 | json | X | contrasts/2b_fresh_len_wide-ctrl/t_ci95/1 | 1 | -126 | 0.5 |
| V156 | 2b_fresh_len_wide-ctrl positive seeds | json | X | contrasts/2b_fresh_len_wide-ctrl/n_pos | 1 | 0 | 0 |
| V157 | Table 5 4b_fresh_untouched closed_by_2048 0.013 | json | X | length_profile/4b_fresh_untouched/closed_by_2048/seed_mean | 1 | 0.013 | 0.0005 |
| V158 | Table 5 4b_fresh_untouched closed_by_8192 0.13 | json | X | length_profile/4b_fresh_untouched/closed_by_8192/seed_mean | 1 | 0.13 | 0.0005 |
| V159 | Table 5 4b_fresh_untouched correct_mean 6921 | json | X | length_profile/4b_fresh_untouched/correct_mean/seed_mean | 1 | 6921 | 0.5 |
| V160 | Table 5 4b_math500_untouched median 5267 | json | X | length_profile/4b_math500_untouched/median/seed_mean | 1 | 5267 | 0.5 |
| V161 | Table 5 4b_math500_untouched closed_by_2048 0.174 | json | X | length_profile/4b_math500_untouched/closed_by_2048/seed_mean | 1 | 0.174 | 0.0005 |
| V162 | Table 5 4b_math500_untouched closed_by_8192 0.637 | json | X | length_profile/4b_math500_untouched/closed_by_8192/seed_mean | 1 | 0.637 | 0.0005 |
| V163 | Table 5 4b_math500_untouched correct_mean 4676 | json | X | length_profile/4b_math500_untouched/correct_mean/seed_mean | 1 | 4676 | 0.5 |
| V164 | Table 5 4b_fresh_warmup closed_by_2048 0.015 | json | X | length_profile/4b_fresh_warmup/closed_by_2048/seed_mean | 1 | 0.015 | 0.0005 |
| V165 | Table 5 4b_fresh_warmup closed_by_8192 0.145 | json | X | length_profile/4b_fresh_warmup/closed_by_8192/seed_mean | 1 | 0.145 | 0.0005 |
| V166 | Table 5 4b_fresh_warmup correct_mean 6825 | json | X | length_profile/4b_fresh_warmup/correct_mean/seed_mean | 1 | 6825 | 0.5 |
| V167 | Table 5 4b_math500_warmup median 4832 | json | X | length_profile/4b_math500_warmup/median/seed_mean | 1 | 4832 | 0.5 |
| V168 | Table 5 4b_math500_warmup closed_by_2048 0.209 | json | X | length_profile/4b_math500_warmup/closed_by_2048/seed_mean | 1 | 0.209 | 0.0005 |
| V169 | Table 5 4b_math500_warmup closed_by_8192 0.67 | json | X | length_profile/4b_math500_warmup/closed_by_8192/seed_mean | 1 | 0.67 | 0.0005 |
| V170 | Table 5 4b_math500_warmup correct_mean 4490 | json | X | length_profile/4b_math500_warmup/correct_mean/seed_mean | 1 | 4490 | 0.5 |
| V171 | Table 5 4b_fresh_ctrl closed_by_2048 0.03 | json | X | length_profile/4b_fresh_ctrl/closed_by_2048/seed_mean | 1 | 0.03 | 0.0005 |
| V172 | Table 5 4b_fresh_ctrl closed_by_8192 0.214 | json | X | length_profile/4b_fresh_ctrl/closed_by_8192/seed_mean | 1 | 0.214 | 0.0005 |
| V173 | Table 5 4b_fresh_ctrl correct_mean 6574 | json | X | length_profile/4b_fresh_ctrl/correct_mean/seed_mean | 1 | 6574 | 0.5 |
| V174 | Table 5 4b_math500_ctrl median 3497 | json | X | length_profile/4b_math500_ctrl/median/seed_mean | 1 | 3497 | 0.5 |
| V175 | Table 5 4b_math500_ctrl closed_by_2048 0.338 | json | X | length_profile/4b_math500_ctrl/closed_by_2048/seed_mean | 1 | 0.338 | 0.0005 |
| V176 | Table 5 4b_math500_ctrl closed_by_8192 0.751 | json | X | length_profile/4b_math500_ctrl/closed_by_8192/seed_mean | 1 | 0.751 | 0.0005 |
| V177 | Table 5 4b_math500_ctrl correct_mean 3752 | json | X | length_profile/4b_math500_ctrl/correct_mean/seed_mean | 1 | 3752 | 0.5 |
| V178 | Table 5 4b_fresh_ctrlpm closed_by_2048 0.041 | json | X | length_profile/4b_fresh_ctrlpm/closed_by_2048/seed_mean | 1 | 0.041 | 0.0005 |
| V179 | Table 5 4b_fresh_ctrlpm closed_by_8192 0.241 | json | X | length_profile/4b_fresh_ctrlpm/closed_by_8192/seed_mean | 1 | 0.241 | 0.0005 |
| V180 | Table 5 4b_fresh_ctrlpm correct_mean 6429 | json | X | length_profile/4b_fresh_ctrlpm/correct_mean/seed_mean | 1 | 6429 | 0.5 |
| V181 | Table 5 4b_math500_ctrlpm median 3000 | json | X | length_profile/4b_math500_ctrlpm/median/seed_mean | 1 | 3000 | 0.5 |
| V182 | Table 5 4b_math500_ctrlpm closed_by_2048 0.386 | json | X | length_profile/4b_math500_ctrlpm/closed_by_2048/seed_mean | 1 | 0.386 | 0.0005 |
| V183 | Table 5 4b_math500_ctrlpm closed_by_8192 0.768 | json | X | length_profile/4b_math500_ctrlpm/closed_by_8192/seed_mean | 1 | 0.768 | 0.0005 |
| V184 | Table 5 4b_math500_ctrlpm correct_mean 3533 | json | X | length_profile/4b_math500_ctrlpm/correct_mean/seed_mean | 1 | 3533 | 0.5 |
| V185 | Table 5 4b_fresh_wide1 closed_by_2048 0.14 | json | X | length_profile/4b_fresh_wide1/closed_by_2048/seed_mean | 1 | 0.14 | 0.0005 |
| V186 | Table 5 4b_fresh_wide1 closed_by_8192 0.475 | json | X | length_profile/4b_fresh_wide1/closed_by_8192/seed_mean | 1 | 0.475 | 0.0005 |
| V187 | Table 5 4b_fresh_wide1 correct_mean 4848 | json | X | length_profile/4b_fresh_wide1/correct_mean/seed_mean | 1 | 4848 | 0.5 |
| V188 | Table 5 4b_math500_wide1 median 839 | json | X | length_profile/4b_math500_wide1/median/seed_mean | 1 | 839 | 0.5 |
| V189 | Table 5 4b_math500_wide1 closed_by_2048 0.685 | json | X | length_profile/4b_math500_wide1/closed_by_2048/seed_mean | 1 | 0.685 | 0.0005 |
| V190 | Table 5 4b_math500_wide1 closed_by_8192 0.902 | json | X | length_profile/4b_math500_wide1/closed_by_8192/seed_mean | 1 | 0.902 | 0.0005 |
| V191 | Table 5 4b_math500_wide1 correct_mean 1901 | json | X | length_profile/4b_math500_wide1/correct_mean/seed_mean | 1 | 1901 | 0.5 |
| V192 | Table 5 4b_fresh_wide closed_by_2048 0.157 | json | X | length_profile/4b_fresh_wide/closed_by_2048/seed_mean | 1 | 0.157 | 0.0005 |
| V193 | Table 5 4b_fresh_wide closed_by_8192 0.485 | json | X | length_profile/4b_fresh_wide/closed_by_8192/seed_mean | 1 | 0.485 | 0.0005 |
| V194 | Table 5 4b_fresh_wide correct_mean 4704 | json | X | length_profile/4b_fresh_wide/correct_mean/seed_mean | 1 | 4704 | 0.5 |
| V195 | Table 5 4b_math500_wide median 753 | json | X | length_profile/4b_math500_wide/median/seed_mean | 1 | 753 | 0.5 |
| V196 | Table 5 4b_math500_wide closed_by_2048 0.703 | json | X | length_profile/4b_math500_wide/closed_by_2048/seed_mean | 1 | 0.703 | 0.0005 |
| V197 | Table 5 4b_math500_wide closed_by_8192 0.897 | json | X | length_profile/4b_math500_wide/closed_by_8192/seed_mean | 1 | 0.897 | 0.0005 |
| V198 | Table 5 4b_math500_wide correct_mean 1849 | json | X | length_profile/4b_math500_wide/correct_mean/seed_mean | 1 | 1849 | 0.5 |
| V199 | split 4b_math500_by_ctrl open share | json | X | split/4b_math500_by_ctrl/share_open/seed_mean | 1 | 0.249 | 0.0005 |
| V200 | split 4b_math500_by_ctrl closed ctrl | json | X | split/4b_math500_by_ctrl/closed/ctrl/seed_mean | 1 | 0.993 | 0.0005 |
| V201 | split 4b_math500_by_ctrl closed ctrlpm | json | X | split/4b_math500_by_ctrl/closed/ctrlpm/seed_mean | 1 | 0.986 | 0.0005 |
| V202 | split 4b_math500_by_ctrl closed wide1 | json | X | split/4b_math500_by_ctrl/closed/wide1/seed_mean | 1 | 0.985 | 0.0005 |
| V203 | split 4b_math500_by_ctrl closed wide | json | X | split/4b_math500_by_ctrl/closed/wide/seed_mean | 1 | 0.976 | 0.0005 |
| V204 | split 4b_math500_by_ctrl open ctrl | json | X | split/4b_math500_by_ctrl/open/ctrl/seed_mean | 1 | 0.578 | 0.0005 |
| V205 | split 4b_math500_by_ctrl open ctrlpm | json | X | split/4b_math500_by_ctrl/open/ctrlpm/seed_mean | 1 | 0.62 | 0.0005 |
| V206 | split 4b_math500_by_ctrl open wide1 | json | X | split/4b_math500_by_ctrl/open/wide1/seed_mean | 1 | 0.79 | 0.0005 |
| V207 | split 4b_math500_by_ctrl open wide | json | X | split/4b_math500_by_ctrl/open/wide/seed_mean | 1 | 0.784 | 0.0005 |
| V208 | split 4b_math500_by_wide1 open share | json | X | split/4b_math500_by_wide1/share_open/seed_mean | 1 | 0.098 | 0.0005 |
| V209 | split 4b_math500_by_wide1 closed ctrl | json | X | split/4b_math500_by_wide1/closed/ctrl/seed_mean | 1 | 0.948 | 0.0005 |
| V210 | split 4b_math500_by_wide1 closed ctrlpm | json | X | split/4b_math500_by_wide1/closed/ctrlpm/seed_mean | 1 | 0.951 | 0.0005 |
| V211 | split 4b_math500_by_wide1 closed wide1 | json | X | split/4b_math500_by_wide1/closed/wide1/seed_mean | 1 | 0.978 | 0.0005 |
| V212 | split 4b_math500_by_wide1 closed wide | json | X | split/4b_math500_by_wide1/closed/wide/seed_mean | 1 | 0.961 | 0.0005 |
| V213 | split 4b_math500_by_wide1 open ctrl | json | X | split/4b_math500_by_wide1/open/ctrl/seed_mean | 1 | 0.347 | 0.0005 |
| V214 | split 4b_math500_by_wide1 open ctrlpm | json | X | split/4b_math500_by_wide1/open/ctrlpm/seed_mean | 1 | 0.378 | 0.0005 |
| V215 | split 4b_math500_by_wide1 open wide1 | json | X | split/4b_math500_by_wide1/open/wide1/seed_mean | 1 | 0.551 | 0.0005 |
| V216 | split 4b_math500_by_wide1 open wide | json | X | split/4b_math500_by_wide1/open/wide/seed_mean | 1 | 0.622 | 0.0005 |
| V217 | split 4b_fresh_by_ctrl open share | json | X | split/4b_fresh_by_ctrl/share_open/seed_mean | 1 | 0.786 | 0.0005 |
| V218 | split 4b_fresh_by_ctrl closed ctrl | json | X | split/4b_fresh_by_ctrl/closed/ctrl/seed_mean | 1 | 0.997 | 0.0005 |
| V219 | split 4b_fresh_by_ctrl closed ctrlpm | json | X | split/4b_fresh_by_ctrl/closed/ctrlpm/seed_mean | 1 | 0.979 | 0.0005 |
| V220 | split 4b_fresh_by_ctrl closed wide1 | json | X | split/4b_fresh_by_ctrl/closed/wide1/seed_mean | 1 | 0.978 | 0.0005 |
| V221 | split 4b_fresh_by_ctrl closed wide | json | X | split/4b_fresh_by_ctrl/closed/wide/seed_mean | 1 | 0.977 | 0.0005 |
| V222 | split 4b_fresh_by_ctrl open ctrl | json | X | split/4b_fresh_by_ctrl/open/ctrl/seed_mean | 1 | 0.333 | 0.0005 |
| V223 | split 4b_fresh_by_ctrl open ctrlpm | json | X | split/4b_fresh_by_ctrl/open/ctrlpm/seed_mean | 1 | 0.355 | 0.0005 |
| V224 | split 4b_fresh_by_ctrl open wide1 | json | X | split/4b_fresh_by_ctrl/open/wide1/seed_mean | 1 | 0.508 | 0.0005 |
| V225 | split 4b_fresh_by_ctrl open wide | json | X | split/4b_fresh_by_ctrl/open/wide/seed_mean | 1 | 0.523 | 0.0005 |
| V226 | split 4b_fresh_by_wide1 open share | json | X | split/4b_fresh_by_wide1/share_open/seed_mean | 1 | 0.525 | 0.0005 |
| V227 | split 4b_fresh_by_wide1 closed ctrl | json | X | split/4b_fresh_by_wide1/closed/ctrl/seed_mean | 1 | 0.828 | 0.0005 |
| V228 | split 4b_fresh_by_wide1 closed ctrlpm | json | X | split/4b_fresh_by_wide1/closed/ctrlpm/seed_mean | 1 | 0.842 | 0.0005 |
| V229 | split 4b_fresh_by_wide1 closed wide1 | json | X | split/4b_fresh_by_wide1/closed/wide1/seed_mean | 1 | 0.962 | 0.0005 |
| V230 | split 4b_fresh_by_wide1 closed wide | json | X | split/4b_fresh_by_wide1/closed/wide/seed_mean | 1 | 0.95 | 0.0005 |
| V231 | split 4b_fresh_by_wide1 open ctrl | json | X | split/4b_fresh_by_wide1/open/ctrl/seed_mean | 1 | 0.156 | 0.0005 |
| V232 | split 4b_fresh_by_wide1 open ctrlpm | json | X | split/4b_fresh_by_wide1/open/ctrlpm/seed_mean | 1 | 0.167 | 0.0005 |
| V233 | split 4b_fresh_by_wide1 open wide1 | json | X | split/4b_fresh_by_wide1/open/wide1/seed_mean | 1 | 0.287 | 0.0005 |
| V234 | split 4b_fresh_by_wide1 open wide | json | X | split/4b_fresh_by_wide1/open/wide/seed_mean | 1 | 0.322 | 0.0005 |
| V235 | G1 gap Gate(0,1) - LoRA arm 1,293 | json | X | g1_decomp/wide1/gap_full_minus_ctrl | 1 | -1293 | 0.5 |
| V236 | G1 72% in co-trained LoRA | json | X | g1_decomp/wide1/share_cotrained_lora | 1 | 0.72 | 0.005 |
| V237 | G1 28% with the gate | json | X | g1_decomp/wide1/share_gate_at_inference | 1 | 0.28 | 0.005 |
| V238 | 48% shorter MATH-500 (Gate(0,1) vs LoRA) | json | X | derived/cut_4b_math500_wide1_vs_ctrl | 1 | 0.48 | 0.005 |
| V239 | 82% shorter GSM8K (Gate(0,1) vs LoRA) | json | X | derived/cut_4b_gsm8k_wide1_vs_ctrl | 1 | 0.82 | 0.005 |
| V240 | 84% shorter GSM8K (Gate(0,2) vs LoRA, 4B) | json | X | derived/cut_4b_gsm8k_wide_vs_ctrl | 1 | 0.84 | 0.005 |
| V241 | 49% shorter GSM8K (Gate(0,2) vs LoRA, 2B read 1) | json | X | derived/cut_2b_gsm8k_wide_vs_ctrl_read1 | 1 | 0.49 | 0.005 |
| V242 | training hardware moves HumanEval ~10 points (0.177 - 0.079) | json | X | derived/humaneval_hw_gap_ctrl_s0 | 1 | 0.1 | 0.005 |
| V243 | Olmo MDE low 3 points | json | X | olmo_mde/noncwe_below_ceiling_paired_mde/min | 1 | 0.03 | 0.005 |
| V244 | Olmo MDE high 10 points | json | X | olmo_mde/noncwe_below_ceiling_paired_mde/max | 1 | 0.1 | 0.005 |
| W1 | 4B budget to match MATH LoRA 2,600 (2 s.f. of 2,590) | json | A4M | arms/ctrl/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 2600 | 50 |
| W2 | ... LoRA-r20 2,600 | json | A4M | arms/ctrlpm/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 2600 | 50 |
| W3 | ... Gate(0,1) 1,000 (1,024) | json | A4M | arms/wide1/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 1000 | 50 |
| W4 | ... Gate(0,2) 880 (882) | json | A4M | arms/wide/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 880 | 5 |
| W5 | GSM8K LoRA 1,400 (1,390) | json | A4G | arms/ctrl/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 1400 | 50 |
| W6 | GSM8K LoRA-r20 490 (494) | json | A4G | arms/ctrlpm/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 490 | 5 |
| W7 | GSM8K Gate(0,1) 390 | json | A4G | arms/wide1/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 390 | 5 |
| W8 | GSM8K Gate(0,2) 320 (323) | json | A4G | arms/wide/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 320 | 5 |
| W9 | fresh 4B warm-up 5,900 (5,881) | json | F4 | warmup/tokens_to_base_max_acc | 1 | 5900 | 50 |
| W10 | fresh 4B LoRA 3,000 (3,045) | json | F4 | arms/ctrl/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 3000 | 50 |
| W11 | fresh 4B LoRA-r20 2,800 (2,784) | json | F4 | arms/ctrlpm/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 2800 | 50 |
| W12 | fresh 4B Gate(0,1) 1,200 (1,205) | json | F4 | arms/wide1/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 1200 | 50 |
| W13 | fresh 4B Gate(0,2) 1,100 (1,081) | json | F4 | arms/wide/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 1100 | 50 |
| W14 | fresh 2B LoRA 710 (715) | json | F2 | arms/ctrl/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 710 | 5 |
| W15 | fresh 2B Gate(0,1) 520 (523) | json | F2 | arms/wide1/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 520 | 5 |
| W16 | fresh 2B Gate(0,2) 300 (303) | json | F2 | arms/wide/tokens_to_base_max_acc/from_seed_mean_curve | 1 | 300 | 5 |
| W17 | 150 GRPO steps | text | PL4 | 150 steps, last checkpoint read out | | | |
| W18 | Qwen3.5-4B revision 851bf6e8 | text | PL4 | revision 851bf6e8 | | | |
| W19 | Qwen3.5-2B revision 15852e8c | text | PL2 | revision 15852e8c | | | |
| W20 | hidden size 2,560 | text | PL4 | hidden 2,560 | | | |
| W21 | trainer PyTorch 2.14.0 | text | env/train.txt | torch==2.14.0+cu130 | | | |
| W22 | flash-linear-attention 0.5.2 | text | env/fla-pylib.txt | flash-linear-attention==0.5.2 | | | |
| W23 | CUDA 13.0 | text | results/ENVIRONMENT.json | "torch": "2.13.0+cu130" | | | |
| W24 | G3 steps 25 to 125 | text | results/mech/gate_stats_summary.txt | wide1_s0_125 | | | |
| W25 | MATH-500 450 items the pilot never read | text | PL2 | 450 unseen items as a check | | | |
| W26 | 14 4B fresh readouts | text | PL4 | 14 readouts | | | |
| W27 | 8 of 14 4B fresh readouts finished before the approval | text | PL4 | 8 finished 4B fresh readout files | | | |
| W28 | approval 06:11 UTC | text | PL4 | chat at 06:11:52 UTC | | | |
| W29 | withdrawn amendment, never used, retry passed | text | LEDG | approved 2026-10-11 ~22:45 but never used | | | |
| W30 | rented total about 210 GPU-h (box time) | json | CH | rented/gpu_hours_total | 1 | 210 | 0.5 |
| W31 | rented logged jobs 171 GPU-h | json | CH | hours/RTX PRO 6000/total | 1 | 171 | 0.5 |
| W32 | GB10 logged at least 125 GPU-h | json | CH | hours/GB10/total | 1 | 125 | 0.5 |
| W33 | GB10 pilots and warm-ups 12 h | json | CH | hours/GB10/train_pilot | 1 | 12 | 0.5 |
| W34 | RTX 5090 Laptop logged 7 h | json | CH | hours/RTX 5090 Laptop/total | 1 | 7 | 0.5 |
| W35 | Gate(0,2) moves beta 1.97 times Gate(0,1) (layer mean) | json | D | mech/g3/wide_over_wide1_by_layer/mean | 1 | 1.97 | 0.005 |
| W36 | GSM8K wait per 1k LoRA 1.09 | json | TS | gsm8k/ctrl/mean/wait_per_1k | 1 | 1.09 | 0.005 |
| W37 | GSM8K wait per 1k LoRA-r20 0.88 | json | TS | gsm8k/ctrlpm/mean/wait_per_1k | 1 | 0.88 | 0.005 |
| W38 | GSM8K wait per 1k Gate(0,1) 0.61 | json | TS | gsm8k/wide1/mean/wait_per_1k | 1 | 0.61 | 0.006 |
| W39 | GSM8K wait per 1k Gate(0,2) 0.68 | json | TS | gsm8k/wide/mean/wait_per_1k | 1 | 0.68 | 0.005 |
| W40 | MATH wait per 1k Gate(0,1) 1.87 | json | TS | math500/wide1/mean/wait_per_1k | 1 | 1.87 | 0.006 |
| W41 | MATH wait per 1k LoRA 1.55 | json | TS | math500/ctrl/mean/wait_per_1k | 1 | 1.55 | 0.005 |
| W42 | MATH check/verify per 1k Gate(0,1) 2.08 | json | TS | math500/wide1/mean/check_verify_per_1k | 1 | 2.08 | 0.005 |
| W43 | MATH check/verify per 1k LoRA 2.44 | json | TS | math500/ctrl/mean/check_verify_per_1k | 1 | 2.44 | 0.005 |
| W44 | Olmo clamp NLL Wikipedia +0.060 | json | ONLL | arms/erase/wiki/delta | 1 | 0.060 | 0.0005 |
| W45 | ... math +0.011 | json | ONLL | arms/erase/math/delta | 1 | 0.011 | 0.0005 |
| W46 | ... web +0.005 | json | ONLL | arms/erase/web/delta | 1 | 0.005 | 0.0005 |
| W47 | ... code +0.004 | json | ONLL | arms/erase/code/delta | 1 | 0.004 | 0.0005 |
| W48 | ... code interval low -0.002 | json | ONLL | arms/erase/code/ci/0 | 1 | -0.002 | 0.0005 |
| W49 | ... code interval high +0.010 | json | ONLL | arms/erase/code/ci/1 | 1 | 0.010 | 0.0005 |
| W50 | forgetting LoRA GB10 seed 4 0.268 | json | D | forgetting_humaneval/4B_main_ctrl_extra/s4/pass1 | 1 | 0.268 | 0.0005 |
| W51 | forgetting LoRA GB10 seed 5 0.165 | json | D | forgetting_humaneval/4B_main_ctrl_extra/s5/pass1 | 1 | 0.165 | 0.0005 |
| W52 | same-hardware trio Gate(0,1) 0.451 | json | D | forgetting_humaneval/4B_same_hw_s0/wide1/pass1 | 1 | 0.451 | 0.0005 |
| W53 | SFT 1,200 steps (held-out loss at step 1200) | text | results/run_logs/broad/step6/ctrl_s0/val_log.jsonl | "step": 1200 |  |  |  |
| W54 | SFT step tokens 33,472 | text | src/gdn_gate/broad/train_sft.py | default=33472 | | | |
| W55 | SFT base model held-out loss at step 0 1.322 | json | D | step6_sft/rows/ctrl_s0/val_loss | 1 | 1.197 | 0.0005 |
| W56 | ... base model 1.322 (val_log step 0) | text | results/run_logs/broad/step6/ctrl_s0/val_log.jsonl | {"val_loss": 1.3218949556568178, "val_tokens": 26312, "step": 0} |  |  |  |
| W57 | 2B record primary: area on both sets, contrasts wide - wide1 and wide - ctrl | text | PL2 | difference (wide minus wide1, wide minus ctrl) excludes 0 | | | |
| W58 | H2 came from a 2B length difference the 2B analysis did not test | text | PL4 | applies the rule to area only | | | |
| W59 | 2B read 1 GSM8K area: no effect for Gate(0,1) - LoRA | json | A2G | effects/wide1 - ctrl/effect | 1 | 0 | 0 |
| W60 | 2B read 1 GSM8K area: no effect for Gate(0,2) - Gate(0,1) | json | A2G | effects/wide - wide1/effect | 1 | 0 | 0 |
| W61 | 1.9% more parameters (+1.91%) | text | PL4 | 10,844,160, +1.91% over wide1 | | | |
| W62 | answer cap 99th percentile rule, multiple of 256 | text | PL2 | answer_max = the smallest multiple | | | |
