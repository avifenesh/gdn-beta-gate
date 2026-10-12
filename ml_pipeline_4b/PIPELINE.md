# ML pipeline: step 7 at Qwen3.5-4B (C4 replication), length-cost GRPO

Owner approval of the plan 2026-10-11 13:00 (in chat: "go ahead" to all eight items). Owner direction 2026-10-11: "do" to "a second, larger model (4B or 9B) and more seeds, about one
box-day". This record is a new experiment with its own gates, its own guard state (test touches 0) and its own
receipts (results/ml7_4b/). It repeats the 2B record (ml_pipeline/PIPELINE.md) at a larger model with the
same data, splits, metric, decision rule, recipe and rules; every difference is listed below. Code under study: the
2B record's instrument, unchanged (readout_vllm.py, train_grpo_vllm.py, gate_vllm.py, rl_common.py, arms.py,
packing.py: sha256 equal to the 2B touch-1 code lock, results/ml7/step14/code.sha256).

## Checklist

| # | Step | Status |
|---|---|---|
| 1 | Data inspection | done 2026-10-11 (the 2B record's step 1, same frozen files, artifacts copied; 4B prompt check template_identity.json PASS) |
| 2 | Exploratory data analysis | done 2026-10-11 (the 2B record's step 2, figures copied) |
| 3 | Define the prediction problem | done 2026-10-11 (the 2B contract with the changes listed below, H1 and H2 pre-registered) |
| A | Gate A | approved-gate 2026-10-11 13:00 |
| 4 | Data cleaning | done 2026-10-11 (reused: same rules, same files) |
| 5 | Data engineering | done 2026-10-11 (reused: split_manifest.json a byte copy, sha256 ef2d531b...) |
| 6 | Train / validation / test split | done 2026-10-11 (reused; fresh guard state ml_pipeline_4b/.guard_state.json, fingerprint b28bc3d5..., test touches 0) |
| 7 | Feature engineering | done 2026-10-11 (not applicable, as the 2B record) |
| 8 | Preprocessing | done 2026-10-11 (not applicable, as the 2B record) |
| B | Gate B | approved-gate 2026-10-11 13:00 |
| 9 | Baseline model | todo |
| 10 | Model training | todo |
| 11 | Hyperparameter tuning | todo (the cap pilot rule) |
| 12 | Model evaluation | todo |
| C | Gate C | approved-gate 2026-10-11 13:00 (line recorded after the 4B 12_ figures) |
| 13 | Error analysis | todo |
| 14 | Final test (test set touched once) | todo |
| 15 | Deployment (only if the owner asks) | todo |
| 16 | Monitoring and retraining plan | todo |
| D | Gate D | todo (owner) |

- Gate C: approved 2026-10-11 13:00 (owner, in chat: "go ahead" to the proposal's item 4): step 14 launches automatically once all 9 4B arms finish and the step 14 analysis is frozen; the test set is read once by this record. Line written by pull_rig_p3.sh after the 4B step 12 figures existed and the ml-pipeline hook accepted it.
- Gate B: approved 2026-10-11 13:00 (owner, in chat: "go ahead" to the proposal's item 3, with items 6 and 8): the recipe identical to the 2B record (LoRA r16 on the GDN projections, gate W for wide and wide1, AdamW lr 2e-5, 16 prompts x 8 samples, Dr. GRPO, no KL, alpha 0.2 on reasoning tokens, 150 steps, last checkpoint read out, shared format warm-up of 40 steps); the answer cap and the cap pilots rerun at 4B by the same rules (answer cap rule of 2026-10-08 22:32, pilot rule of 2026-10-08 21:06 at caps 512, 1,024, 2,048); an arm with the gate (wide, wide1) failing the startup parity check on the decode pair alone may rerun once at k 2.5; step 9 cap mode kept, record-only.
- Gate A: approved 2026-10-11 13:00 (owner, in chat: "go ahead" to the proposal's items 1 and 2): model Qwen/Qwen3.5-4B revision 851bf6e8 for the C4 replication, the 9B only as a follow-up if the 4B result holds; the step 3 contract below with H1 (MATH-500 wide - ctrl area > 0) and H2 (GSM8K natural length wide < ctrl, 3/3 seeds plus the paired item bootstrap interval) pre-registered; steps 1 to 8 reused from the 2B record's frozen artifacts (byte copies, COPIED_FROM_2B.sha256).

## Model

- Qwen/Qwen3.5-4B, revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (post-trained, hybrid thinking, thinking on by
  default; Apache-2.0). Configs and tokenizer are on the workstation (hf-models snapshot); the weights are not, so the box
  downloads them by revision and checks every shard against the hub's sha256 before use.
- Same tokenizer as the 2B (tokenizer.json, vocab.json, merges.txt sha256 equal). The chat templates differ in one
  place: the default of enable_thinking (2B off, 4B and 9B on). rl_common.Thinker passes enable_thinking=True, so the
  prompts are identical: all 9,284 prompts of the four frozen files render to the same token ids with the 2B, 4B and
  9B tokenizers, and the ids Thinker uses (</think>, <think>, <|im_end|>, <|endoftext|>, eos, the forced close) are
  equal. Red arm: rendered without enable_thinking, the 2B and 4B prompts differ. Receipt
  results/ml7_4b/template_identity.json (src/gdn_gate/ml7/template_identity_cpu.py).
- Same vendor thinking sampling on both cards for text tasks: temperature 1.0, top_p 0.95, top_k 20, presence 1.5.
- Architecture: 32 layers (24 Gated DeltaNet, 8 full attention), hidden 2,560, GDN 16 key x 32 value heads; the 2B
  has 24 layers (18 GDN), hidden 2,048, 16 x 16. The 4B has the GDN geometry of the 9B. Per-token matmul parameters
  4.20B vs 1.88B (2.2x); LoRA r16 on the GDN projections 8.7M parameters (2B 4.7M); gate W 1.97M (2B 0.59M).
- Why 4B and not 9B (chosen on task, training reliability, licence, size and cost; which engine serves it is not a
  criterion): both keep the template, tokenizer and family fixed, so size is the only change. The 4B fits the full
  protocol (step 9, warm-up, three cap pilots, 9 arms, steps 12 and 14) inside about one box-day; the 9B costs about
  1.6 to 1.9x the 4B's GPU-hours (per-token compute 4.2x the 2B vs 2.2x), about 1.7 box-days expected. The 9B stays
  the next step if the 4B replicates.

## Step 3: contract, changes from the 2B record only

- Population, unit, target, budgets, readout instrument, metric (normalized area under accuracy vs log2 budget,
  256 to 8,192), decision rule (3 of 3 seeds same sign AND paired item bootstrap 95% interval excluding 0), seeds
  0, 1, 2, contrasts (wide - ctrl, wide1 - ctrl, wide - wide1, each arm - warm-up, each arm - untouched): unchanged.
- Model: Qwen3.5-4B (above). The untouched-model baseline is the untouched 4B.
- Pre-registered confirmatory hypotheses, from the 2B record's test (touch 1, 2026-10-11), named before any 4B data:
  - H1: on MATH-500, wide - ctrl area > 0 under the decision rule (2B: +0.0270, interval +0.019 to +0.035, 3/3).
  - H2: on GSM8K test, wide reasons less than ctrl at the 8,192 budget (natural length) on 3 of 3 seeds, with the
    paired item bootstrap 95% interval of the seed-mean difference in reasoning tokens excluding 0 (2B: 1,025 vs 2,022
    tokens). The 2B analysis reports natural length per seed but applies the rule to area only; H2 needs a length
    block in step14_analysis.py (same bootstrap, same index matrix), added and CPU-tested before the analysis is frozen.
  - Everything else is reported with the same rule and no new claim.
- Constraints: one rented 8x RTX PRO 6000 box (owner approval of the budget below), development and evaluation only;
  every 4B number from one device type (the validate step refuses mixed GPUs).

## Train / validation / test

- The 2B record's frozen files, unchanged (split_manifest.json here is a byte copy): RL pool 6,965, validation 500,
  test GSM8K 1,319 + MATH-500 500 (fingerprint b28bc3d5...).
- The 2B record read this test set once (touch 1) and H1 and H2 come from it. No 4B model, cap, answer cap, recipe
  value or checkpoint is chosen with it: caps and the answer cap come from validation, the last checkpoint is read
  out, the analysis is frozen before the read. This record's guard state starts at test touches 0; the test set is
  read once, after all 9 arms.

## Gate B preparation

Model rationale:
- traits: RL fine-tuning of a pretrained LLM (Qwen3.5-4B, thinking mode), the 2B record's data unchanged: 6,965 RL prompts (GSM8K train, integer answers), 500 hashed validation items, 1,819 fixed test items (GSM8K test 1,319, MATH-500 500); target per item = correct (0/1) and reasoning tokens at each budget; no datetime; template siblings removed; the profile's imbalanced flag is set membership only
- baseline: the untouched Qwen3.5-4B curve (hard truncation at each cap, same readout and sampling), the 4B shared format warm-up, and the ctrl arm (LoRA r16 only) under the identical GRPO recipe
- candidates: wide (gate beta in (0,2) plus LoRA r16) and wide1 (same added parameters, beta in (0,1)), 3 seeds each, the 2B record's recipe, training cap and answer cap set again at 4B by the same rules on validation
- ruled out: Qwen3.5-9B in this record (about 1.7 box-days; the follow-up if the 4B result holds); a new optimizer (would confound size with optimizer); choosing a cap, answer cap, checkpoint or seed on the test set (the 2B record already read it once, H1 and H2 come from that read); grading through the string fallback
- metric: normalized area under accuracy vs log2 budget, caps 256 to 8,192, per test set, mean of 3 seeds (primary; H1 on MATH-500); natural length at 8,192 under the same rule (H2 on GSM8K); accuracy at each cap, tokens to the untouched model's 8k accuracy, cut-off share (secondary); an effect needs 3 of 3 seeds agreeing in sign and a paired item bootstrap interval excluding 0

- Recipe, identical to the 2B record: LoRA r16 (alpha 32) on the GDN projections, plus the gate W for wide
  (beta in (0, 2)) and wide1 (beta in (0, 1)); AdamW (0.9, 0.95), weight decay 0, lr 2e-5 for both groups, constant
  after 5% warmup, grad clip 1.0; 16 prompts x 8 samples, rollouts at temperature 1.0; Dr. GRPO advantage, no KL;
  reward rl_common.length_reward, alpha 0.2 on reasoning tokens only; 150 steps; the last checkpoint is read out.
  The optimizer stays AdamW on purpose: this is a replication of an arm contrast, and a new optimizer would confound
  size with optimizer.
- Answer cap (rule of 2026-10-08 22:32, rerun at 4B): the untouched 4B validation curve at answer cap 4,096; answer
  cap = smallest multiple of 256 at or above the 99th percentile of answers that end; the curve at that cap is the
  4,096 run truncated and regraded. The same answer cap goes into the rollouts, every readout and step 14.
- Shared format warm-up (design change of 2026-10-09 10:00): ctrl, correctness only (alpha 0), cap 512, the 4B
  answer cap, 40 steps from the untouched 4B; its validation readout is the baseline the cap pilots and arms are
  compared against.
- Training cap (rule of 2026-10-08 21:06): ctrl seed 0 from the warm-up, the first 40 steps of the 150-step schedule
  at caps 512, 1,024 and 2,048; each step-40 checkpoint read on validation; the cap with the largest paired area gain
  over the warm-up wins, caps whose interval holds the winner's point estimate tie, ties go to the cheaper cap
  (pilot_pick.py). The chosen pilot continues as ctrl seed 0.
- Step 9 cap mode (base accuracy at caps 256 to 4,096, train sampling, 4 samples) is run for the record; it selects
  nothing.
- Not recipe (memory and speed only): vLLM gpu_memory_utilization 0.30, training micro-row 8,192 tokens (the 2B used
  16,384; the trainer's startup memory probe decides), one run per GPU.
- Engine/trainer parity check at every start, k = 2 (unchanged). Owner decision 2026-10-11 13:00 (item 6): a gated
  arm that fails the startup parity check on the decode pair alone may rerun once at k = 2.5 (grpo_box.sh
  PARITY_RETRY_K=2.5, recorded in the run's parity_retry.json); any other parity failure stays a failure.
- Parity probe 0.05 for this record (owner approval 2026-10-11 ~14:55 Israel, in chat: "ok" to "probe 0.05 for the 4B
  record"): the startup engine/trainer parity self-test uses a probe of 0.05 nats per token instead of the 2B
  record's 0.2, in every 4B training run (warm-up, cap pilots, arms: train_grpo_vllm.py --probe_effect 0.05, an
  existing argument, the trainer unchanged) and every 4B readout (step 9, pilots, step 12, step 14:
  readout_vllm.py --probe_effect 0.05, an argument added by a documented amendment whose default 0.2 keeps every 2B
  readout unchanged; diff and ledger in src/gdn_gate/ml7/amendments/). Reason: at 4B the probe calibrator
  does not reach a 0.2 probe (it stops near 0.022 nats, under the visibility bound), and at probe 0.2 the gated arms'
  decode pair sits at 2.2 to 3.6x the floor; at 0.05, ctrl, wide and wide1 pass every check at k 2 (decode pair 1.10
  to 1.18x the floor, LoRA effect slope 0.98, corr 0.98). k stays 2.0, with the single 2.5 retry above. The self-test
  runs before training or generation and does not touch the scored generations.
- Step 9 cap mode stays record-only at 4B (owner decision 2026-10-11 13:00, item 8).

## Budget and hardware (owner decision 2026-10-11 13:00, item 7)

- One 8x RTX PRO 6000 on-demand box (600 W, never Max-Q), the cheapest of the GPU rental providers compared at
  rent time (spot and on-demand priced), shared with the 2B seeds 3-5 extension. Budget cap <price> for the box, with
  two stop-and-ask checks: after the calibration (about <price>) and after the cap pick (about <price>).
- GRPO speed on the PRO 6000 is measured by the calibration (a 10-step ctrl run and a 3-step wide run of the 4B, a
  2B extension rate check) before anything is committed.
- Expected 4B GPU-hours (plan estimate before calibration): about 64 at a picked cap of 512 (77 at 1,024, 97 at 2,048).

## Calibration on the rented box (2026-10-11, results/FINDINGS.md "Rented box (Phase 3)")

- Box: cloud 8x RTX PRO 6000 Server Edition (600 W), on-demand <price>, accepted 10:31 UTC; staged and verified.
- Speed: 4B GRPO about 41 s/step at cap 512 (wide 45), 63.5 at 1,024, about 108 at 2,048; memory fits at micro-row
  8,192 (peak 84.4 of 95 GiB at cap 2,048). 2B extension 24.7 s/step.
- Parity at the 2B record's settings (k 2, probe 0.2 nats per token) refuses every 4B run: ctrl on lora_visible (the
  probe calibrator stops at 0.022 nats at 4B), wide on the decode pair alone at 3.26x the floor (above the approved
  2.5), wide1 at 2.37x. Seeds 1-2: ctrl passes (the calibrator's failure depends on the random direction), wide1
  passes at the approved 2.5 retry (2.01 / 2.40x), wide fails at 2.5 (2.91 / 3.63x). The readout of a 4B checkpoint
  runs the same check with probe 0.2 and k 2.0 fixed in readout_vllm.py: ctrl passes, wide fails at 2.16x. With
  --probe_effect 0.05 every 4B arm passes every training check at k 2. Step 9 onward waits for the owner's call on
  the parity settings for training and readouts; nothing of this record has run.
- Note for the answer cap and cap pilot rules: the untouched 4B scores 0.957 at 8,192 on 128 validation items (area
  0.939), near the ceiling, so the pilot rule's validation area gains will be small.

## Runbook

- Box scheduler src/gdn_gate/box/orch.py with BOXQ_PHASE=3 (jobs c4_*), runners grpo_box.sh,
  step12_val_box.sh and step14_test_box.sh with the record overrides (STEP12_*/STEP14_* MODEL, ML=ml_pipeline_4b,
  WARM_RUN=ctrl_s0_cap512_fmt_box, ARM_TAG=cap<CAP>_r_box, ARM_CAP, ANSWER_MAX).
- Step 14 waits for the file ROOT/gates/c4_test.OK, written only with the owner's gate C line in this file.

## Steps 1-8: reused from the 2B record (frozen artifacts, 2026-10-11)

Steps 1 to 8 read the same frozen files as the 2B record (RL pool, validation slice, both test files: sha256 in
split_manifest.json, a byte copy of ml_pipeline/split_manifest.json) and the same tokenizer (template_identity.json:
all 9,284 prompts render to the same token ids with the 2B and 4B tokenizers, thinking on). Nothing in steps 1 to 8
depends on the model weights, so the 2B record's step 1-8 artifacts are this record's, copied byte for byte (not
recomputed) on 2026-10-11: data_profile.json, grader_check.json, step1_3_summary.json, step4_8_summary.json,
overlap.json and the 01_, 02_, 04_ and 06_ figures. sha256 of every copy, checked equal to the 2B file:
ml_pipeline_4b/COPIED_FROM_2B.sha256. The 2B record's step 12 figures (12_*) are 2B validation results and are not
copied: this record draws its own from the 4B step 12 analysis. The figure explanations below are the 2B record's
lines, unchanged; the pilot figures (02_pilot_*) describe the 2B (and the 4B-Base pilot7 rollouts) and are step 2
problem context, not 4B results.

## Figures

- Figure 01_missingness.png: 1 of 11 columns have missing values; worst is answer_log10_abs at 1.2%.
- Figure 02_target_balance.png: set=gsm8k_train is 80.4% of 9,292 rows, so a majority-class dummy scores 80.4%; the minority share is 5.4%.
- Figure 02_distributions.png: Histograms of 5 numeric features; 5 are strongly skewed (q_chars, q_tokens, prompt_tokens, answer_tokens, answer_log10_abs), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with set: answer_tokens (0.23); anything above 0.95 is a leakage suspect.
- Figure 02_question_length.png: Question length in Qwen3.5 tokens: GSM8K train median 56 (95th pct 100), GSM8K test median 57, MATH-500 median 50 (95th pct 215, max 831). GSM8K train and test have the same shape, so the RL prompts match the in-domain test set; MATH-500 is shorter at the median but has a long tail. The chat template and boxed-answer instruction add 26 tokens to every prompt. Prompts are tiny next to the 2k reasoning cap, so the cost of a rollout is the reasoning, not the question.
- Figure 02_answer_magnitude.png: GSM8K answers are numbers (100.0% of train parse as a number; median size about 10^1.6, largest about 10^8.3); GSM8K test has the same profile. MATH-500 answers are numeric for 78% of items and the rest are expressions, tuples, intervals or text (17% expressions). So the RL reward on GSM8K only ever checks a number, while the MATH-500 readout also needs symbolic answers the grader (math_verify) must match: MATH-500 measures transfer, GSM8K test measures the trained skill.
- Figure 02_train_test_overlap.png: For every held-out question, how close the nearest GSM8K train question is (1.0 = identical text). GSM8K test: 0 exact copies, 0 near-duplicates (same problem reworded, same answer), 3 template siblings (same wording with different numbers and answer, ratio 0.85 or more once numbers are masked), median closest ratio 0.4527. MATH-500: 0 exact, 0 near-duplicates, 0 template siblings, median 0.3333. The flagged train items (listed in overlap.json) are what step 4 proposes to drop from the RL prompt pool, so a GSM8K test gain cannot come from a memorized training problem.
- Figure 02_pilot_reasoning_length.png: Pilot7 readout, untouched Qwen3.5-2B thinking on 50 MATH-500 items (vendor sampling, one sample each): only 14 of 50 closed their thinking inside 8704 tokens (shortest 1262, median 4339); 36 were still thinking. Items that finished were right 100% of the time at the 8k cap, items cut off 0%. None finished under 1,000 tokens, which is why a 2k training cap on MATH-level problems gives almost no correct rollouts and no reward signal.
- Figure 02_pilot_accuracy_vs_budget.png: The accuracy-vs-tokens curve, the shape step 7 must move: 0.16 at 512, 0.16 at 1,024, 0.14 at 2,048, 0.20 at 4,096, 0.28 at 8,192 reasoning tokens, with 72% of items still cut off at 8k. With 50 items every interval is about plus or minus 0.11, so 512 to 4k are statistically flat and only 8k is plausibly higher; 7 items are right at both 512 and 8k. This baseline curve is from MATH-500, a held-out set, so it is used here only to describe the problem, not to pick the training cap.
- Figure 02_pilot_grpo_rollouts.png: Pilot7 GRPO, wide arm, seed 0, 2k cap on MATH train: 2B rollouts were cut off 99% of the time and correct 2.1%; 4B cut off 93%, correct 14.1%. Most prompt groups were all wrong (2B: 13 to 16 of 16), and an all-wrong group has zero advantage, so the policy gradient is near zero. The length penalty only acts on correct rollouts, so it needs a prompt set where the untouched model is right 30 to 60% of the time at the training cap: GSM8K train, with the cap read off a validation slice.
- Figure 04_grader_selfcheck.png: The grader run on answers built from the gold answer of every item (84,077 checks). Gold written verbatim passes 100.0% with math_verify and 100.0% with the string fallback; the red arms (wrong value, no box, empty box, a wrong last box) are accepted 0.00% and 0.00% of the time. The graders differ on harmless rewrites the model often makes: the fallback accepts 0% of '18.00', 0% of '$18', 0% of '18 dollars' and 0% of '1,000', where math_verify accepts at least 100% of every rewrite. A grader that marks right answers wrong lowers accuracy and feeds false zero rewards into GRPO, so math_verify has to be installed wherever step 7 runs.
- Figure 06_split_composition.png: The frozen splits: 6,965 RL prompts and 500 validation items out of 7,473 GSM8K train items; 4 train items dropped as siblings of a held-out question (4 pairs) and 4 more as siblings of a validation item (4 pairs). Question length matches across pool (median 56), validation (median 54) and GSM8K test, so a cap chosen on validation carries over to the in-domain test.
- Figure 12_val_accuracy_vs_budget.png: Validation slice (500 held-out GSM8K train items x 2 samples, one GPU for every model): accuracy against reasoning budget 256 to 8,192 tokens for the untouched model, the shared format warm-up and each arm (thick line = mean of its seeds, thin = single seeds). Area under the curve: untouched 0.936; warm-up 0.942; ctrl 0.952 (seeds 0.952/0.951/0.954); wide 0.960 (seeds 0.958/0.961/0.960); wide1 0.957 (seeds 0.956/0.961/0.952). The arms start from the warm-up, so the warm-up curve is the bar: an arm above it at small budgets got more accuracy for fewer reasoning tokens. All 9 arm runs present (ctrl seeds 0/1/2, wide seeds 0/1/2, wide1 seeds 0/1/2).
- Figure 12_val_area_contrasts.png: Validation slice (500 held-out GSM8K train items x 2 samples, one GPU for every model): every pre-registered contrast on the primary metric, as the seed-mean difference in area with its paired item bootstrap 95% interval and the single-seed differences. Rule: an arm effect needs all 3 seeds with the same sign and an interval that excludes 0; warm-up minus untouched has one readout per side, so it is descriptive. wide - ctrl +0.0071 [-0.0034, +0.0184] no effect; wide1 - ctrl +0.0041 [-0.0068, +0.0155] no effect; wide - wide1 +0.0030 [-0.0024, +0.0087] no effect; ctrl - untouched +0.0162 [+0.0081, +0.0250] effect, positive; wide - untouched +0.0233 [+0.0104, +0.0371] effect, positive; wide1 - untouched +0.0203 [+0.0077, +0.0338] effect, positive; ctrl - warmup +0.0109 [+0.0030, +0.0190] effect, positive; wide - warmup +0.0180 [+0.0045, +0.0324] effect, positive; wide1 - warmup +0.0150 [+0.0012, +0.0297] effect, positive; warmup - untouched +0.0053 [-0.0040, +0.0147] descriptive. All 9 arm runs present (ctrl seeds 0/1/2, wide seeds 0/1/2, wide1 seeds 0/1/2).
- Figure 12_val_accuracy_vs_tokens.png: Validation slice (500 held-out GSM8K train items x 2 samples, one GPU for every model): accuracy against the mean reasoning tokens each model actually spent at each budget, so a curve further left at the same height reasons less for the same accuracy. Natural length (mean reasoning tokens at 8,192) and budget to reach the untouched model's 8k accuracy (0.950): untouched 3110 tokens, cut off 0.089; warm-up 2694, cut off 0.050, reaches the untouched 8k accuracy at 2353; ctrl 1546, cut off 0.016, reaches it at 927; wide 265, cut off 0.001, reaches it at 263; wide1 302, cut off 0.000, reaches it at 329. All 9 arm runs present (ctrl seeds 0/1/2, wide seeds 0/1/2, wide1 seeds 0/1/2).
- Step 14 final test: test_rows_read_by_all_models=1.0000 (touch 1, 2026-10-11)

## Extension: parameter-matched control ctrlpm (owner approval 2026-10-11 ~23:55 Israel)

- Override: final test - user approved a second read of this record's test set (touch 2) for the 6 new readouts of the parameter-matched control ctrlpm only (ctrlpm x seeds 0, 1, 2 x GSM8K test and MATH-500) 2026-10-11 23:55 - reason: owner, in chat: "ok" to the proposal "one parameter-matched control arm at 4B: extra LoRA with the same added parameter count as the gate, without the gate; if the gate still wins the claim is recurrent gating, if it ties the claim shrinks to extra parameters; about half a box-day, roughly <price>"; H3 and H4 below were named before any ctrlpm run; the untouched, warm-up, ctrl, wide and wide1 test readouts of touch 1 are adopted by sha256 (STEP14_REUSE), so the 6 ctrlpm readouts are the only new reads of the test set.

Written 2026-10-12 00:30 Israel, before any ctrlpm run, any ctrlpm readout and any new read of the test set.

- Question: at 4B the gated arms beat ctrl on MATH-500 area (wide1 - ctrl +0.065) and reason much shorter on GSM8K
  (wide1 - ctrl -1,370 tokens). wide1 adds a gate W (1,966,080 trainable parameters) to ctrl's LoRA. Is the gain the
  recurrent gate, or just more trainable parameters?
- Arm ctrlpm: no gate (beta = sigmoid(b), the library forward, as ctrl) and LoRA rank 20 instead of 16 on the same four
  GDN projections (in_proj_qkv, in_proj_a, in_proj_b, out_proj), lora_alpha = 2 r (scale 2, as every arm). Trainable
  parameters, counted on the real Qwen3.5-4B config with the trainer's own arms.install and arms.add_lora (CPU test
  P1): ctrl 8,675,328; wide1 8,675,328 + 1,966,080 = 10,641,408; ctrlpm 10,844,160, +1.91% over wide1. One LoRA rank
  costs 542,208 here; rank 19 gives -3.19%, rank 21 +7.0%, so rank 20 is the closest uniform rank and the fallback (extra
  LoRA on other GDN projections) is not needed. ctrlpm carries slightly more parameters than wide1, the conservative
  direction for H3.
- Why extra rank on the same projections: the gate lives in the same GDN layers and is fed by the same hidden state;
  more rank there is the plainest "more parameters, no gate" alternative, and it keeps every other module of ctrl.
- Init: the 4B warm-up checkpoint (ctrl_s0_cap512_fmt_box step 40, sha256 ee5cc709...) widened to rank 20 by
  src/gdn_gate/ml7/pad_rank.py, one file per arm seed: the first 16 components bit-equal to the warm-up, the
  4 new A rows drawn from PEFT's own LoRA A init (U(-1/sqrt(in), 1/sqrt(in)), generator seeded by seed and tensor
  name), the 4 new B columns exactly zero. So at step 0 ctrlpm computes the warm-up's function exactly (CPU test P3:
  identical logits), as wide and wide1 do at W = 0, and the new components train like a fresh LoRA.
- Everything else as the 4B arms: GRPO recipe of Gate B, cap 512, answer cap 1,280, alpha 0.2 on reasoning tokens, 150
  steps, last checkpoint read out, AdamW lr 2e-5, 16 x 8 rollouts, micro-row 8,192, vLLM gpu_mem 0.30, one run per GPU,
  parity probe 0.05 at k 2 with no 2.5 retry (ctrl-type arm, as ctrl), seeds 0, 1, 2, every readout with the 4B
  readout arguments and probe 0.05, on RTX PRO 6000 Blackwell Server Edition (the analysis refuses mixed GPUs).
- Instrument: unchanged. readout_vllm.py, train_grpo_vllm.py, gate_vllm.py, rl_common.py, arms.py, packing.py and
  step14_analysis.py keep the sha256 of the 4B touch-1 code lock (CPU test P0); ctrlpm uses existing arguments
  (trainer --arm ctrl --rank 20 --init_from, readout --arm ctrl --rank 20), so no code-ledger amendment is needed. New
  files: pad_rank.py, ctrlpm_analysis.py (imports the frozen analysis, adds the arm, the contrasts wide1 - ctrlpm,
  wide - ctrlpm, ctrlpm - ctrl, and H3/H4; it must reproduce the touch-1 receipts bit for bit on the original arms or
  refuses), step12_ctrlpm_box.sh. Additive options with unchanged defaults: grpo_box.sh (ARM ctrlpm), step14_test_box.sh
  (STEP14_ARMS, STEP14_REUSE_LABELS, STEP14_PM_RANK), step14_guard.py (ctrlpm provenance: meta arm ctrl, rank 20, init =
  a pad_rank.py output whose receipt names the warm-up's sha256), orch.py (Phase 4 job list, ROOT/NGPU).
- Pre-registered confirmatory hypotheses for this extension, same decision rule as H1/H2 (3 of 3 seeds with the same
  sign AND the paired item bootstrap 95% interval of the seed-mean difference excluding 0):
  - H3: MATH-500 test, wide1 - ctrlpm area > 0.
  - H4: GSM8K test, wide1 - ctrlpm natural length (reasoning tokens at the 8,192 budget) < 0.
  - Reported with the same rule, no claim of their own: ctrlpm - ctrl (area and natural length, both sets), wide -
    ctrlpm, ctrlpm - warm-up, ctrlpm - untouched; H1 and H2 re-evaluate identically (their contrasts do not involve
    ctrlpm) and keep their touch-1 verdicts.
  - Reading named now: H3 and H4 supported = the 4B effect needs the gate, the same number of extra LoRA parameters
    does not give it (claim: recurrent gating). H3 failing with wide1 and ctrlpm tied = the claim shrinks to extra
    parameters. A split result (one of H3/H4) is reported as split.
- Validation (step 12, same box): the 3 ctrlpm runs read on the validation slice; the untouched, warm-up, ctrl, wide
  and wide1 validation readouts adopted from results/ml7_4b/step12_box after sha256 and code-lock checks;
  ctrlpm_analysis.py --split val. It selects nothing (recipe, checkpoint and analysis are fixed above).
- Test: touch 2 of this record for the 6 ctrlpm readouts only, after step 12; the 20 touch-1 readouts adopted by
  sha256 and code lock (STEP14_REUSE, STEP14_REUSE_LABELS). Why a second read is honest here: nothing is chosen with it
  (fresh runs, everything fixed above), and the touch-1 verdicts stand as recorded.
- Budget: <price> for the box (owner). One rented 4x RTX PRO 6000 Server Edition on-demand box; calibration first (s/step,
  memory, parity at k 2 with probe 0.05, a readout of a rank-20 checkpoint); if the projection exceeds <price> the long runs
  wait for the owner.
- Step 14 final test: test_rows_read_by_all_models=1.0000 (touch 2, 2026-10-12)
- 2026-10-12 H3/H4 (ctrlpm_analysis.py, test): H3 wide1 - ctrlpm MATH-500 area +0.060 (3/3, CI [0.049, 0.072]) supported;
  H4 wide1 - ctrlpm GSM8K natural length -1,000 tokens (3/3, CI [-1,043, -958]) supported. ctrlpm - ctrl: MATH-500 area
  no effect, GSM8K area +0.006 effect, lengths no effect. Verdict and publication are the owner's.

## Fresh test set (pre-registration, Phase 5, owner approval 2026-10-12)

Written 2026-10-12 before any model read of the fresh set: no readout of it exists, the box has not been rented, and
the fresh set's guard state shows test touches 0. Owner approval 2026-10-12 (relayed by the coordinator: "Lane wg,
Phase 5 rental (owner approved 2026-10-12, cap <price> total box spend)"; goal A: a fresh, harder test set at 4B and 2B,
pre-registered before any read).

- Question: do the 4B effects (gated arms score higher on MATH-500 area and reason far shorter) hold on harder, newer
  competition math that no step of this record has seen? The arms trained on GSM8K only.
- Data, frozen 2026-10-12 by src/gdn_gate/ml7/fresh/build_fresh.py from Hugging Face datasets at pinned
  revisions: AIME 2024 (HuggingFaceH4/aime_2024 2fe88a2f, 30), AIME 2025 (MathArena/aime_2025 c94da77e, 30), AIME 2026
  (MathArena/aime_2026 d2de22f3, 30; contests 2026-02-05 and 02-11), AMC 2023 (AI-MO/aimo-validation-amc 69d78a4a,
  the 40 rows of 2023 AMC 12A/12B, checked equal to math-ai/amc23 80815d37). Integer answers. Dedup with the 2B
  record's sibling scan (02_prep.py rule: n-gram cosine top 20, character ratio >= 0.85 with numbers kept or masked)
  against rl_pool.jsonl, val.jsonl and within the set: 0 hits, 0 dropped (test_gsm8k / test_math500 reported: 0
  hits). File <runs>/ml7/data/test_fresh.jsonl, 130 items, sha256 9ce51dff11355439...;
  manifest fresh_manifest.json beside it (sources, file sha256, dedup lists). Ids fresh:<source>-<contest>-<n>.
- Data period: Qwen3.5-4B / 2B weights were published 2026-02-27 / 02-28; the model card names no data cutoff and
  reports HMMT Nov 2025 as its newest benchmark. AIME 2026 postdates every benchmark the card reports (likely after
  the data period, not provable); AIME 2024, AIME 2025 and AMC 2023 predate the models and may be in their training
  data. Contamination would raise every arm alike (all share the base); per-source rows are reported, descriptively.
- Guard: a new held-out set with its own guard state, ml_pipeline_4b/fresh_test/.guard_state.json (fingerprint
  02966849..., frame id, problem, answer in file order, test touches 0) and its own manifest
  ml_pipeline_4b/fresh_test/split_manifest.json. One guard.final_test() call (src/gdn_gate/ml7/fresh/
  fresh_guard.py final) wraps every fresh readout of both models and both analyses; the set is read once (touch 1).
  Nothing is chosen with it: every checkpoint, recipe and analysis is fixed here.
- Models and checkpoints (sha256 must equal the ones the earlier test readouts recorded, fresh/ckpt_expect.json):
  Qwen3.5-4B 851bf6e8: untouched, warm-up (ctrl_s0_cap512_fmt_box step 40), ctrl, wide, wide1 and ctrlpm x seeds 0, 1,
  2 (step 150; 14 readouts). Qwen3.5-2B 15852e8c: untouched, ctrl, wide, wide1 x seeds 0-5 (step 150; seeds 0-2
  cap512_r_gb10, seeds 3-5 cap512_r_box; 19 readouts).
- Readout (same code: readout_vllm.py, train_grpo_vllm.py, gate_vllm.py, rl_common.py, arms.py, packing.py,
  step14_analysis.py, ctrlpm_analysis.py at the 4B touch-2 lock sha256; the fresh runner locks them again at its first
  readout; any change is an additive ledger amendment with a CPU test): caps 256, 512, 1,024, 2,048, 4,096, 8,192,
  samples 8 per item (small sets), seed 20261006, gpu_mem 0.45, vendor thinking sampling. 4B: answer cap 1,280, probe
  0.05, ctrlpm --arm ctrl --rank 20. 2B: answer cap 1,792, probe 0.2 (each record's own values). Every readout on one
  device type (the analysis refuses mixed GPUs).
- Metric and rule (unchanged): normalized area under accuracy vs log2 budget (norm_auc_log2) and natural length
  (reasoning tokens at 8,192), pooled over the 130 items; an effect needs every seed's difference to share a sign AND
  the paired item bootstrap 95% interval of the seed-mean difference to exclude 0 (10,000 draws, numpy
  default_rng(20261009)). Analysis: src/gdn_gate/ml7/fresh/fresh_analysis.py, which imports the frozen
  step14_analysis.py and ctrlpm_analysis.py unchanged and only sets their module constants (test file name, id
  prefix, two hypotheses); CPU-tested before the read.
- Pre-registered confirmatory hypotheses (4B, seeds 0-2):
  - H5: fresh set, wide1 - ctrl area > 0 under the decision rule.
  - H6: fresh set, wide1 - ctrl natural length < 0 under the decision rule.
  - Reported under the same rule, no claim of their own: wide1 - ctrlpm, wide - wide1 (area and length), wide - ctrl,
    wide - ctrlpm, ctrlpm - ctrl, each arm - warm-up and - untouched; per source (AIME 2024, 2025, 2026, AMC 2023)
    descriptively only. 2B (seeds 0-5): the H5 and H6 contrasts and the rest under the 6-seed rule, a secondary
    replication (H5-2B, H6-2B).
  - Reading named now: H5 and H6 supported = the 4B gated arms keep both the accuracy gain and the short reasoning on
    harder, newer problems. H6 alone = shorter but no accuracy gain on these items. Neither = the effect does not reach
    this difficulty. AIME at an 8k budget may put the 4B near its accuracy floor; that limits H5's power and is reported.
- Budget order (owner cap <price> for the whole box, coordinator's priority 2026-10-12): (1) the 4B fresh readouts, (2)
  mechanism G1, (3) G3, (4) the 2B fresh readouts, (5) G2. The 2B readouts are in the fresh plan only if the projection
  after the box calibration fits the cap; otherwise they are removed from the plan before the fresh read starts and
  the omission is recorded here. Mechanism jobs read the validation slice only, never a test set.
- Amendment 2026-10-12 05:58 Israel (before any fresh readout finished; no fresh result seen): readout gpu_mem 0.44 instead of 0.45 for every fresh readout. At 0.45 a second readout cannot start beside a running one on this box (vLLM: free memory 42.5 GiB < 0.45 x 94.97 GiB, because the starting readout already holds the 8.4 GB parity model); the first attempts of 8 readouts failed at engine start and the 5 running ones were restarted, so all 33 fresh readouts run at 0.44. gpu_mem sizes the KV cache only (memory knob); caps, samples, seed, answer caps, probe and code are unchanged. Retries per readout 4 (the parity self-test lora_visible check failed once on 4b_ctrl_1 with the fresh prompts; a retry reruns it unchanged).
- Amendment 2026-10-12 06:00 Israel, supersedes the 05:58 line (still before any fresh readout finished): at 0.44 a second readout also cannot start beside a running one (a running readout holds about 46 GiB). So every fresh readout runs ALONE on its GPU (scheduler cost 2) at the pre-registered gpu_mem 0.45; the readouts started at 0.44 were stopped before finishing. No fresh readout argument differs from the pre-registration; only the packing (one readout per GPU) and the retry count (4) changed.
- Amendment 2026-10-12 09:40 Israel (owner approval "approve", relayed by the coordinator): the pre-readout self-test of readout_vllm.py uses effect_visible 4.0 instead of 5.0 for the fresh-set readouts (test_fresh.jsonl) of the Qwen3.5-4B checkpoints ctrl_s1_cap512_r_box and wide_s2_cap512_r_box (step 150), which failed it on the visibility checks alone with the fresh prompts (ctrl_1: lora effect 0.0405 vs bar 0.0419; wide_2: lora 0.0433 vs 0.0436, gate 0.0398 vs 0.0436; every agreement, red-arm and start check passing). Every other check and every other readout keeps its bar; the self-test runs before generation, so scored generations and scoring are unchanged. Ledger entry src/gdn_gate/ml7/amendments/ledger.json (readout_vllm.py 9ef24b62 -> b7c477c2), CPU test fresh/test_effect_visible_cpu.py (red arms), the fresh code lock updated by that one documented line.
- Step 14 final test: fresh_rows_read_by_all_models=1.0000 (touch 1, 2026-10-12)
- 2026-10-12 H5/H6 (fresh_analysis.py, fresh set, 130 items x 8 samples, touch 1): H5 wide1 - ctrl area +0.110 (3/3, CI
  [0.089, 0.132]) supported; H6 wide1 - ctrl natural length -1,329 tokens (3/3, CI [-1,585, -1,081]) supported.
  wide1 - ctrlpm: area +0.098 and length -1,231, both effects; wide - wide1 area +0.012 effect (3/3), length no effect;
  ctrlpm - ctrl no effect on either. 2B secondary (6 seeds): H5-2B +0.012 and H6-2B -126 tokens, not supported (signs
  split); wide - ctrl length -307 effect. Verdict and publication are the owner's.
- Correction 2026-10-12 (coordinator, append-only): the amendment line above is stamped "09:40 Israel", which is when it was written. The owner's "approve" is in the chat at 06:11:52 UTC (09:11 Israel); the code change and the 4b_ctrl_1 requeue followed at 06:19 UTC (09:19 Israel). Other fresh readouts had finished by then; the approval request carried only the self-test values, and no fresh-set score was computed before the single analysis run (guard touch 1).
- Correction 2 2026-10-12 (coordinator, append-only): the previous correction's "no fresh-set score was computed before the single analysis run" is wrong as written. At the 06:11 UTC approval, 8 finished 4B fresh readout files (untouched, warm-up, ctrl_0, ctrl_2, ctrlpm_0, ctrlpm_1, wide1_0, wide1_1), each holding its own per-budget accuracy summary, were already on the workstation (pulled and sha256-checked by the puller). What holds: no contrast or hypothesis test was computed before the single analysis run (guard touch 1); the approval request and its decision used only the two failing self-test values; the coordinator did not open those readout files. The two rescued readouts (ctrl_1, wide_2) are not chosen by outcome: the rule applies to any 4B fresh readout that fails visibility alone, and ctrl_1 turned out to give the smallest wide1 - ctrl seed difference (+0.082).
