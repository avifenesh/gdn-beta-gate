# ML pipeline: step 7, length-cost GRPO (Qwen3.5-2B record)

Owner direction 2026-10-08: step 7 follows the ml-pipeline discipline. Scope: GRPO with a correctness-gated length
penalty on Qwen3.5-2B (thinking mode), arms wide / wide1 / ctrl, 3 seeds each, readout = accuracy vs reasoning tokens.
Code under study: `src/gdn_gate/broad/` (train_grpo.py, rl_common.py, readout_len.py, arms.py) and
`src/gdn_gate/vllm/` (budget_vllm.py, gate_vllm.py). Notebook: `ml_pipeline/01_eda.py` (marimo; run
`marimo edit ml_pipeline/01_eda.py`, or `python ml_pipeline/01_eda.py` to regenerate every artifact below);
`ml_pipeline/02_prep.py` covers steps 4 to 8.
Scratch venv for the notebook: `~/.cache/gdn_gate/mlp/venv` (pandas 3.0.6, matplotlib 3.11.2, marimo 0.25.1,
transformers 5.17.0 tokenizer only, math-verify 0.9.0).

## Checklist

| # | Step | Status |
|---|---|---|
| 1 | Data inspection | done 2026-10-08 |
| 2 | Exploratory data analysis | done 2026-10-08 |
| 3 | Define the prediction problem | done 2026-10-08 |
| A | Gate A | approved-gate 2026-10-08 |
| 4 | Data cleaning | done 2026-10-08 |
| 5 | Data engineering | done 2026-10-08 |
| 6 | Train / validation / test split | done 2026-10-08 |
| 7 | Feature engineering | done 2026-10-08 (not applicable, reasons in step 7) |
| 8 | Preprocessing | done 2026-10-08 (not applicable, reasons in step 8) |
| B | Gate B | approved-gate 2026-10-08 |
| 9 | Baseline model | todo |
| 10 | Model training | todo |
| 11 | Hyperparameter tuning | todo |
| 12 | Model evaluation | todo |
| C | Gate C | overridden by owner 2026-10-10 |
| 13 | Error analysis | todo |
| 14 | Final test (test set touched once) | todo |
| 15 | Deployment (only if the owner asks) | todo |
| 16 | Monitoring and retraining plan | todo |
| D | Gate D | todo |

- Override: Gate C - user approved launching the step 14 test without the validation review 2026-10-10 16:30 - reason: owner, in chat: "approved", answering the offer to skip the Gate C review and launch the test once all 9 arms finish. Kept: the test set is touched once, after all 9 arms and with the step 14 analysis (warm-up baseline) frozen; step 12 validation readouts and figures still run on the workstation for the record.
- Gate B: approved 2026-10-08 15:47 (owner, in chat: "approved", after the summary of baselines, candidates, recipe, ruled-out list and metric, with the condition that rollouts and scoring run the GDN delta rule in fp32 for every arm if the Olmo 2x2 rerun confirms the bf16 beta > 1 noise). Not committed (worktree rule).
- Gate A: approved 2026-10-08 (owner, in chat: contract as written, 2B model, 500-item validation slice sets the cap, primary metric = area under accuracy vs log budget 256-8k). Not committed: the lane keeps this worktree uncommitted by rule.

## Step 1: data inspection (done)

Raw files, read-only (sha256 recorded in `step1_3_summary.json`):

| set | role | rows | file | frozen by |
|---|---|---|---|---|
| gsm8k_train | RL prompt pool | 7,473 | `<runs>/vllm/data/gsm8k_train.jsonl` | vllm stream; no hash manifest existed, sha256 `efa1d1f7b903...` recorded here as the freeze |
| gsm8k_test | held out | 1,319 | `<runs>/broad/eval_data/gsm8k.jsonl` | broad stream; sha256 matches `eval_data/manifest.json` |
| math500 | held out | 500 | `<runs>/broad/eval_data/math500.jsonl` | broad stream; sha256 matches `eval_data/manifest.json` |

Manifest: `ml_pipeline/manifest.parquet` (with question text) and `manifest.csv` (without), one row per item: id,
set, source, role, level, subject, question length in characters and in Qwen3.5 tokens (tokenizer of
`broad/models/Qwen3.5-2B`, revision 15852e8c), full prompt tokens (chat template, enable_thinking, boxed-answer
suffix as in rl_common.Thinker), answer type, answer tokens, log10 |answer| for numeric answers, and whether the
item was one of the 50 MATH-500 items the pilot7 readout already looked at (`in_pilot50`).

Results (all measured):
- 9,292 rows, 11 profiled columns; no missing values except `answer_log10_abs` (1.2%, the non-numeric MATH-500
  answers). No exact duplicate rows, no duplicate ids, no duplicate questions inside any set.
- Question length (Qwen3.5 tokens): GSM8K train median 56 (95th pct 100, max 212), GSM8K test median 57 (95th pct
  105, max 188), MATH-500 median 50 (95th pct 215, max 831). Template plus instruction add 26 tokens per prompt.
- Answers: every GSM8K answer (train and test) is an integer, median about 10^1.6. MATH-500: 65.4% integer, 11.2%
  fraction, 1.0% decimal, 17.0% expression, 3.8% tuple or interval, 1.6% text.
- `guard.profile` (`data_profile.json`) ran with target = `set`, because the RL targets (correctness, reasoning
  tokens) only exist after generation. Read its traits as "how the three sets differ", not as the RL target:
  `imbalanced: true` means MATH-500 is 5.4% of the rows, nothing more. No leakage suspects from the profile
  (no column predicts the set at 98% purity or 0.95 correlation).
- Train/test overlap (GSM8K train against each held-out question; `overlap.json`, `nearest_train.csv`). Method:
  hashed word 1-3 gram cosine finds the 10 closest train questions, then a character match ratio (difflib, no
  autojunk) on normalized text picks the closest; the same again with every number masked.
  - Exact copies: 0 in GSM8K test, 0 in MATH-500.
  - Near-duplicates (ratio 0.85 or more and the same gold answer): 0 and 0.
  - Template siblings (same wording, different numbers and answer; ratio 0.85 or more once numbers are masked):
    3 in GSM8K test (test 602 / train 5162, and train 1314 at ratio 0.945 with numbers kept; test 632 / train 20, test 824 / train 3726), 0 in MATH-500.
  - 12 GSM8K test items have a train question at ratio 0.70 or more; read by eye they are the same GSM8K story
    templates with different numbers. MATH-500 has none above 0.64.
  - The broad stream's MATH train file already removed MATH-500 problems (0 found, `eval_data/manifest.json`).

## Step 2: EDA (done)

Figures are in `ml_pipeline/figures/` with their explanations in the Figures section below. The four `guard`
figures (missingness, target balance, distributions, correlations) describe the manifest with `set` as the target,
so "majority-class dummy 80.4%" there means 80.4% of rows are GSM8K train, not a model score.

Hypotheses and problems spotted:
- The pilot's reward was flat because the set was too hard for the cap, not because the recipe is broken: on MATH
  the untouched 2B finishes its thinking under 2,048 tokens on 3 of 50 items, and every correct 8k answer came from a
  finished trace (14 of 14 finished traces correct, 0 of 36 cut-off traces correct). GSM8K is the right difficulty
  for a 2k-class cap, but its base accuracy at each cap is not measured yet: the GSM8K budget job
  (`src/gdn_gate/vllm/budget_spark.sh`) has no receipt.
- At 512 to 4,096 tokens the MATH-500 curve is flat within noise (n = 50, intervals about plus or minus 0.11), so the
  readout needs all 500 items, and GSM8K test (1,319) is the better-powered set for the arm comparison.
- GSM8K rewards check integers only; MATH-500 needs symbolic matching (math_verify). MATH-500 is the transfer test,
  GSM8K test the in-domain test.
- GSM8K is built from story templates, so a few test items have train siblings (3 at 0.85 masked). Small, but
  removing them from the RL pool costs nothing.
- Selection risk already present: the pilot plan said to pick the training cap from the MATH-500 readout. That
  would choose a hyperparameter on the test set. The contract below moves that choice to a validation slice.
- Forced answers are scored from a single sampled answer per cap, so per-item results at small caps are noisy (one
  item is right at 512 and 1,024, wrong at 2,048, right at 4,096). Seed-level and item-level resampling are needed.

## Step 3: problem contract (done)

- **Target.** For each item and reasoning budget c: (a) correct, 1 if the last `\boxed{}` in the answer matches the
  gold answer under rl_common.correct (math_verify, string fallback), else 0; (b) reasoning tokens, the number of
  Qwen3.5 tokens before `</think>`, at most c. Also logged: total completion tokens, cut-off (forced) flag, silent
  stop flag, answer-ended flag.
- **Unit and population.** One problem, one vendor-sampled completion per item per (arm, seed, budget), budgets as
  exact prefixes of one long sample (readout_len.py semantics). Population: grade-school word problems (GSM8K,
  in-domain) and competition problems (MATH-500, transfer).
- **Known at prediction time.** Only the chat-templated problem with the boxed-answer instruction, and the budget
  (a trace still thinking at c gets `\n</think>\n\n` appended and its answer sampled). The gold answer is never in
  context; it is used only by the grader (the reward in training, the score in readouts). Group statistics (mean
  and spread of correct lengths) exist only inside the training reward.
- **Objective.** Cut reasoning tokens while keeping accuracy, and answer the lane question: under identical
  length-cost GRPO, does wide (gate beta in (0,2), negative eigenvalues allowed, plus LoRA r16) move the
  accuracy-vs-tokens curve more than wide1 (same added parameters, beta in (0,1)) and ctrl (LoRA r16 only)?
- **Metric.**
  - Primary, declared now so it cannot be picked later: normalized area under accuracy against log2(budget), budgets
    256, 512, 1,024, 2,048, 4,096, 8,192, per set, mean of 3 seeds per arm. Higher means more accuracy for fewer tokens.
  - Accuracy at each matched budget (the same six caps for every arm).
  - The full accuracy-vs-tokens curve (accuracy against mean reasoning tokens used).
  - Tokens at fixed accuracy: the budget (log2-interpolated) at which an arm reaches the untouched model's accuracy at
    8,192, per set.
  - Natural length: mean reasoning tokens at 8,192, and the cut-off share.
  - Checks reported beside every number: cut-off share, silent stops, missing `\boxed{}` rate.
  - Decision rule for an arm effect: all 3 seeds agree in sign, and the paired item bootstrap interval of the
    difference (wide minus wide1, wide minus ctrl) excludes 0. Otherwise the result is reported as no effect.
- **Constraints.** Owned hardware only: the workstation RTX 5090 Laptop (24 GiB) and the two DGX Spark GB10 units, through
  gpulock and sparklock; no paid compute and no 27B job. Rollouts on vLLM with the gate plugin (parity accepted
  for generation, `results/vllm/parity_4b_local.json`); tier-1 state-tracking scoring stays on HF.
  Readouts use the Qwen3.5 card's thinking sampling (temperature 1.0, top_p 0.95, top_k 20, presence 1.5);
  rollouts use plain temperature 1.0 as in train_grpo.py; greedy only for verification. Seeds 0, 1, 2 per arm. One
  recipe for all arms (LoRA r16, alpha 0.2, lr 2e-5, 16 prompts x 8 samples, 150 steps, no KL, Dr. GRPO advantage),
  with the training cap chosen once on validation and then fixed.
- **Train / validation / test plan.**
  - Train (RL prompt pool): GSM8K train minus the validation slice minus every train item flagged as a sibling of a
    held-out item. About 6,970 items; a 150-step run draws 2,400 prompts.
  - Validation: 500 GSM8K train items chosen by sha256 of the id (fixed, never trained on). Used for the training
    cap (base accuracy 30 to 60% at the cap, measured with budget_vllm.py on this slice), the untouched-model
    validation curve, and monitoring during training. The checkpoint read out is the last step, fixed in advance, so
    no checkpoint is picked by score.
  - Test, touched once: GSM8K test (1,319) and MATH-500 (500), one readout per arm and seed plus the untouched model,
    run only after all 9 runs finish and the analysis script is frozen. The pilot already read 50 MATH-500 items with
    the untouched model; MATH-500 is reported on all 500 and on the 450 unseen items as a check.
- **Baselines.** (1) The untouched Qwen3.5-2B curve, same readout and caps (validation at step 9, test at step 14).
  Its hard-truncation curve is also the "just cap it" baseline. (2) The ctrl arm under the same RL: the bar the gate
  arms must beat. (3) wide1, which separates negative eigenvalues from extra parameters.
- **Model playbook note.** `references/model-selection.md` covers tabular, text classification and images; RL
  fine-tuning of a pretrained LLM is outside it. The model family is fixed by the lane (Qwen3.5-2B, three arms), so
  at Gate B the rationale block records the arm design and the baselines above instead of a family search.

## Step 4: data cleaning (done)

Notebook `ml_pipeline/02_prep.py` (marimo; `python ml_pipeline/02_prep.py` rebuilds every file below). No rule used
model outputs or test scores; every rule reads only question text and gold answers. Before / after counts:

| rule | what | before | removed | after |
|---|---|---|---|---|
| 4.1 | GSM8K train items that are a near-duplicate or template sibling (match ratio 0.85 or more with numbers kept or masked, top 20 neighbours by n-gram cosine) of any held-out item | 7,473 | 4 (train 1314, 5162, 20, 3726; siblings of GSM8K test 602, 632, 824; none of MATH-500; 0 same-answer near-duplicates) | 7,469 |
| 4.2 | remaining train items that are siblings of a validation item (so the validation curve is not lifted by trained-on templates) | 6,969 (after the 500 validation items) | 4 (train 5168, 3849, 4079, 3252) | 6,965 |
| 4.3 | gold answers: present, non-empty, readable by rl_common.gold_of | 9,292 | 0 | 9,292 |

Grader self-check (rule 4.3, `grader_check.json`, figure 04_grader_selfcheck). rl_common.correct, run on 84,077
answers built from every item's gold, once with math_verify 0.9.0 and once with it blocked:
- Accept arms (should pass): gold verbatim 100% / 100%; with spaces 100% / 100%; `18.00`, `\$18`, `18 dollars`,
  `1,000` 100% with math_verify, 0% with the fallback; `\dfrac` for `\frac` 100% / 100%.
- Red arms (must fail): wrong value, no box, empty box, a wrong last box are accepted 0.00% by both. The check is
  decisive: the graders separate on the accept arms and both reject every red arm.
- Finding (measured): math_verify is not installed in any venv the step 7 code runs in: workstation `<venv>`, and
  `<work>/venv` and `<work>/venv-vllm` on both Sparks (directory listing, 2026-10-08). rl_common.correct
  catches the ImportError and silently uses the string fallback. So every pilot7 accuracy number was graded by the
  fallback, which marks right answers wrong whenever the model writes them in a harmless other form. How often the
  2B does that is not measured (pilot7 kept 2 texts per step). Step 7 must not run on the fallback:
  `src/gdn_gate/ml7/preflight_grader.py` fails closed without math_verify (red arm checked: rc 1 in the
  workstation trainer venv, rc 0 in a venv with math_verify 0.9.0).

## Step 5: data engineering (done)

Frozen files in `<runs>/ml7/data/` (sha256 in `ml_pipeline/split_manifest.json`):

| file | rows | sha256 |
|---|---|---|
| rl_pool.jsonl | 6,965 | 1cc1ef6b8f8339cd84847ced1c52f799db0095984def0afbe6142d5afec6c448 |
| val.jsonl | 500 | 8251b0d493e1371f4aece78b1e163708078d8cda35ece92c2d16f7b6892750dd |
| test_gsm8k.jsonl | 1,319 | bfa80506b84afadbb7c6f0dd261d580ca5407893c973b87bf666ad45ad709652 |
| test_math500.jsonl | 500 | 3283af634ac823b31a6e5d0d7bdbc7c5350ce90a81f94217007b916c3a3bb337 |

- One row per prompt: id, problem, answer, set, level (`gsm8k` or the MATH level), subject for MATH-500, and for the
  GSM8K files a `solution` field `\boxed{answer}` so the existing train_grpo.py pool loader reads them unchanged
  (`pool_rows(rl_pool.jsonl, ["gsm8k"])` returns 6,965 rows, gold equals answer on all; checked on the workstation CPU).
- Quality checks in the notebook: unique ids across all four files, every gold non-empty, every prompt non-empty,
  pool + validation + dropped = 7,473, test counts 1,319 and 500. The 50 MATH-500 items the pilot saw are listed in
  split_manifest.json (`pilot50_math500_ids`) for the 450-item check.

## Step 6: train / validation / test split (done)

- Validation: the 500 GSM8K train items with the smallest sha256(VAL_SALT + id) (VAL_SALT in 02_prep.py), drawn from items not
  dropped by rule 4.1. Train: the rest minus rule 4.2. Test: GSM8K test + MATH-500, fixed by the contract.
- `guard.check_no_overlap` passes on (problem, answer) and on problem text alone across train, validation and test.
- Test frozen: guard fingerprint `b28bc3d5cda5a27b30430c730ed381e95289e23f7a7c028616e3f6e8ce9de646` over 1,819 rows
  (columns id, problem, answer; GSM8K test then MATH-500, file order) in `ml_pipeline/.guard_state.json`,
  test_touches 0. guard.split() was not used: it draws its own random or stratified split and cannot express a
  fixed external test set, so the guard's own fingerprint and state format are written directly and
  guard.final_test() reads them at step 14.
- Figure 06_split_composition: validation question length (median 54) matches the pool (56) and GSM8K test (57).

## Step 7: feature engineering (not applicable)

- Not applicable: the model reads raw problem text through a fixed chat template (rl_common.Thinker.prompt:
  enable_thinking, boxed-answer suffix). Nothing is computed from data statistics, so there is nothing to fit on
  train and apply to test.

## Step 8: preprocessing (not applicable)

- Not applicable: tokenization is the model's own frozen tokenizer and template; no scaler, encoder or imputer
  exists. The one transform with a parameter, budget forcing at the cap, is part of the contract and is identical
  for every arm, the baseline and every split.

## Gate B preparation

Model rationale:
- traits: RL fine-tuning of a pretrained LLM (Qwen3.5-2B, thinking mode), not a tabular fit; 6,965 RL prompts (GSM8K train, integer answers), 500 hashed validation items, 1,819 fixed test items (GSM8K test 1,319, MATH-500 500); target per item = correct (0/1) and reasoning tokens at each budget; no datetime; template siblings removed instead of a group split; the profile's imbalanced flag is set membership only
- baseline: the untouched Qwen3.5-2B curve (hard truncation at each cap, same readout and sampling) and the ctrl arm (LoRA r16 only) under the identical GRPO recipe
- candidates: wide (gate beta in (0,2) plus LoRA r16) and wide1 (same added parameters, beta in (0,1)), 3 seeds each; recipe for all arms: GRPO with the correctness-gated length penalty (alpha 0.2), Dr. GRPO advantage, no KL, lr 2e-5, 16 prompts x 8 samples, 150 steps, training cap set by step 9 on validation
- ruled out: Qwen3.5-4B (owner chose 2B at Gate A; 2.5x slower rollouts); MATH train as the RL pool (pilot7: 2.1% correct at the 2k cap, flat reward); picking the cap, a checkpoint or a seed on GSM8K test or MATH-500 (selection leakage; last step is read out); grading through the string fallback (rejects 100% of harmless rewrites such as 18.00, $18, 1,000); HF generate rollouts for full runs (92-97% of step time in pilot7)
- metric: normalized area under accuracy vs log2 budget, caps 256 to 8,192, per test set, mean of 3 seeds (primary); accuracy at each cap, tokens to reach the untouched model's 8k accuracy, natural length and cut-off share at 8k (secondary); an arm effect needs 3 of 3 seeds agreeing in sign and a paired item bootstrap interval excluding 0

Training cap, data rule (pre-registered 2026-10-08 about 21:00, before the per-sample data exists; owner: "answer should
come from data"). Step 9 cap mode put every cap in the 30-60% band (GB10, 500 val items x 4, train sampling: 256 0.454,
512 0.491, 1024 0.504, 2048 0.481, 4096 0.440), so the band does not choose. The cap is the one that maximizes GRPO
advantage mass per rollout token: for each cap, rerun cap mode with per-sample (correct, completion tokens) recorded;
for each item draw groups of 8 with replacement from its 4 samples (200 draws, seed 20261008), reward with
rl_common.length_reward (alpha 0.2), Dr. GRPO advantage A = r - mean(r); S(cap) = E[sum |A|] / E[sum completion
tokens] over all draws. Reported beside it: the correctness-only and length-only parts of S, and S's bootstrap 95%
interval over items. Ties within the interval go to the cheaper cap (fewer tokens per group).
Revised about 21:03, still before the per-sample data exists: a synthetic self-test showed total S is set by cost alone
(advantage mass per group is about equal across caps, so the cheapest cap wins), and at 256 99.85% of rollouts are
forced, so the length penalty, the point of length-cost GRPO, has almost nothing to act on there. Every cap in the
band already carries correctness signal (mixed groups at every cap). The cap is therefore chosen on S_length: the
length-penalty part of the advantage (r minus its correctness part) per rollout token, best by point estimate, ties
within its 95% interval to the cheaper cap; total S and the correctness part are reported, not used.
Revised again about 21:06, before any per-sample or pilot data exists: rl_common.length_reward z-normalizes lengths within
the group, so the length term is scale free and any per-token statistic of base-model samples ranks caps by cost alone
(the synthetic self-test picked the cheapest cap under both versions). The per-sample route is dropped. The cap is
chosen on the primary metric itself, on validation: ctrl, seed 0, the first 40 steps of the real 150-step schedule
(--stop_after 40, so the pilot is the start of that cap's full run) at caps 512, 1024 and 2048 (256: 99.85% of
rollouts forced, no reasoning to shorten; 4096: measured rollout cost 279 s per 128 samples, 3.7x 1024, about a week
for 9 runs). Each pilot's step-40 checkpoint is read out with readout_vllm.py on the 500 validation items (2 samples,
caps 256..8192, seed 20261006), the same instrument and seed as the step 9 untouched curve. Score = normalized area
under accuracy vs log2 budget minus the untouched curve's, paired over items, 95% bootstrap interval. The cap with the
largest gain wins; caps whose interval overlaps the winner's point estimate tie, and ties go to the cheaper cap.

Answer cap, data rule (2026-10-08 about 22:32, before the uncensored data exists). The step 9 curve (answer cap 512)
left 30.3% of samples at 8192 without a \boxed{} answer: after </think> the 2B writes a full step-by-step solution and
the cap cuts it (median 1,851 characters, about the cap); 38.5% at 256. The GRPO rollouts used 384. So both the primary
metric and the reward were measuring answer verbosity against an arbitrary cap. The cap pilots were stopped before
step 1. Rule: rerun the step 9 curve with --answer_max 4096 (same seed and items); answer_max = the smallest multiple
of 256 at or above the 99th percentile of answer tokens among answers that end within 4096, over all budgets. Because
sampling is per token with fixed seeds, the curve at that cap is the 4096 run truncated at the cap and regraded (no
second run). The same answer_max goes into the rollouts, every readout and step 14. Answer tokens are reported per
budget beside the reasoning tokens, so reasoning carried into the answer is visible.

Step 9 job, ready to queue after Gate B (`src/gdn_gate/ml7/step9_val_readout.sh UNIT MODE`): untouched
Qwen3.5-2B on val.jsonl through budget_vllm.py (stock vLLM engine) under sparklock, refusing to start if val.jsonl
does not match its frozen hash or if preflight_grader.py fails.
- `MODE=cap`: rollout sampling, caps 256 to 4,096, 4 samples per item; the training cap is the largest cap with
  base accuracy in 30 to 60%.
- `MODE=curve`: vendor thinking sampling, caps 256 to 8,192, 2 samples per item; the untouched validation curve.
- Each mode: spark_job timeout 3 h, lease TTL 200 min. Token volume at most about 16M per mode; vLLM tok/s for the 2B
  on GB10 is not measured yet, so the wall time is an estimate bounded by the timeout.
- math_verify is installed in `<work>/venv-vllm` on gb10a and in the workstation vLLM venv; gb10b had no vLLM venv until a copy of the gb10a one (same uv cpython 3.12.13) was made on 2026-10-08 16:10; preflight_grader.py checks it before the job starts.
- GDN precision (the condition attached to the design sign-off) resolved 2026-10-08 18:50: the Olmo 2x2 shows fp32 vs bf16 fla moves logprobs by about 0.013 on both stacks, an order of magnitude below any arm effect of interest; every arm runs bf16 (FINDINGS 18:50).

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
- Figure 12_val_accuracy_vs_budget.png: Validation slice (500 held-out GSM8K train items x 2 samples, one GPU for every model): accuracy against reasoning budget 256 to 8,192 tokens for the untouched model, the shared format warm-up and each arm (thick line = mean of its seeds, thin = single seeds). Area under the curve: untouched 0.728; warm-up 0.838; ctrl 0.896 (seeds 0.895/0.896); wide 0.888 (seeds 0.888); wide1 0.896 (seeds 0.896/0.894/0.899). The arms start from the warm-up, so the warm-up curve is the bar: an arm above it at small budgets got more accuracy for fewer reasoning tokens. PARTIAL: 6 of 9 arm runs present (ctrl seeds 0/1, wide seeds 0, wide1 seeds 0/1/2); redrawn as runs finish, no arm verdict until 3 seeds per arm.
- Figure 12_val_area_contrasts.png: Validation slice (500 held-out GSM8K train items x 2 samples, one GPU for every model): every pre-registered contrast on the primary metric, as the seed-mean difference in area with its paired item bootstrap 95% interval and the single-seed differences. Rule: an arm effect needs all 3 seeds with the same sign and an interval that excludes 0; warm-up minus untouched has one readout per side, so it is descriptive. wide - ctrl -0.0077 [-0.0234, +0.0080] incomplete (1 of 3 seeds), no verdict; wide1 - ctrl -0.0009 [-0.0104, +0.0083] incomplete (2 of 3 seeds), no verdict; wide - wide1 -0.0080 [-0.0219, +0.0060] incomplete (1 of 3 seeds), no verdict; ctrl - untouched +0.1673 [+0.1444, +0.1903] incomplete (2 of 3 seeds), no verdict; wide - untouched +0.1594 [+0.1350, +0.1841] incomplete (1 of 3 seeds), no verdict; wide1 - untouched +0.1677 [+0.1446, +0.1916] effect, positive; ctrl - warmup +0.0578 [+0.0418, +0.0738] incomplete (2 of 3 seeds), no verdict; wide - warmup +0.0499 [+0.0318, +0.0681] incomplete (1 of 3 seeds), no verdict; wide1 - warmup +0.0582 [+0.0431, +0.0741] effect, positive; warmup - untouched +0.1095 [+0.0916, +0.1278] descriptive. PARTIAL: 6 of 9 arm runs present (ctrl seeds 0/1, wide seeds 0, wide1 seeds 0/1/2); redrawn as runs finish, no arm verdict until 3 seeds per arm.
- Figure 12_val_accuracy_vs_tokens.png: Validation slice (500 held-out GSM8K train items x 2 samples, one GPU for every model): accuracy against the mean reasoning tokens each model actually spent at each budget, so a curve further left at the same height reasons less for the same accuracy. Natural length (mean reasoning tokens at 8,192) and budget to reach the untouched model's 8k accuracy (0.798): untouched 5730 tokens, cut off 0.467; warm-up 3947, cut off 0.222, reaches the untouched 8k accuracy at 275; ctrl 1735, cut off 0.046, reaches it at 256; wide 828, cut off 0.013, reaches it at 256; wide1 1412, cut off 0.035, reaches it at 256. PARTIAL: 6 of 9 arm runs present (ctrl seeds 0/1, wide seeds 0, wide1 seeds 0/1/2); redrawn as runs finish, no arm verdict until 3 seeds per arm.

Design note (owner, 2026-10-09 00:05): no per-arm alpha = 0 control in step 7. The question is the arm contrast (ctrl,
wide, wide1) under one reward and one optimizer (Dr. GRPO, which removes the length-normalization bias the census found
behind alpha = 0 shortening in Arora and Zanette); 9 runs as approved. Added from the census at no GPU cost: the
correctness-conditioned length report beside the area metric.

Design change (owner, 2026-10-09 10:00, from the cap 512 pilot readout): the pilot lifted the validation area 0.736 ->
0.853 mostly by teaching the 2B to end and box its answer (unboxed 29% -> 12%), reasoning pinned at the cap; that shared
format effect would sit under every arm near the ceiling. New design: (1) one shared format warm-up: ctrl LoRA,
correctness reward only (alpha 0), cap 512, answer_max 1792, 40 steps (run ctrl_s0_cap512_fmt_gb10); its step-40 val
readout is the baseline the arms are compared against; (2) every arm (ctrl, wide, wide1 x 3 seeds) initializes its LoRA
from the warm-up checkpoint and trains with the length penalty on reasoning tokens only (alpha 0.2), so the arms
differ on reasoning efficiency; (3) the training cap is chosen by the same pilot rule (ctrl, 40 steps at 512/1k/2k,
largest paired val area gain, ties to the cheaper cap), now from the warm-up init and against the warm-up baseline. The
old-design pilots at 512 and 1k (both read out) stay as recorded diagnostics; the old cap 2k pilot was stopped 45 min in.
- Step 14 final test: test_rows_read_by_all_models=1.0000 (touch 1, 2026-10-11)

## Extension: seeds 3-5 (owner approval 2026-10-11 13:00, in chat: "go ahead" to item 5)

- Override: final test - user approved a second read of the test set (touch 2) for the 18 new readouts of the seeds 3-5 extension only (ctrl, wide, wide1 x seeds 3, 4, 5 x GSM8K test and MATH-500) 2026-10-11 13:00 - reason: owner, in chat: "go ahead" to the pre-registered extension: seeds 3-5 judged alone by the frozen rule (step14_analysis.py --seeds 3,4,5), the touch-1 verdict kept as this record's verdict, pooled seeds 0-5 descriptive; the untouched and warm-up test readouts of touch 1 are adopted by sha256 (STEP14_REUSE), so the 18 arm readouts are the only new reads of the test set.

- Readout code check of the adoption (2026-10-11): the STEP12_REUSE / STEP14_REUSE check accepts readout_vllm.py only
  when it equals the reused set's code lock or differs from it by owner-approved amendments recorded in the ledger
  src/gdn_gate/ml7/amendments/ledger.json (old and new sha256, diff file and its sha256, approval;
  code_ledger.py reverse-applies each diff and refuses any other change). Two amendments are recorded: parity_k 2.5
  for one 2B checkpoint (owner 2026-10-11 ~10:50) and the --probe_effect argument, default 0.2 (owner 2026-10-11
  ~14:55, for the Qwen3.5-4B record). Neither changes what a 2B readout computes; the seeds 3-5 readouts run at the
  default probe 0.2 like seeds 0-2.

Proposal as approved (written before the approval, kept as it was):

- Owner direction 2026-10-11 ("do": more seeds, about one box-day). Proposal: 9 more arm runs (ctrl, wide, wide1 x
  seeds 3, 4, 5) with this record's recipe, cap 512, answer cap 1,792, warm-up checkpoint (ctrl_s0_cap512_fmt_gb10
  step 40) and step-150 readout, trained on the rented PRO 6000 box (run tag r_box; seeds 0-2 trained on GB10; each
  seed's three arms on one hardware, as in step 6).
- The touch-1 result above stays this record's verdict and is not revised. Seeds 3-5 are a replication set read by
  the same analysis (step14_analysis.py --seeds 3,4,5; its default reproduces the frozen script's output exactly,
  checked on the step 12 box receipts), with the same rule applied to seeds 3-5 alone. Contrasts named before any
  extension run: H1, MATH-500 wide - ctrl area > 0; H2, GSM8K natural length (reasoning tokens at the 8,192 budget)
  wide < ctrl. Pooled seeds 0-5 numbers are reported as descriptive, with no verdict.
- Validation: the step 12 box set for seeds 3-5 (results/ml7/step12_box_ext/; the untouched and warm-up
  readouts adopted from step12_box/ after a sha256 and readout-code check).
- Test: a second read of the test set (touch 2) for the 18 new readouts only (9 runs x 2 sets); the untouched and
  warm-up test readouts of touch 1 are adopted, not read again. This needs the owner's agreement to a new test
  strategy, recorded here as the guard's line only after that agreement; until then step14_test_box.sh refuses.
- Why a second read can be honest here: nothing is selected with it (fresh runs; recipe, cap, checkpoint and
  analysis frozen). The risk is optional stopping (more seeds because the first three looked good), handled by
  keeping the touch-1 verdict, naming the contrasts now and reporting seeds 3-5 whatever they show.
- Alternative with no second read: a fresh set never read in this lane, the 4,499 MATH test problems outside
  MATH-500 (MATH-lighteval test, minus MATH-500, minus 1 problem also in the broad MATH train file), a hashed
  1,000-item subset frozen before any extension run, read once by all 6 seeds, the untouched model and the warm-up
  (about 29 extra readout slot-hours on the box).
- Step 14 final test: test_rows_read_by_all_models=1.0000 (touch 2, 2026-10-11)

- 2026-10-12: the 2B arms (ctrl, wide, wide1 x seeds 0-5) and the untouched 2B are also read on a fresh held-out set (AIME 2024/2025/2026, AMC 2023) with its own guard state; pre-registration in ml_pipeline_4b/PIPELINE.md, "Fresh test set". This record's test set is not read again.
