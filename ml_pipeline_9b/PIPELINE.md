# ML pipeline: step 7 at Qwen3.5-9B, length-cost GRPO, AIME 2025 read

This record holds the 9B size point of lane wg. The 9B arms were trained and read on the validation slice under
the lab's planning note for the follow-up study (not released; its sections "9B validation run" and "9B seeds 1 and 2" report the receipts in results/ml7_9b/). The record reads one test
set: AIME 2025, once, under its own guard state (ml_pipeline_9b/aime2025_test, test touches 0). It reads no other test
set: GSM8K test, MATH-500 and the other fresh-set rows stay untouched by every 9B model.

## Owner direction (2026-10-15, Israel time)

- ~23:40: "1" to the proposal "Harder read on these same six 9B models: MATH-500 plus the fresh competition set. This
  is the read that tests the accuracy claim at 9B. It would be the first read of those test sets on any 9B model."
- ~23:50 (relayed by the coordinator): the goal is to get off the GSM8K ceiling, and MATH-500 would sit near ceiling at
  9B too, so read the fresh set only, with the 6 trained arms; MATH-500 stays untouched for 9B.
- ~23:55: "why pushing to 130? go with generic aime". So the read is the standard AIME 2025 benchmark only: the 30 AIME
  2025 problems (I and II), 8 samples each, the 4B fresh readout protocol, the 6 trained arms.

## Data

- File <runs>/ml7/data/test_aime2025.jsonl, 30 rows, sha256
  f45253f801b78bf81e462061f7f3b618ab2d0b55d77734d4dfecf590f78baa43; manifest aime2025_manifest.json beside it.
- Cut 2026-10-15 by src/gdn_gate/ml7/test9b/build_aime2025.py from the frozen Phase 5 fresh file
  test_fresh.jsonl (sha256 9ce51dff11355439..., the 4B record's "Fresh test set"): every line whose id is
  fresh:aime2025-<I|II>-<n>, in file order, bytes unchanged. Only the id field was read; no problem text was printed
  or inspected. Upstream: MathArena/aime_2025 revision c94da77e (fresh_manifest.json), integer answers, deduplicated
  against the RL pool and the validation slice by the Phase 5 sibling scan (0 hits).
- Guard: ml_pipeline_9b/aime2025_test/.guard_state.json, fingerprint 3aa2a7d749c5315f... (frame id, problem, answer in
  file order, guard._fingerprint), test touches 0; ml_pipeline_9b/aime2025_test/split_manifest.json holds the file's
  sha256 for the analysis and the guard.
- What was seen before: the 4B and 2B records read the whole fresh set once (2026-10-12), and their analysis reports
  AIME 2025 as a descriptive per-source row for those models. No 9B model has read any AIME 2025 problem.
- Data period: AIME 2025 (February 2025) predates the Qwen3.5 release, so it may be in the base model's training
  data. Contamination would raise every arm alike (all share the base); it is not testable here.

## Models (Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a)

- Model files on both Spark units identical, sha256 of all 13 files in ml_pipeline_9b/model_Qwen3.5-9B.sha256 (checked
  2026-10-15; every readout job checks them again on its unit before it starts).
- Recipe (the v1 recipe on Qwen3.5-9B, the lab's planning note): GSM8K GRPO, length penalty alpha 0.2 on reasoning
  tokens, training cap 512, answer cap 1,280, LoRA r16 on the GDN projections, AdamW lr 2e-5, a shared 40-step format
  warm-up (ctrl_s0_cap512_fmt9b_gb10, correctness only) and then 150 steps from it; micro-row 4,096 (a memory knob).
- The six trained models, last checkpoint (step 150), sha256 in src/gdn_gate/ml7/test9b/ckpt_expect_9b.json
  (equal to the sha256 each one's validation readout recorded, and to the workstation and unit copies):
  - LoRA (ctrl): ctrl_s0_cap512_r9b_gb10, ctrl_s1_cap512_r9b_gb10, ctrl_s2_cap512_r9b_gb10
  - Gate(0,1) (wide1, gate beta in (0, 1) plus LoRA r16): wide1_s0_cap512_r9b_gb10, wide1_s1_cap512_r9b_gb10,
    wide1_s2_cap512_r9b_gb10
  - every run's meta.json: init from the warm-up checkpoint b109f3779eab01f7..., cap 512, answer cap 1,280, rank 16,
    probe 0.05, alpha 0.2.
- One more readout, the untouched Qwen3.5-9B: the frozen analysis needs it as an input (step14_analysis.py and
  fresh_analysis.py take --baseline as a required argument, and S.validate refuses unless it is an arm-stock readout:
  it is the reference for caps, items, samples, sampling, GPU and data sha256). The warm-up is optional in the frozen
  analysis and is not read. So 7 readouts.
- Validation results, known before this pre-registration (GSM8K validation slice, 500 items x 2): Gate(0,1) - LoRA
  natural length -1,324 tokens pooled (3 of 3 seeds, 63-79% shorter), accuracy at 8k flat (-0.000, [-0.006, +0.006]).
  GSM8K sits near the 9B ceiling, so it cannot show an accuracy gain; that is the reason for this read.

## Readout

- readout_vllm.py, the instrument of the 9B validation readouts and of the 4B fresh read (sha256 b7c477c2..., equal to
  the 4B fresh code lock results/fresh/code.sha256): caps 256, 512, 1,024, 2,048, 4,096, 8,192; answer cap
  1,280; samples 8 per problem (as the 4B fresh read); seed 20261006; vendor thinking sampling (temperature 1.0,
  top_p 0.95, top_k 20, presence 1.5); parity self-test probe 0.05 at k 2 with the default visibility bar 5.0;
  gpu_mem 0.35 (the 9B validation readouts on GB10); --rank 16; --arm stock (untouched), ctrl (LoRA), wide1 (Gate).
- Hardware: the two DGX Spark GB10 units (owned), one readout per unit at a time under sparklock, every readout on
  "NVIDIA GB10" (the analysis refuses mixed GPUs). Runner src/gdn_gate/ml7/test9b/test9b_spark.sh.
- Placement (owner: gated and LoRA arms interleaved across the units, both finishing about together): gb10a reads
  untouched, ctrl_0, wide1_0; gb10b reads ctrl_1, wide1_1, ctrl_2, wide1_2. Each seed pair sits on one unit, so a unit
  difference cancels in its seed-paired difference. A unit that empties its queue first takes the other's last
  unstarted label (it then starts earlier than it would have); the receipts record which unit read each file.
- Self-test failures: a readout that fails the pre-readout engine/trainer self-test is retried once unchanged; if it
  fails again the read stops before any analysis and the guard records no touch. No bar or argument changes without
  the owner, and no finished readout file is opened while the read is stopped.

<!-- the pre-registration block below is read by test9b_guard.py preflight: its sha256 lines must match the files -->

## AIME 2025 test read at 9B (pre-registration, owner approval 2026-10-15)

Written 2026-10-16 ~00:05 Israel, before any 9B readout of AIME 2025: no readout of test_aime2025.jsonl exists, no job
has been started, and the guard state shows test touches 0. This is touch 1 of AIME 2025 for the 9B record, and the
only one.

The 4B record's fresh-set pre-registration, quoted from ml_pipeline_4b/PIPELINE.md ("Fresh test set (pre-registration,
Phase 5, owner approval 2026-10-12)"):

> - Metric and rule (unchanged): normalized area under accuracy vs log2 budget (norm_auc_log2) and natural length
>   (reasoning tokens at 8,192), pooled over the 130 items; an effect needs every seed's difference to share a sign AND
>   the paired item bootstrap 95% interval of the seed-mean difference to exclude 0 (10,000 draws, numpy
>   default_rng(20261009)). Analysis: src/gdn_gate/ml7/fresh/fresh_analysis.py, which imports the frozen
>   step14_analysis.py and ctrlpm_analysis.py unchanged and only sets their module constants (test file name, id
>   prefix, two hypotheses); CPU-tested before the read.
> - Pre-registered confirmatory hypotheses (4B, seeds 0-2):
>   - H5: fresh set, wide1 - ctrl area > 0 under the decision rule.
>   - H6: fresh set, wide1 - ctrl natural length < 0 under the decision rule.
>   - Reported under the same rule, no claim of their own: wide1 - ctrlpm, wide - wide1 (area and length), wide - ctrl,
>     wide - ctrlpm, ctrlpm - ctrl, each arm - warm-up and - untouched; per source (AIME 2024, 2025, 2026, AMC 2023)
>     descriptively only. 2B (seeds 0-5): the H5 and H6 contrasts and the rest under the 6-seed rule, a secondary
>     replication (H5-2B, H6-2B).
>   - Reading named now: H5 and H6 supported = the 4B gated arms keep both the accuracy gain and the short reasoning on
>     harder, newer problems. H6 alone = shorter but no accuracy gain on these items. Neither = the effect does not reach
>     this difficulty. AIME at an 8k budget may put the 4B near its accuracy floor; that limits H5's power and is reported.

Applied to 9B Gate(0,1) vs LoRA on AIME 2025, seed-paired (seed s of wide1 minus seed s of ctrl, s = 0, 1, 2; both arms
of a seed start from the same warm-up checkpoint):

- Metric and rule, unchanged: norm_auc_log2 over caps 256 to 8,192 and natural length (reasoning tokens at 8,192),
  pooled over the 30 AIME 2025 problems x 8 samples (per problem the mean over its samples); an effect needs all 3
  per-seed differences to share a sign AND the paired item bootstrap 95% interval of the seed-mean difference to exclude
  0 (10,000 draws, items resampled with replacement, numpy default_rng(20261009), one index matrix for every contrast).
- Pre-registered confirmatory hypotheses (9B, seeds 0-2):
  - H5-9B: AIME 2025, wide1 - ctrl area > 0 under the decision rule.
  - H6-9B: AIME 2025, wide1 - ctrl natural length < 0 under the decision rule.
  - Reported under the same rule, no claim of their own: ctrl - untouched and wide1 - untouched (area and length);
    accuracy per budget and the descriptive blocks of the frozen analysis. Not applicable at 9B: every contrast with
    wide, ctrlpm or the warm-up (not trained or not read here), the per-source rows (one source), the 2B rows.
  - Reading named now: H5-9B and H6-9B supported = the 9B gated arms keep both the accuracy gain and the short
    reasoning on harder problems. H6-9B alone = shorter but no accuracy gain on these problems. H5-9B alone = more
    accurate but not shorter, reported as such. Neither = the effect does not reach this difficulty at 9B. AIME at an
    8k budget may put the 9B near its accuracy floor; that limits H5-9B's power and is reported.
- Sample size: 30 problems is small. The paired item bootstrap interval over the 30 problems is the main uncertainty
  and is reported with every estimate; 8 samples per problem reduce sampling noise within a problem, not the
  problem-to-problem spread. The per-seed differences are reported beside it. No multiplicity correction (as the 4B
  record: two confirmatory hypotheses, each read on its own).
- Analysis: src/gdn_gate/ml7/test9b/test9b_analysis.py. fresh_analysis.py cannot take this read unchanged:
  its --record accepts 4b (four arms incl. ctrlpm) or 2b (three arms incl. wide), and its file constant is
  test_fresh.jsonl. The wrapper imports fresh_analysis.py and step14_analysis.py unchanged, runs FA.setup(), and sets
  only module constants: TEST_FILES = (test_aime2025.jsonl,), ARMS = (ctrl, wide1), CONTRASTS = ((wide1, ctrl),),
  HYPOTHESES = H5/H6 with the spec fields of fresh_analysis.H56 and the text naming AIME 2025 at 9B. It then runs
  S.validate and S.analyze (fresh_analysis's 2b path), seeds 0, 1, 2, no warm-up, and refuses unless every readout has
  rank 16, probe 0.05, answer cap 1,280, 8 samples and only fresh:aime2025- ids. CPU tests:
  src/gdn_gate/ml7/test9b/test_test9b_cpu.py (N2: supported on synthetic gated-better-shorter readouts,
  not supported on the red arm with identical rows, seven refusals).
- Guard: one guard.final_test() (ml_pipeline/guard.py) around the 7 readouts and the analysis, through
  src/gdn_gate/ml7/test9b/test9b_guard.py, which reuses fresh/fresh_guard.py unchanged (constants FILE,
  BLOCK, LABELS) and adds: the plan's labels and arguments, each run's meta.json provenance, this block's sha256 lines,
  and test touches exactly 0. The touch is recorded only after every readout and the analysis receipt exist.
- Frozen code. Every file below is frozen by sha256 now, before the first readout; test9b_guard.py preflight refuses
  if any differs, and the runner locks the instrument again at the first readout (results/ml7_9b/
  aime2025/code.sha256), refusing any change during the read. Paths relative to the worktree:

    b7c477c2d6a168a586672767c9fc79b92cc10a3e85334ef5f93bce72b660698b  src/gdn_gate/ml7/readout_vllm.py
    fd43756063484b043d79618656924faf8952706d86b836d2a3a69835dae2e1bf  src/gdn_gate/ml7/train_grpo_vllm.py
    efb694059e39f4eab8f241e077519c921780022055a0d8c353ea3b3d3fc29e6a  src/gdn_gate/vllm/gate_vllm.py
    f64d634ea7820d2a034a0f2480744983dcbbbbe8f1852525489cb91d26d1a1d1  src/gdn_gate/broad/rl_common.py
    a4303e9a39474466f8aabece3d5ea8bca556201c14f2c7c2215a4ce3a740bbe3  src/gdn_gate/broad/arms.py
    4d567ac5bc2dc95093bf39d9c27a0cef55a8b55ca59a1a4e32887b93f1c2e927  src/gdn_gate/broad/packing.py
    0d1f341bb132809622bcd0621dc058bd6b91bf60c4b824071e503ed6a92500f6  src/gdn_gate/ml7/step14_analysis.py
    c30c549d3cc4ed96db9acb32739bddb61fe6ca5ee77a2eda503a305930f291ac  src/gdn_gate/ml7/ctrlpm_analysis.py
    13308a3ef6c8dd151ce6abaa6dcf4ec29ac65c415f2c78a818138d30b5f5ba89  src/gdn_gate/ml7/fresh/fresh_analysis.py
    7413847b6b2eebb8265f3d6752dd7a051c1d0071d1bd065c3a2dcf512898f68a  src/gdn_gate/ml7/test9b/test9b_analysis.py
    ce5186cc96671f4d6f1f18b3afc41d3b78c5b2d6d9e8a7e02ea682510eb7fe6e  ml_pipeline/guard.py
    1cf28594a693e26681b31d1b6f832aed854e6b0bce679b0e98c7935c945954ec  src/gdn_gate/ml7/fresh/fresh_guard.py
    53a00bec8ca69468216196b3e484bd81b0bba84f597e3c8e0bb0c5853312ce84  src/gdn_gate/ml7/test9b/test9b_guard.py
    46063e64cd6664fecd8faaaf090c8bd474cdcd6f69effff2007aeb8649bef0fe  src/gdn_gate/ml7/test9b/build_aime2025.py
    65f28ebd8afc4a074a8179965b916d47c908b10bc2c124e771eada7b3e20eadf  src/gdn_gate/ml7/test9b/ckpt_expect_9b.json
    160c7859d700543632cc01d77eaf4f51b552a8e6f452c1409aaca76d2cf734d5  src/gdn_gate/ml7/test9b/test9b_spark.sh
    2ddb1e0f26ca94b382f0ff9b04f95252b07c0fbfda633f819dc92e07263cc0fd  src/gdn_gate/ml7/test9b/t9b_remote.sh
    c4dce57a7f477bc9f964b8a1b301dc9c4689515b78109dba8e9a83d9ec95adaf  ml_pipeline_9b/model_Qwen3.5-9B.sha256
    f65eb3caa80e3b9f09117375b4670c59f6e2571872a7a0b17ce3199f6fa7ee62  ml_pipeline_9b/aime2025_test/split_manifest.json

## Wall time (estimate, 2026-10-16)

- Per readout: 240 sequences (30 x 8). Tokens per sequence from the 4B fresh readouts restricted to AIME 2025 (phase 1
  untouched 9,315, ctrl 9,070, wide1 7,957; phase 2 7,385 / 7,145 / 6,214: nearly every trace runs to the 8,192 cap),
  scaled by the 9B/4B validation token ratio (0.92 untouched to 1.16 ctrl), phase 1 capped at 9,472. Rates from the
  9B GB10 validation readouts: phase 1 539 / 568 / 507 tokens/s, phase 2 399 / 430 / 435 (untouched / ctrl / wide1).
  At gpu_mem 0.35 the GB10 KV cache holds 590,480 tokens (61 sequences at full length, from the 9B validation log), so
  long AIME traces run at lower concurrency than the validation slice; the central case takes 0.75 of the validation
  rates, the range 1.0 to 0.6. Staging and checks add about 5 minutes per readout.
- Hours per readout, central (range): untouched 3.3 (2.3-4.4), ctrl 3.1 (2.1-4.1), wide1 2.9 (2.0-4.0).
- Wall on two units: gb10a 9.3 h (6.4-12.3), gb10b 12.0 h (8.2-16.0); about 12 h central (8-16 h), 21 Spark-hours.
  Without the untouched readout (not possible with the frozen analysis) the units would balance at about 9 h.
- Step 14 final test: fresh_rows_read_by_all_models=1.0000 (touch 1, 2026-10-16)
