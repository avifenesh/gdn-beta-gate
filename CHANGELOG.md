# Changelog

## v1.2.0 (2026-10-16)

The Qwen3.5-9B RL record. Code, receipts and analyses of v1.1.1 are unchanged; the new record runs the same trainer,
readout and frozen analyses.

Added

- The 9B record: the 4B recipe on Qwen3.5-9B (revision c2022362) on two DGX Spark (GB10) units, one 40-step warm-up
  and LoRA and Gate(0,1) x seeds 0 to 2 for 150 steps (micro-row 4,096, a memory setting). Validation reads on the
  GSM8K validation slice (`results/ml7_9b/step9_*.json`): Gate(0,1) reasons 502 tokens against 1,825 for LoRA (seed
  means, 63 to 79% shorter per seed) at the same accuracy at 8,192 (0.976).
- The single read of AIME 2025 at 9B (30 problems, 8 samples), pre-registered in `ml_pipeline_9b/PIPELINE.md` with
  H5-9B and H6-9B under the 4B fresh-set rule: both supported. Gate(0,1) - LoRA area +0.103 [+0.061, +0.147], length
  -659 [-1,036, -323] tokens, accuracy at 8,192 0.482 against 0.275. Receipts in `results/ml7_9b/aime2025/`
  (7 per-item readouts, the analysis, `final.json`, the code lock); the guard state in `ml_pipeline_9b/aime2025_test/`.
- `src/gdn_gate/ml7/test9b/`: the AIME 2025 file cut (`build_aime2025.py`), the analysis wrapper over the frozen
  `fresh_analysis.py` and `step14_analysis.py` (`test9b_analysis.py`), the guard (`test9b_guard.py`), the GB10 runner
  (`test9b_spark.sh`, `t9b_remote.sh`), the checkpoint sha256 list, and 32 CPU checks (`test_test9b_cpu.py`, tier B).
- Run configs and logs of the seven 9B runs (`results/run_configs/9B_rl/`, `results/run_logs/ml7_9b/grpo/`).
- The 7 adapters on the Hugging Face Hub as `Avifenesh/gdn-beta-gate-qwen3.5-9b-grpo` (`warmup/seed0`,
  `lora/seed0` to `seed2`, `gate01/seed0` to `seed2`), the 9B GRPO record in `scripts/adapters.py`, and the model card
  `hf/qwen3.5-9b-grpo.md`. 72 adapters in six repositories.
- The 14 trace files of the 9B record in version 1.1.0 of the Zenodo trace record (`gdn-beta-gate-traces-9b.zip`);
  `results/TRACES_LEDGER.json` lists 243 files.
- The paper's 9B numbers: rows in `paper/CLAIMS.md` (1181 rows, all checked), `paper/figs/derive_9b.py` and its output
  `paper/data/r9b.json`, the figure `paper/figs/fig_9b.pdf`, and the GB10 hours in `paper/data/compute_hours.json`.

Changed

- `scripts/rebuild_data.py` also writes `test_aime2025.jsonl` and checks it (6 files). `scripts/rerun_analysis.py`
  reruns the 9B AIME 2025 analysis (16 receipts). `results/ENVIRONMENT.json` lists the 9B readouts.
- README.md, REPRODUCE.md (section 2.12), docs/ADAPTERS.md, docs/COMPUTE.md (about 99 GB10-hours for the 9B record),
  docs/PREREGISTRATION.md, results/README.md and tests/README.md describe the 9B record. The other five model cards
  say the RL results cover 2B, 4B and 9B.

## v1.1.1 (2026-10-13)

- The generation traces of every readout are a separate archive on Zenodo ([10.5281/zenodo.23340255](https://doi.org/10.5281/zenodo.23340255), CC-BY-NC-SA-4.0, 229 files,
  290,306 traces). `results/TRACES_LEDGER.json` lists them with their sha256; `scripts/rerun_traces.py` reruns
  `text_stats.py`, `answer_cap.py`, `truncate_curve.py`, `build_inputs.py` and `paper/figs/example_traces.py` from them
  and compares with the released files.
- Dev dependency: pytest 9.0.3 (GHSA-6w46-j5rx-g56g).

## v1.1.0 (2026-10-12)

Corrections and additions for the paper draft now titled "Beta Gate: Shorter RL Reasoning in Qwen3.5 Without Negative
Eigenvalues" (Avi Fenesh, Tiyuvta.ai). The training, readout and analysis code and the receipts of v1.0.0 are
unchanged.

Added

- The 6 Qwen3.5-9B-Base forgetting adapters (LoRA and Gate(0,2), seeds 0 to 2) on the Hugging Face Hub as
  `Avifenesh/gdn-beta-gate-qwen3.5-9b-base-forgetting` (revision 8be3d70f, docs/ADAPTERS.md), with their model card
  (`hf/qwen3.5-9b-base-forgetting.md`), their run configs (`results/run_configs/forgetting/main9_*`) and the 9B record
  in `scripts/adapters.py`. 65 adapters in five repositories.
- Ledger rows and data of the paper's implementation appendix (I1 to I51): the frozen `meta.json` of three 4B runs and
  the launch logs of the forgetting server under `results/run_logs/`, read by those rows.
- The paired example traces of the implementation appendix: `paper/data/example_traces.txt` and the script that
  decoded it, `paper/figs/example_traces.py`. Also `paper/figs/compute_hours.py`, the script behind
  `paper/data/compute_hours.json`.
- New ledger rows from the paper revision: P30a to P30c (the forgetting rental) and W35b (the learning-rate pilots of
  the forgetting recipe on the RTX 5090 Laptop GPU); W30 and W31 now check the corrected compute totals.

Corrected

- Compute: the rented total is about 293 GPU-hours, not 210. The 2026-09-29 rental that trained the 4B and 9B
  forgetting runs (8 x RTX PRO 6000, 10.4 h, 83 GPU-hours) was missing from docs/COMPUTE.md;
  `paper/data/compute_hours.json` now counts it and the forgetting server's training logs (logged RTX PRO 6000 time
  233 h, was 171 h).
- Hardware statements in README.md, REPRODUCE.md, docs/COMPUTE.md and the model cards: chat SFT seeds 1 to 5 trained
  on RTX PRO 6000 (only seed 0 on GB10); the 4B forgetting runs of seeds 0 to 2 and all 9B forgetting runs trained on
  RTX PRO 6000 (LoRA seeds 4 and 5 on GB10); the 9B runs were scored on GB10 except LoRA seed 1; the 2B warm-up and
  pilots trained on GB10; the per-run times of the SFT and forgetting runs.
- The claims ledger had 12 rows the checker never checked: a merged row (F11 with V245) and rows whose text cell held
  `|x|`, which split the row. `paper/figs/check_claims.py` silently skipped any row that did not parse; it now fails
  on a malformed row, and reads gzipped readouts (`readout_*.json.gz`) that ledger rows cite. The ledger has 1044
  rows, all checked.
- Paper section, table and figure references in REPRODUCE.md and results/README.md follow the revised paper.
- Figure axis labels say "thinking budget".

Changed

- Paper title, in README.md, CITATION.cff (now with affiliation), the BibTeX blocks and the five model cards. Prose
  uses "thinking budget", and "anytime accuracy (area under the accuracy-budget curve)" for the area metric.

## v1.0.0 (2026-10-12)

First public release, matching the paper draft of October 2026, "Retrofitting a Gate on the Delta-Rule Write
Strength Shortens RL-Trained Reasoning in Qwen3.5".

- Training, readout and analysis code of the Qwen3.5-2B and Qwen3.5-4B RL records, the chat SFT and forgetting
  recipes and the Olmo-Hybrid-7B lesions, with the pre-registration records and the code-lock amendment ledger.
- The fresh held-out set (AIME 2024, AIME 2025, AIME 2026, AMC 2023; 130 items): pre-registration, guard state, id
  list and byte-exact rebuild from the public sources, the 33 per-item readouts and both analyses (H5 and H6 supported
  at 4B, the 2B replication not supported), the effect_visible amendment with its CPU test.
- The mechanism tests on the GSM8K validation slice: gate ablation (G1), gate dose (G2), teacher-forced beta
  statistics (G3), text and training-dynamics analyses, with their code and receipts.
- All analysis receipts and per-item readouts behind the paper's numbers, the claims ledger (976 rows) and checker,
  the analysis and mechanism reruns, figure scripts, the data manifest and rebuild script.
- The 59 trained adapters on the Hugging Face Hub (four repositories, docs/ADAPTERS.md), with the export and
  conversion tool.
