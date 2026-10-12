# Changelog

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
