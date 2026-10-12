# Changelog

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
