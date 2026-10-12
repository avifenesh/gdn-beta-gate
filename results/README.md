# Results and receipts

License: CC-BY-4.0 (LICENSES/CC-BY-4.0.txt). Every file here was written by the code in `src/` during the runs the
paper reports, except the derived files marked below. Paths inside the files are placeholders: `<repo>` (this
repository), `<runs>` (the run root, `WG_RUNS`), `<hf-cache>` (the Hugging Face cache), `<models>`, `<work>`,
`<cache>`, `<venv>` (lab directories), `<host>` (a redacted host name), `<price>` (a redacted rental price).

| Directory | Contents | Paper |
|---|---|---|
| `ml7/` | Qwen3.5-2B record: answer and training cap receipts, validation reads (`step12*`), test read 1 (`step14`), test read 2 for seeds 3 to 5 (`step14_ext`), readout parity fixtures for the CPU tests (`readout_parity`) | Results (Qwen3.5-2B, range) |
| `ml7_4b/` | Qwen3.5-4B record: caps, cap pilots, validation (`step12_box`, `step12_ctrlpm`), test read 1 (`step14`), test read 2 with LoRA-r20 (`step14_ctrlpm`) | Results (Qwen3.5-4B, range), Table 1 |
| `fresh/` | fresh held-out set (AIME 2024/2025/2026, AMC 2023): 33 per-item readouts (14 at 4B, 19 at 2B), both analyses (`analysis_fresh_4b.json`, `analysis_fresh_2b.json`), `final.json` (the single guarded read), `fresh_table.md`, the code lock before and after the amendment, two failed self-test records | Results (fresh set) |
| `mech/` | mechanism, Qwen3.5-4B, GSM8K validation slice: G1 ablation readouts (`ablate/`), G2 dose readouts (`dose/`, inputs in `dose_receipt.json`; G1 hybrid inputs in `hybrid_receipt.json`), G3 teacher-forced beta statistics (`gate_stats/`, summary in `gate_stats_summary/`), `g1g2_summary.json`, text statistics, item-matched accuracy, training dynamics, gate weights, CPU probe checks | Mechanism |
| `run_configs/` | the full argument set of every paper training run (`meta.json` / `result.json`), with its startup parity self-test | Protocol, Implementation details, appendix (SFT and forgetting) |
| `run_logs/` | training logs of the RL runs, validation logs of the chat SFT runs, the Olmo MMLU-Pro cloze results; read by `paper/figs/derive.py`. Also the frozen `meta.json` of three 4B runs (warm-up, Gate(0,1) seed 0, LoRA-r20 seed 0) and the launch logs of the forgetting server (`retrofit/box/main*.log`), which the implementation appendix rows of `paper/CLAIMS.md` read | Results (Qwen3.5-4B), appendices (SFT and forgetting, Olmo-Hybrid-7B) |
| `rescore/`, `w1main/`, `vllm/step6_s0/` | HumanEval and battery scores of the forgetting and chat SFT adapters, item level; the Gate(0,1) comparison | Appendix (SFT and forgetting) |
| `unsloth/` | the check that the Unsloth SFT stack matches the plain trainer (losses and gradients within calibrated bounds, red arms) | Appendix (SFT and forgetting) |
| `olmo/` | Olmo-Hybrid-7B lesions: per-chunk NLL sums (`arms.jsonl`) and summary, CPU/GPU patch checks, vLLM battery (`vllm2/battery`), paired RULER receipts, beta usage | Appendix (Olmo-Hybrid-7B) |
| `vcount/` | Qwen3.5 chat-SFT counting pointer at 32k | Appendix (Olmo-Hybrid-7B) |
| `ENVIRONMENT.json` | derived: env blocks of every released readout (`scripts/collect_env.py`) | Protocol |
| `SCRUB_LEDGER.json` | derived: for every released file, the sha256 of the lab original and of the published copy | |

Per step directory: `analysis_*.json` (the analysis receipt; its `receipt.argv` is the exact command),
`readout_*.json.gz` (per-item scored rows, one per item and sample and budget; no generated text), `final.json`
(read summary with the hashes of the lab files), `code.sha256` (the code lock of that read), `plan.tsv` (the
readout plan), `env.txt` (GPU list and code lock at launch), `steps.log`, `*.parity_fail.json` (a failed startup
parity self-test kept as a record), `DONE`.

Derived, content-free files: `olmo/ruler_items/` and `vcount/ruler_items/` keep only `doc_id`, `doc_hash` and the
score of each RULER item, in the layout `src/gdn_gate/olmo/ruler_paired.py` reads; the model outputs are not
released. `scripts/rerun_ruler_pairing.py` reproduces the four paired receipts from them.

The readouts in `mech/ablate/` and `mech/dose/` are plain JSON (no generated text), read directly by
`src/gdn_gate/mech/g1g2_summary.py`. `mech/example_val_trace.json`, the lab's one paired example of generated text,
is not released.

Not here: the generation traces (`*.traces.jsonl.gz`, 229 files, 1.40 GB, one per readout of the RL records, the
fresh set and the G1 and G2 mechanism readouts) that hold the sampled reasoning tokens and answer texts. They are a
separate archive on Zenodo, [10.5281/zenodo.23340255](https://doi.org/10.5281/zenodo.23340255), that unpacks into this directory next to the readouts.
`TRACES_LEDGER.json` lists each file with its trace count, its readout and the sha256 of the file and of its
uncompressed content; the publication scrub changed none of them. `scripts/rerun_traces.py` reruns the scripts that
read them. The four small trace files in `ml7/readout_parity/` are test fixtures.

## Publication scrub

The lab files carried absolute paths, internal project names and host names. Before release they were rewritten by
fixed string replacements (paths to the placeholders above, names to their public equivalents, host values to
`<host>`); numbers and structure are untouched, and JSON files stay valid. Hash values inside the receipts
(`code.sha256`, readout `env.sha256`, `final.json`) refer to the lab files. `SCRUB_LEDGER.json` lists, per released
file, `sha256_original` (lab file) and `sha256_released` (the published content; uncompressed for `.gz` files), so
a hash recorded in a receipt can be followed to the file published here.
