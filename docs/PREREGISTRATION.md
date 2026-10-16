# Pre-registration records

The RL comparisons followed a written protocol: each hypothesis was named in a lab record before the runs that test
it, every selection used validation data, the analysis code was frozen by sha256 before each read of the test set,
and a guard counted the reads. The records are released as they were, with the publication edits listed at the end.

## Files

| File | What it is |
|---|---|
| `ml_pipeline/PIPELINE.md` | The Qwen3.5-2B record (created 2026-10-08): data inspection, problem contract (metric, decision rule, contrasts), split, validation rules for the training and answer caps, the warm-up design change, test read 1, the seeds 3 to 5 extension and test read 2 |
| `ml_pipeline_4b/PIPELINE.md` | The Qwen3.5-4B record (created 2026-10-11): H1 and H2 named from the 2B read 1 before any 4B run, calibration, cap rules rerun at 4B, test read 1, the LoRA-r20 extension with H3 and H4, test read 2; then the fresh held-out set ("Fresh test set"): its pre-registration with H5 and H6, three amendments, the read (touch 1), the H5/H6 result line and two append-only corrections |
| `ml_pipeline_4b/fresh_test/` | The fresh set's own guard state (fingerprint of the frame, test touches 1) and its manifest (`test_fresh.jsonl` sha256) |
| `ml_pipeline_9b/PIPELINE.md` | The Qwen3.5-9B record (v1.2.0; created 2026-10-15): the owner's choice of the read, the AIME 2025 file cut from the fresh set, the six trained models and their checkpoint sha256, the readout, then the pre-registration of the AIME 2025 read ("AIME 2025 test read at 9B") with H5-9B and H6-9B, the frozen-code sha256 list, a wall-time estimate, and the guard's result line after the read (touch 1) |
| `ml_pipeline_9b/aime2025_test/` | The AIME 2025 file's own guard state (fingerprint of the frame, test touches 1) and its manifest (`test_aime2025.jsonl` sha256) |
| `ml_pipeline_9b/model_Qwen3.5-9B.sha256` | sha256 of the 13 model files of Qwen3.5-9B (revision c2022362) that every 9B readout checked on its unit |
| `ml_pipeline*/.guard_state.json` | The guard's state: fingerprint of the test frame and the number of test reads (2 per record) |
| `ml_pipeline_4b/COPIED_FROM_2B.sha256` | sha256 of the 2B artifacts the 4B record reused byte for byte (split, grader check, data profile) |
| `src/gdn_gate/ml7/amendments/ledger.json` and `*.diff` | Every approved change to a frozen instrument file: old and new sha256, the diff and its sha256, the approval time and what it changes; one withdrawn amendment |
| `results/**/code.sha256` | The code lock of each read: sha256 of the readout, trainer, plugin and analysis files at read time |
| `results/ml7/step14/code_lock_amendment.txt` | The 2B read-1 amendment record (parity limit 2.5 for one checkpoint, before its readout ran) |

## Timestamps

The times inside the records are the author's own entries in Israel time (UTC+3), written as each step happened;
gate approvals and the hypothesis blocks carry their time. Key entries:

| When (Israel time) | Record | Entry |
|---|---|---|
| 2026-10-08 | 2B | Gate A: contract, model, 500-item validation slice, primary metric (area under accuracy vs log2 budget, 256 to 8k) |
| 2026-10-08 15:47 | 2B | Gate B: recipe, candidates, ruled-out list, metric |
| 2026-10-08 21:06, 22:32 | 2B | training-cap pilot rule; answer-cap rule |
| 2026-10-09 10:00 | 2B | design change: shared format warm-up |
| 2026-10-11 ~10:50 | 2B | amendment: parity limit 2.5 for one checkpoint (before its readout) |
| 2026-10-11 (after 2B read 1) | 4B | H1, H2 named, before any 4B run; Gate A at 13:00 |
| 2026-10-11 13:00 | 2B | seeds 3 to 5 extension and its two named contrasts, before those runs |
| 2026-10-11 ~14:55 | 4B | parity probe 0.05 for every 4B run, before any 4B training |
| 2026-10-12 00:30 | 4B | H3, H4 written, before any LoRA-r20 run, readout or new test read |
| 2026-10-12, before the rental | 4B | fresh held-out set pre-registered: data frozen (manifest `created_utc` 01:51 UTC), guard touches 0, H5 and H6 named |
| 2026-10-12 05:58, 06:00 | 4B | amendments: readout packing (one readout per GPU), retries 4; no readout argument changed |
| 2026-10-12 09:40 (written; approval at 09:11) | 4B | amendment: visibility bar 4.0 for two 4B fresh readouts' self-test (ledger entry `readout_effect_visible_fresh.diff`) |
| 2026-10-12 (touch 1) | 4B | the single read of the fresh set; H5 and H6 supported at 4B, the 2B replication not supported |
| 2026-10-14, 2026-10-15 | 9B | warm-up and the six arm runs on the two GB10 units, validation reads only (no test set) |
| 2026-10-15 ~23:40 to ~23:55 | 9B | owner direction: read harder problems, AIME 2025 only, with the six trained arms |
| 2026-10-16 ~00:05 | 9B | AIME 2025 read pre-registered (file cut on 2026-10-15, guard touches 0): H5-9B and H6-9B named, code frozen by sha256, before any 9B readout of the file |
| 2026-10-16 00:58 to 12:51 (touch 1) | 9B | the single read of AIME 2025 (7 readouts, then the analysis); H5-9B and H6-9B supported |

The two correction lines at the end of the fresh-set block were added after the read, append-only, by the
coordinator: the first fixes the approval time of the 09:40 amendment, the second states that eight finished 4B
fresh readouts (each with its own accuracy summary) existed at the time of that approval, that no contrast or
hypothesis test had been computed, and that the rule covers any 4B fresh readout failing on visibility alone.

What these timestamps do and do not establish: the records were kept as files in the lab worktree, which was not
under version control, and they were not deposited with a public registry or a timestamping service at the time.
The ordering therefore rests on the author's entries, on the file sha256 values recorded inside the receipts (code
locks, readout env blocks, the amendment ledger), and on the receipts' own creation times (`created_utc` in each
analysis receipt, the run configs and logs). A reader can check that every code lock in `results/**/code.sha256`
was recorded before the readouts that list the same hashes, and that the analysis receipts of read 2 reproduce read 1
on the original arms (`ctrlpm/touch1_check/equal_on_original_arms` = true in the 4B read-2 receipts).

## Publication edits to the records

Beyond the fixed scrub rules (paths, internal names, host names), these lines differ from the lab files:

| File | Line | Edit |
|---|---|---|
| `ml_pipeline/PIPELINE.md` | 1 | internal project name dropped from the title |
| `ml_pipeline/PIPELINE.md` | 37, 129, 170, 176, 191, 272 | the lab workstation's internal name replaced by "workstation" |
| `ml_pipeline/PIPELINE.md` | 130, 131 | internal names in "the gate plugin" and in the parity receipt's file name |
| `ml_pipeline/PIPELINE.md` | 198 | the validation salt string replaced by a reference to `VAL_SALT` in `02_prep.py` (the scrub would have changed its value) |
| `ml_pipeline_4b/PIPELINE.md` | 1 | internal project name dropped from the title |
| `ml_pipeline_4b/PIPELINE.md` | 42, 317 | the lab workstation's internal name replaced by "workstation" |
| `ml_pipeline_4b/PIPELINE.md` | 131 | the list of rental providers compared replaced by "the GPU rental providers compared" |
| `ml_pipeline_4b/PIPELINE.md` | 132, 133, 140, 194, 243, 244, 255, 303 | rental prices and budget caps replaced by `<price>` |
| `ml_pipeline_9b/PIPELINE.md` | 1 | internal project name dropped from the title |
| `ml_pipeline_9b/PIPELINE.md` | 4, 38 | the file name of the lab's planning note (not released) replaced by "the lab's planning note"; line 4 also says where its 9B receipts are |
| `ml_pipeline_9b/PIPELINE.md` | 42 | the lab workstation's internal name replaced by "workstation" |

The fresh-set block (lines 251 to 317 of the 4B record) and the appended last line of the 2B record are otherwise the
lab text after the fixed scrub rules; the lane's internal name appears as `wg`, the scrub rule's replacement.

The released copies differ from the lab files only by these replacements: internal project and lane names, absolute
paths (replaced by `<repo>`, `<runs>`, `<hf-cache>` and similar placeholders), host names, the name of the
workstation, and two cloud-provider names. The validation split salt in `ml_pipeline/02_prep.py` is kept byte-exact
in hex form; `data/splits/*.ids` list the resulting item ids, and `scripts/rebuild_data.py` reproduces the frozen
files' sha256. `results/SCRUB_LEDGER.json` gives each released file's original sha256 and its published sha256, so
the author can show the unedited originals on request.
