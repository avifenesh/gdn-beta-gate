# Beta Gate: Shorter RL Reasoning in Qwen3.5 Without Negative Eigenvalues

Code, receipts and checks for the paper "Beta Gate: Shorter RL Reasoning in Qwen3.5 Without Negative Eigenvalues" (Avi
Fenesh, Tiyuvta.ai, 2026; the arXiv identifier is added to this README and `CITATION.cff` when it exists). This repository is a frozen research artifact: it holds the exact training, readout and analysis code the
paper ran, the pre-registration records, every analysis receipt and per-item readout the paper's numbers come from, a
claims checker that ties each number in the paper to a JSON field, and the figure scripts. It does not hold model
weights (the trained adapters are on the Hugging Face Hub, see [docs/ADAPTERS.md](docs/ADAPTERS.md)), dataset text
(rebuilt from the public sources by checksum), the raw generation traces (a separate archive on Zenodo, [10.5281/zenodo.23340255](https://doi.org/10.5281/zenodo.23340255)), or the lab's
GPU scheduling and server provisioning scripts.

Write-ups: [Hugging Face blog](https://huggingface.co/blog/Avifenesh/beta-gate-qwen35) and [tiyuvta.ai](https://tiyuvta.ai/research/beta-gate-length-rl/).

The method in one line: a zero-initialized linear map `W` per Gated DeltaNet layer moves the write strength `beta`
toward its floor or its ceiling, `tau = tanh(W x)`; ceiling 1 is Gate(0,1), ceiling 2 opens the negative-eigenvalue
range and is Gate(0,2). We call it the β gate. It is trained with LoRA under GRPO with a correctness-gated length
penalty.

## Headline results

All RL numbers: test sets read under the locked protocol with vendor sampling (temperature 1.0, top-p 0.95, top-k 20,
presence penalty 1.5), thinking budgets of 256 to 8,192 tokens, mean of the seeds with the per-seed range. Area =
anytime accuracy (area under the accuracy-budget curve), normalized over log2 budgets. Length = mean reasoning tokens
at the 8,192 budget. Every 2B and 4B RL test readout and every 4B RL run used one NVIDIA RTX PRO 6000 Blackwell
Server Edition (96 GB); the 2B warm-up, pilots and runs of seeds 0 to 2 trained on NVIDIA GB10, and every 9B run and
readout used one NVIDIA DGX Spark (GB10). Software: vLLM 0.31.0, PyTorch 2.13.0
(cu130), transformers 5.17.0 (trainer forward: PyTorch 2.14.0, flash-linear-attention 0.5.2).

| Claim (paper part) | Model | Arm | Metric | Seeds | Mean (per-seed range) |
|---|---|---|---|---|---|
| Gate shortens reasoning (Results, 4B) | Qwen3.5-4B | LoRA | MATH-500 area | 3 | 0.786 (0.773 to 0.794) |
| | | LoRA-r20 | MATH-500 area | 3 | 0.792 (0.766 to 0.812) |
| | | Gate(0,1) | MATH-500 area | 3 | 0.852 (0.851 to 0.853) |
| | | Gate(0,2) | MATH-500 area | 3 | 0.849 (0.839 to 0.856) |
| | | LoRA / Gate(0,1) | MATH-500 length | 3 | 4,223 (4,043 to 4,416) / 2,202 (1,947 to 2,461) |
| | | LoRA / LoRA-r20 / Gate(0,1) | GSM8K length | 3 | 1,666 / 1,296 / 297 (218 to 381) |
| Holds on a fresh held-out set (Results, fresh set) | Qwen3.5-4B | Gate(0,1) - LoRA | fresh area (H5) | 3 | +0.110 [+0.089, +0.132], supported |
| | | Gate(0,1) - LoRA | fresh length (H6) | 3 | -1,329 [-1,585, -1,081] tokens, supported |
| | Qwen3.5-2B (secondary) | Gate(0,1) - LoRA | fresh area / length | 6 | +0.012 / -126 tokens, not supported (seed signs split) |
| Effect at 2B (Results, 2B) | Qwen3.5-2B | Gate(0,2) - LoRA | MATH-500 area | 3 + 3 fresh | +0.027 (read 1), +0.025 (read 2) |
| Holds at 9B on a hard test set (Results, 9B) | Qwen3.5-9B | Gate(0,1) - LoRA | AIME 2025 area (H5-9B) | 3 | +0.103 [+0.061, +0.147], supported |
| | | Gate(0,1) - LoRA | AIME 2025 length (H6-9B) | 3 | -659 [-1,036, -323] tokens, supported |
| | | LoRA / Gate(0,1) | AIME 2025 accuracy at 8,192 | 3 | 0.275 / 0.482 (untouched 0.225) |
| Shortens at 9B at the same accuracy (Results, 9B, validation) | Qwen3.5-9B | LoRA / Gate(0,1) | GSM8K validation length | 3 | 1,825 (1,769 to 1,888) / 502 (375 to 698); accuracy at 8,192 0.976 / 0.976 |
| Range adds nothing in RL (Results, range) | 2B, 4B | Gate(0,2) - Gate(0,1) | MATH-500 area | 6, 3 | +0.006, +0.003, -0.002 (no effect) |
| The gate carries the shortening (Mechanism, G1) | Qwen3.5-4B | Gate(0,1): trained / gate removed / gate only | GSM8K validation length | 3 | 307 / 665 / 970 (LoRA 1,600, warm-up 2,916) |
| Dose response (Mechanism, G2) | Qwen3.5-4B | Gate(0,1), W scaled 0 to 2 | GSM8K validation length | 3 | falls monotonically in every seed; area 0.949 to 0.963 |
| Gate moves beta more than LoRA does (Mechanism, G3) | Qwen3.5-4B | gate vs LoRA arms | mean abs change of beta, step 150 | 3 | 0.018 to 0.021 (Gate(0,1)), 0.038 to 0.041 (Gate(0,2)) |
| Range costs held-out loss in SFT (Appendix) | Qwen3.5-4B-Base | Gate(0,2) - Gate(0,1) | held-out NLL, nats/token | 6 | +0.0032 (6 of 6 seeds) |
| Gate reduces forgetting (Appendix) | Qwen3.5-4B-Base | LoRA / Gate(0,2) / Gate(0,1) | HumanEval pass@1 | 3 | 0.152 / 0.264 / 0.409 |
| Pretrained range needed for counting (Appendix) | Olmo-Hybrid-7B | clamp beta at 1 | RULER CWE, 4k / 8k / 16k | 500 items | -3.8 / -6.3 / -6.0 points |

The fresh held-out set is 130 competition problems (AIME 2024, AIME 2025, AIME 2026, AMC 2023) that no step of the
records saw, pre-registered with hypotheses H5 and H6 before any read and read once (8 samples per item). The
mechanism tests ran on the GSM8K validation slice (500 items, 1 sample), never on a test set.

The 9B record (new in v1.2.0) runs the 4B recipe on Qwen3.5-9B on two DGX Spark (GB10) units: one 40-step warm-up,
then LoRA and Gate(0,1) for 150 steps, seeds 0 to 2. No Gate(0,2) or LoRA-r20 arm was trained at 9B. It read the
GSM8K validation slice (500 items, 2 samples) and one test set once: AIME 2025, the 30 AIME 2025 problems of the fresh
set (8 samples per problem), pre-registered with H5-9B and H6-9B in `ml_pipeline_9b/PIPELINE.md`. MATH-500, the
GSM8K test set and the rest of the fresh set stay unread at 9B. On GSM8K validation, where 9B is near its ceiling,
the gate cuts reasoning by 63 to 79% per seed at the same accuracy. On AIME 2025 the gated arm is more accurate than
LoRA at every budget in every seed and reasons shorter. Receipts: `results/ml7_9b/`.

Every number in the paper, with its receipt path and tolerance, is in [paper/CLAIMS.md](paper/CLAIMS.md).

## Install

CPU checks (claims, analysis rerun, figures, data rebuild), pinned by `uv.lock`:

    uv sync --locked

GPU environments, one per stack, pinned by exact versions in `env/` (Python 3.12.13, Linux, CUDA 13.0 wheels):

    uv venv -p 3.12.13 .venv-vllm && uv pip install -p .venv-vllm -r env/vllm.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130         # GRPO training, readouts, vLLM batteries
    uv pip install --target "$WG_ML7_PYLIB" --no-deps -r env/fla-pylib.txt   # trainer kernels (flash-linear-attention)
    uv venv -p 3.12.13 .venv-train && uv pip install -p .venv-train -r env/train.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130         # SFT, forgetting recipe, Olmo HF lesions
    # env/olmo.txt = env/train.txt plus lm-eval 0.4.13, for src/gdn_gate/olmo/battery.py

Tested on: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, compute capability 12.0), NVIDIA DGX Spark (GB10,
aarch64) and an RTX 5090 Laptop GPU (24 GB), CUDA 13.0 runtime wheels. The gate plugin for vLLM checks the hashes of
the vLLM 0.31.0 sources it rewrites and refuses any other version. The driver version of the rented servers was not
recorded; the workstation ran driver 615.71.09.

Paths: `WG_RUNS` (default `./runs`, git-ignored) holds data files, checkpoints and run logs; `WG_HF_CACHE` (default
`~/.cache/huggingface/hub`) is the Hugging Face cache the scripts read model snapshots from, by pinned revision.

## Quickstart (CPU, about 10 minutes)

    uv run python paper/figs/check_claims.py        # 1181 rows checked, 0 failed
    uv run python scripts/rerun_analysis.py         # reruns 16 frozen analyses (incl. the fresh set, 9B AIME 2025) from the released readouts: all OK
    uv run python scripts/rerun_mech.py             # reruns 4 mechanism summaries from released data: all OK
    SOURCE_DATE_EPOCH=1760227200 uv run python paper/figs/make_figs.py   # rewrites paper/figs/*.pdf, byte for byte
    uv run python scripts/rebuild_data.py           # downloads GSM8K, MATH-500, AIME and AMC at pinned revisions; 6 x OK
    uv run pytest tests/                            # the CPU checks above as tests: 8 passed (tests/README.md)

`rerun_analysis.py` decompresses the per-item readouts (`results/**/readout_*.json.gz`), runs the frozen
`step14_analysis.py` / `ctrlpm_analysis.py` (through `fresh_analysis.py` for the fresh set and `test9b_analysis.py`
for the 9B AIME 2025 read) with the command line recorded in each receipt, and compares every key.

## Generation traces

The sampled reasoning tokens and answer texts of every readout (`*.traces.jsonl.gz`, 243 files, 298,986 traces, 1.44
GB) are a separate archive under CC-BY-NC-SA-4.0: [10.5281/zenodo.23340255](https://doi.org/10.5281/zenodo.23340255).
The first archive, `gdn-beta-gate-traces.zip` (229 files, 290,306 traces, 1.40 GB), comes in 34 parts of 40 MiB; join
them and check the result against the record's `SHA256SUMS`. The 9B traces (14 files, 8,680 traces, 37.6 MB) are a
second archive, `gdn-beta-gate-traces-9b.zip`, added in version 1.1.0 of the same record. Both unpack at the repository root,
each file next to its readout; `results/TRACES_LEDGER.json` lists every file with its sha256.

    cat gdn-beta-gate-traces.zip.part* > gdn-beta-gate-traces.zip
    unzip -n gdn-beta-gate-traces.zip                    # at the repository root; -n keeps tracked copies
    unzip -n gdn-beta-gate-traces-9b.zip
    uv run python scripts/rebuild_data.py --skip_fresh
    uv run --with transformers==5.17.0 --with jinja2==3.1.6 python scripts/rerun_traces.py --verify

`scripts/rerun_traces.py` reruns the five scripts that read generated text and compares their output with the
released files: `text_stats.py` (`results/mech/text_stats.json`), `answer_cap.py` (both `answer_cap.json`),
`paper/figs/example_traces.py` (`paper/data/example_traces.txt`), `truncate_curve.py` (the step 9 curves at the chosen
answer caps) and `build_inputs.py` (the G3 teacher-forcing inputs). The last two import `rl_common.py` and need the pins
of `env/train.txt`; a script whose imports are missing is reported as SKIP.

## Reproduce the paper

About 293 GPU-hours on RTX PRO 6000 GPUs: 83 for the 4B and 9B forgetting runs, 173 for the RL records and 37 for the
fresh held-out set and the mechanism readouts, plus owned GB10 and RTX 5090 time ([docs/COMPUTE.md](docs/COMPUTE.md)).
One 4B GRPO run takes about 1.4 h on one RTX PRO 6000; one test readout of one checkpoint on one test set takes 0.8 to
1.4 h.

| Result | Command (details in REPRODUCE.md) | Hardware | Wall time | Expected | Tolerance |
|---|---|---|---|---|---|
| Tables and figures from receipts | `uv run python scripts/rerun_analysis.py`, `scripts/rerun_mech.py` | CPU | minutes | all OK | exact |
| Every number in the paper | `uv run python paper/figs/check_claims.py` | CPU | seconds | 0 failed | per row |
| 4B GRPO arm run | `bash src/gdn_gate/ml7/run_grpo.sh ARM SEED 512 r_box <recipe>` | 1 GPU, 96 GB | ~1.4 h | `train_log.jsonl` | see REPRODUCE.md |
| 4B test readout | `python src/gdn_gate/ml7/readout_vllm.py ...` | 1 GPU, 96 GB | 0.8 to 1.4 h | `readout_*.json` | sampling noise |
| Fresh held-out set | `readout_vllm.py --data test_fresh.jsonl --samples 8 ...`, `fresh_analysis.py` | 1 GPU, 96 GB per readout | ~1.3 h per 4B readout | H5, H6 supported at 4B | decision rule |
| 9B GRPO arm run | `GRPO_MODEL=<9B> MICRO=4096 bash src/gdn_gate/ml7/run_grpo.sh ARM SEED 512 r9b_gb10 <recipe>` | 1 DGX Spark (GB10) | 10.8 to 12.0 h | `train_log.jsonl` | see REPRODUCE.md |
| 9B AIME 2025 read | `readout_vllm.py --data test_aime2025.jsonl --samples 8 ...`, `test9b/test9b_analysis.py` | 1 DGX Spark (GB10) per readout | 2.4 to 4.5 h per readout | H5-9B, H6-9B supported | decision rule |
| Mechanism G1 to G3 | `src/gdn_gate/mech/mech_box.sh g1/g2/g3 ...` | 1 GPU, 96 GB per job | 0.03 to 0.4 h per readout | `results/mech/` | seed range |
| Chat SFT (Appendix) | `python src/gdn_gate/broad/train_sft.py ...` | 1 GB10 or larger | hours | `val_log.jsonl` | seed spread |
| Olmo lesions (Appendix) | `src/gdn_gate/olmo/`, `src/gdn_gate/vllm/vbattery_olmo.py` | 1 GPU, 24 GB+ | hours | `results/olmo/` | sampling noise |

Full claims map, commands, recipes and what differs: [REPRODUCE.md](REPRODUCE.md). Pre-registration records and
their timestamps: [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md). Trained adapters:
[docs/ADAPTERS.md](docs/ADAPTERS.md).

## Names in code and receipts

| Paper | Code and receipts |
|---|---|
| LoRA | `ctrl` |
| LoRA-r20 (parameter-matched control) | `ctrlpm` (trainer `--arm ctrl --rank 20`) |
| Gate(0,1) | `wide1` |
| Gate(0,2) | `wide` |
| shared warm-up | `warmup`, run tag `fmt` |
| Qwen3.5-2B RL record / Qwen3.5-4B RL record | `ml7`, `ml_pipeline/` / `ml7_4b`, `ml_pipeline_4b/` |
| Qwen3.5-9B RL record (warm-up / arms, GB10) | `ml7_9b`, `ml_pipeline_9b/`, run tags `fmt9b_gb10` / `r9b_gb10` |
| 9B AIME 2025 read | `results/ml7_9b/aime2025`, `src/gdn_gate/ml7/test9b`, hypotheses H5-9B and H6-9B |
| validation read / test read | `step12*` / `step14*` |
| 2B read 2 (seeds 3 to 5) / 4B read 2 (LoRA-r20) | `step14_ext` / `step14_ctrlpm` |
| chat SFT / forgetting recipe / Gate(0,1) forgetting runs | `broad/step6` / `src/retrofit` (`main_*`) / `w1main` |
| clamp / halve (Olmo lesions) | `erase` / `half` |
| gate matrix W | `gate_w` |
| fresh held-out set / mechanism (Phase 5) | `results/fresh`, `ml_pipeline_4b/fresh_test` / `results/mech`, `src/gdn_gate/mech` |
| β gate (earlier working name: write gate) | the `WG_` environment variables and the `wg` prefix in receipts |

## Layout

    src/gdn_gate/broad/    gate arms (arms.py), reward and grader (rl_common.py), packing, chat SFT trainer
    src/gdn_gate/ml7/      GRPO trainer with vLLM rollouts, readout, step 12/14 analyses, LoRA-r20 widening, CPU tests
    src/gdn_gate/ml7/fresh/  fresh held-out set: builder, guard, analysis, runner, CPU tests
    src/gdn_gate/ml7/test9b/ 9B AIME 2025 read: file cut, guard, analysis, GB10 runner, CPU tests
    src/gdn_gate/mech/     mechanism: gate ablation and dose inputs, beta statistics, text and training analyses
    src/gdn_gate/vllm/     the gate inside vLLM 0.31.0 (gate_vllm.py), HumanEval and counting scorers
    src/gdn_gate/olmo/     Olmo-Hybrid-7B beta lesions (HF and vLLM), NLL, battery, RULER pairing
    src/gdn_gate/rescore/  HumanEval and battery scorer for the forgetting runs
    src/gdn_gate/w1main/   Gate(0,1) forgetting comparison (McNemar)
    src/retrofit/          GDN patch (patch.py), forgetting-recipe trainer, synthetic tasks, general eval
    ml_pipeline*/          pre-registration records, data split code and manifests, guard
    results/               analysis receipts, per-item readouts (gz), run configs and logs; see results/README.md
    paper/                 CLAIMS.md, check_claims.py, derive.py, figure scripts and figures
    data/                  MANIFEST.json (sources, revisions, checksums), split id lists
    scripts/               CPU tools: rerun analyses, mechanism summaries and trace analyses, rebuild data, collect env,
                           export adapters
    env/                   pinned GPU environments
    docs/                  compute, pre-registration notes, adapters on the Hub
    hf/                    the model cards of the six Hub repositories

## Limits

- Frozen snapshot, not a library. Code paths follow the lab's run layout; scripts take paths from flags and from
  `WG_RUNS` / `WG_HF_CACHE`.
- Released code differs from the hash-locked lab files only by publication edits: internal names and absolute paths
  replaced, path defaults moved to environment variables, the gate tensor renamed to `gate_w`, and the code-lock
  checks taught to read the published hashes. The locked hashes in the receipts refer to the lab files;
  `results/SCRUB_LEDGER.json` maps every lab file hash to its published hash. Two mechanism scripts
  (`hybrid_ckpts.py`, `traj_extra.py`) were written for the release and reproduce outputs the lab computed inline.
- Readouts are sampled. Rerunning a readout reproduces the protocol, not the bytes: vLLM promises identical outputs
  only on the same hardware and version, and the readouts ran two jobs per GPU. The analysis rerun from the released
  readouts is exact.
- The GPU scheduling, provisioning and transfer scripts of the rented servers are not released; REPRODUCE.md gives
  the per-run and per-readout commands they issued.
- The raw generation traces are a separate archive (see Generation traces); `paper/data/example_traces.txt` holds the one decoded pair the paper's
  implementation appendix quotes, with its GSM8K test problem (MIT license). The fresh set's problem text is not redistributed: `scripts/rebuild_data.py`
  rebuilds it from the public sources by checksum.
- The 9B record has two arms and reads one test set of 30 problems; its item interval is wide. AIME 2025 predates
  the Qwen3.5 release and may be in the base model's training data, which would raise every arm alike.
- Untested: other GPUs than the three named above, other vLLM, PyTorch or transformers versions.

## Citation

    @misc{fenesh2026betagate,
      title  = {Beta Gate: Shorter RL Reasoning in Qwen3.5 Without Negative Eigenvalues},
      author = {Fenesh, Avi},
      year   = {2026},
      note   = {arXiv identifier to be added. Code and receipts: https://github.com/avifenesh/gdn-beta-gate}
    }

Machine-readable metadata: [CITATION.cff](CITATION.cff).

## License and credits

Code: Apache-2.0 ([LICENSE](LICENSE)). Results, receipts, records and documentation (`results/`, `ml_pipeline*/`,
`paper/`, `docs/`, `data/`): CC-BY-4.0 ([LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)). Trained adapters:
Apache-2.0, on the Hub. Dataset and model licenses stay with their sources, listed in
[data/MANIFEST.json](data/MANIFEST.json); the AIME 2025 and AIME 2026 copies used for the fresh set are CC-BY-NC-SA-4.0
on their cards, which is one reason their text is not redistributed here.

The patches copy the Gated DeltaNet forward of the installed transformers (Apache-2.0) and rewrite the installed vLLM
(Apache-2.0) GDN sources at runtime; no third-party source file is redistributed here. Kernels come from
flash-linear-attention. Models: Qwen3.5 (Qwen team), Olmo-Hybrid-7B (Ai2).
