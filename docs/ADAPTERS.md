# Adapters on the Hugging Face Hub

The 72 trained adapters the paper reads are published on the Hub under the account `Avifenesh`, one repository per
training recipe and base model, one folder per arm and seed, exported to safetensors with `scripts/adapters.py`.
The model cards are in `hf/` (each is the repository's `README.md`). License: Apache-2.0, as the base models
(Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-4B-Base and Qwen3.5-9B-Base are Apache-2.0 on their Hub cards). Collection:
https://huggingface.co/collections/Avifenesh/gate-qwen35-6accb63081800b2f2d583b80

| Hub repository | Revision (commit, adapters plus card) | Contents | Adapters | Size |
|---|---|---|---|---|
| [`Avifenesh/gdn-beta-gate-qwen3.5-4b-grpo`](https://huggingface.co/Avifenesh/gdn-beta-gate-qwen3.5-4b-grpo) | `ee1e6c0b031b13f009b50009278cc03f5b1d07ec` | Qwen3.5-4B RL record: LoRA, LoRA-r20, Gate(0,1), Gate(0,2) x seeds 0 to 2 (step 150), warm-up (step 40) | 13 | 525 MB |
| [`Avifenesh/gdn-beta-gate-qwen3.5-9b-grpo`](https://huggingface.co/Avifenesh/gdn-beta-gate-qwen3.5-9b-grpo) | `3e9fbcdb39716267d8062f3b159b21ea05dc58f2` | Qwen3.5-9B RL record (v1.2.0): LoRA, Gate(0,1) x seeds 0 to 2 (step 150), warm-up (step 40); no LoRA-r20 or Gate(0,2) arm at 9B | 7 | 347 MB |
| [`Avifenesh/gdn-beta-gate-qwen3.5-2b-grpo`](https://huggingface.co/Avifenesh/gdn-beta-gate-qwen3.5-2b-grpo) | `9bb0c56a6b4b43f10a7f6b2cfe50349c2b80eb6f` | Qwen3.5-2B RL record: LoRA, Gate(0,1), Gate(0,2) x seeds 0 to 5 (step 150), warm-up (step 40) | 19 | 388 MB |
| [`Avifenesh/gdn-beta-gate-qwen3.5-4b-base-chat-sft`](https://huggingface.co/Avifenesh/gdn-beta-gate-qwen3.5-4b-base-chat-sft) | `333c4445c97aa603e998f5ddb297db71c769fff2` | chat SFT (appendix): LoRA, Gate(0,1), Gate(0,2) x seeds 0 to 5, step 1,200 | 18 | 720 MB |
| [`Avifenesh/gdn-beta-gate-qwen3.5-4b-base-forgetting`](https://huggingface.co/Avifenesh/gdn-beta-gate-qwen3.5-4b-base-forgetting) | `20bb1d22ae83042682c8f024ce9e8628dc4f628e` | forgetting recipe (appendix): LoRA, Gate(0,1), Gate(0,2) x seeds 0 to 2, step 1,200 | 9 | 360 MB |
| [`Avifenesh/gdn-beta-gate-qwen3.5-9b-base-forgetting`](https://huggingface.co/Avifenesh/gdn-beta-gate-qwen3.5-9b-base-forgetting) | `8be3d70f03f5f6211536a37debbacca7885c728a` | forgetting recipe at 9B (appendix): LoRA, Gate(0,2) x seeds 0 to 2, step 1,200 | 6 | 303 MB |

Total about 2.6 GB. Not published: the intermediate checkpoints (every 25 RL steps, every 100 or 300 SFT steps) and
the mechanism inputs derived from the published checkpoints (the hybrid and dose checkpoints;
`src/gdn_gate/mech/hybrid_ckpts.py` and `dose_ckpts.py` rebuild them, their sha256 are in
`results/mech/hybrid_receipt.json` and `dose_receipt.json`).

These are not PEFT adapter folders: the gated arms need the patched Gated DeltaNet forward of this repository (the
cards set `library_name: pytorch`: plain PyTorch tensors for this code). Each `adapter.safetensors` holds the
trainable tensors under the training code's names (PEFT LoRA tensors with adapter name `default`; for the gated
arms one float32 `gate_w` matrix per GDN layer, zero at initialization). `adapter_meta.json` gives record, arm, seed, step, LoRA rank and alpha (alpha = 2 x rank), gate
ceiling, base revision and the sha256 of the source checkpoint. `SHA256SUMS` covers every file of the repository.

## Use

    hf download Avifenesh/gdn-beta-gate-qwen3.5-4b-grpo --include "gate01/seed0/*" --local-dir adapters
    python scripts/adapters.py to-pt adapters/gate01/seed0/adapter.safetensors gate01_s0.pt
    python src/gdn_gate/ml7/readout_vllm.py --model <Qwen3.5-4B snapshot 851bf6e8> --arm wide1 --ckpt gate01_s0.pt ...

`--arm` per folder: `lora` ctrl, `lora-r20` ctrl with `--rank 20`, `gate01` wide1, `gate02` wide, `warmup` ctrl (9B:
`--model` the Qwen3.5-9B snapshot c2022362, `--rank 16`). The
forgetting adapters load through `src/gdn_gate/rescore/rescore_4b.py` (9B: `rescore_9b.py`) with the run's
`result.json` from `results/run_configs/forgetting/` beside `trainable.pt`; the model cards give the exact commands
per repository.

## Export and verification (author, CPU, 2026-10-12)

    python scripts/adapters.py list   --runs <lab run root> --retrofit_runs <lab retrofit run root>   # 72 found, 0 missing
    python scripts/adapters.py export --runs <lab run root> --retrofit_runs <lab retrofit run root> --out <dir>

`export` reads each checkpoint with `torch.load(weights_only=True)`, renames the gate tensor to `gate_w`, and writes
`<record>/<arm>/seed<k>/adapter.safetensors`, `adapter_meta.json` and `<record>/SHA256SUMS`. Checks before upload: all
59 exported files equal their source checkpoints tensor by tensor (same keys after the rename, same dtype,
`torch.equal`), and the round trip `export` then `to-pt` gives back every tensor bit for bit on one gated adapter per
repository (2B, 4B, chat SFT: Gate(0,1) seed 0; forgetting: Gate(0,2) seed 0). The 9B forgetting repository was
exported later the same way (`--record qwen3.5-9b-base-forgetting`): its 6 files equal their source checkpoints tensor
by tensor, and the round trip is bit for bit on LoRA seed 0 and Gate(0,2) seed 0. The 9B GRPO repository (v1.2.0,
2026-10-16, `--record qwen3.5-9b-grpo`): each source checkpoint's sha256 equals `src/gdn_gate/ml7/test9b/ckpt_expect_9b.json`,
its 7 files equal their source checkpoints tensor by tensor, and the round trip is bit for bit on all 7. After upload, each Hub repository's
file list and sizes equal the local export (plus the Hub's `.gitattributes`), and one downloaded adapter per
repository matches the `SHA256SUMS` line of its files.

## Upload (author)

    for R in qwen3.5-4b-grpo qwen3.5-2b-grpo qwen3.5-4b-base-chat-sft qwen3.5-4b-base-forgetting qwen3.5-9b-base-forgetting \
             qwen3.5-9b-grpo; do
      cp hf/$R.md <dir>/$R/README.md
      hf repo create Avifenesh/gdn-beta-gate-$R --repo-type model
      hf upload Avifenesh/gdn-beta-gate-$R <dir>/$R . --repo-type model --commit-message "Adapters for the paper, release v1.0.0"
    done

A Hub DOI per repository can be minted later (Settings, Generate DOI).
