---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: adapter
library_name: pytorch
pipeline_tag: text-generation
language:
- en
datasets:
- allenai/tulu-3-sft-mixture
tags:
- lora
- gated-deltanet
- qwen3.5
- beta-gate
- sft
- chat
- safetensors
- research
---

# β gate adapters: Qwen3.5-4B-Base, chat SFT

Trained adapters for the paper "Beta Gate: Shorter RL Reasoning in Qwen3.5 Without Negative Eigenvalues" (Avi Fenesh, Tiyuvta.ai, 2026; appendix on supervised fine-tuning and forgetting). The β gate is a zero-initialized
matrix `W` per Gated DeltaNet layer that moves the delta-rule write strength `beta` toward its floor or its ceiling,
`tau = tanh(W x)`. These are the chat SFT adapters of the paper's supervised check: Qwen3.5-4B-Base fine-tuned on a Tulu 3 mixture in three arms (LoRA, and LoRA plus the β gate with ceiling 1 or 2), 6 seeds each. Gate(0,1) lowers held-out loss in every seed; opening the negative-eigenvalue range (ceiling 2) gives back about half of that gain.

Code, receipts and the claims checker: https://github.com/avifenesh/gdn-beta-gate. Write-up: https://huggingface.co/blog/Avifenesh/beta-gate-qwen35. arXiv: to be added.

Base model: [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base) at revision `1001bb4d826a52d1f399e183466143f4da7b741b` (Apache-2.0).
18 adapters, 720 MB in total.

## How to load

These are not PEFT adapter folders and `peft` cannot load them: the gated arms need the patched Gated DeltaNet
forward from the code repository. `library_name` is `pytorch` because the files are plain PyTorch tensors for that
code, not a PEFT or transformers repository. Each `adapter.safetensors` holds the
trainable tensors under the training code's names: PEFT LoRA tensors (adapter name `default`) and, for the gated arms,
one float32 `gate_w` matrix per GDN layer. `scripts/adapters.py to-pt` turns a file back into the `.pt` checkpoint
that the code's `--ckpt` and `--init_from` flags read; the round trip is bit for bit.

    hf download Avifenesh/gdn-beta-gate-qwen3.5-4b-base-chat-sft --include "gate01/seed0/*" --local-dir adapters
    git clone https://github.com/avifenesh/gdn-beta-gate && cd gdn-beta-gate
    python scripts/adapters.py to-pt ../adapters/gate01/seed0/adapter.safetensors gate01_s0.pt
    # HumanEval and the broad battery with the gate inside vLLM (argv of the paper's runs: results/vllm/step6_s0/*.json)
    python src/gdn_gate/vllm/vscore_4b.py --items <items.json> --engine wide1 --job gate01_s0=gate01_s0.pt --out_dir out
    # or as the starting point of further SFT: src/gdn_gate/broad/train_sft.py --init_from gate01_s0.pt ...

`--engine` / `--arm` per folder: `lora` ctrl (stock engine), `gate01` wide1, `gate02` wide.

## Contents

| Folder | Paper name | Code name | What it is |
|---|---|---|---|
| `lora/seed0` to `seed5` | LoRA | `ctrl` | LoRA only; beta = sigmoid(b) in (0, 1) |
| `gate01/seed0` to `seed5` | Gate(0,1) | `wide1` | LoRA plus the β gate with ceiling 1: beta in [0, 1] |
| `gate02/seed0` to `seed5` | Gate(0,2) | `wide` | LoRA plus the β gate with ceiling 2: beta in [0, 2) |

Each folder holds `adapter.safetensors` and `adapter_meta.json` (record, arm, seed, step, LoRA rank and alpha, gate
ceiling, base revision, sha256 of the source checkpoint). `SHA256SUMS` covers every file:
`sha256sum -c SHA256SUMS` after a full download.

## Training

Chat SFT on the permissive subsets of allenai/tulu-3-sft-mixture (the CC-BY-NC No Robots subset dropped), 1,200 steps of about 33,000 tokens, LoRA rank 16, alpha 32, on the four GDN projections, lr 1e-4 for LoRA and gate. Unsloth stack; seed 0 on NVIDIA GB10 (DGX Spark), seeds 1 to 5 on one NVIDIA RTX PRO 6000 Blackwell Server Edition per run; checked against the plain trainer of the code repository.

## Evaluation

Held-out loss on 26,312 tokens at step 1,200, paired by seed: Gate(0,1) - LoRA -0.0061 nats/token (6 of 6 seeds lower), Gate(0,2) - LoRA -0.0029 (5 of 6), Gate(0,2) - Gate(0,1) +0.0032 (6 of 6 higher).

Every number maps to a receipt and a check in the code repository (`paper/CLAIMS.md`, `paper/figs/check_claims.py`).

## Limitations

- Research artifacts for reproducing and probing the paper's results. They are not tuned for deployment or safety
  and inherit the base model's limitations.
- The RL results cover one model family (Qwen3.5) at 2B, 4B and 9B, 3 to 6 seeds per arm, GSM8K training prompts only,
  and thinking budgets up to 8,192 tokens.
- Readouts are sampled; a rerun reproduces the protocol and, within seed and item noise, the numbers, not the bytes.
- Tested with vLLM 0.31.0, PyTorch 2.13 and transformers 5.17 on NVIDIA RTX PRO 6000, GB10 and RTX 5090 GPUs.

## License

Apache-2.0, as the base model. Training data licenses stay with their sources (code repository,
`data/MANIFEST.json`).

## Citation

    @misc{fenesh2026betagate,
      title  = {Beta Gate: Shorter RL Reasoning in Qwen3.5 Without Negative Eigenvalues},
      author = {Fenesh, Avi},
      year   = {2026},
      note   = {arXiv identifier to be added. Code: https://github.com/avifenesh/gdn-beta-gate}
    }
