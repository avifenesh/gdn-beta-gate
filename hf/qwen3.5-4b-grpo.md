---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: adapter
library_name: pytorch
pipeline_tag: text-generation
language:
- en
datasets:
- openai/gsm8k
tags:
- lora
- gated-deltanet
- qwen3.5
- beta-gate
- grpo
- reasoning
- math
- safetensors
- research
---

# β gate adapters: Qwen3.5-4B, GRPO

Trained adapters for the paper "Beta Gate: Shorter RL Reasoning in Qwen3.5 Without Negative Eigenvalues" (Avi Fenesh, Tiyuvta.ai, 2026; Results and Mechanism sections). The β gate is a zero-initialized
matrix `W` per Gated DeltaNet layer that moves the delta-rule write strength `beta` toward its floor or its ceiling,
`tau = tanh(W x)`. These are the adapters of the paper's main record: Qwen3.5-4B trained with GRPO on GSM8K under a correctness-gated length penalty, in four arms (LoRA, a parameter-matched LoRA-r20, and LoRA plus the β gate with ceiling 1 or 2) with 3 seeds each, plus the shared warm-up every arm starts from. The gated arms score higher on MATH-500 and reason far shorter; the parameter-matched LoRA does not reproduce this.

Code, receipts and the claims checker: https://github.com/avifenesh/gdn-beta-gate. Write-up: https://huggingface.co/blog/Avifenesh/beta-gate-qwen35. arXiv: to be added.

Base model: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (Apache-2.0).
13 adapters, 525 MB in total.

## How to load

These are not PEFT adapter folders and `peft` cannot load them: the gated arms need the patched Gated DeltaNet
forward from the code repository. `library_name` is `pytorch` because the files are plain PyTorch tensors for that
code, not a PEFT or transformers repository. Each `adapter.safetensors` holds the
trainable tensors under the training code's names: PEFT LoRA tensors (adapter name `default`) and, for the gated arms,
one float32 `gate_w` matrix per GDN layer. `scripts/adapters.py to-pt` turns a file back into the `.pt` checkpoint
that the code's `--ckpt` and `--init_from` flags read; the round trip is bit for bit.

    hf download Avifenesh/gdn-beta-gate-qwen3.5-4b-grpo --include "gate01/seed0/*" --local-dir adapters
    git clone https://github.com/avifenesh/gdn-beta-gate && cd gdn-beta-gate
    python scripts/adapters.py to-pt ../adapters/gate01/seed0/adapter.safetensors gate01_s0.pt
    uv run python scripts/rebuild_data.py                       # test files under runs/ml7/data, checked by sha256
    python src/gdn_gate/ml7/readout_vllm.py --model $WG_HF_CACHE/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
           --data runs/ml7/data/test_math500.jsonl --arm wide1 --ckpt gate01_s0.pt \
           --caps 256,512,1024,2048,4096,8192 --answer_max 1280 --samples 2 --seed 20261006 --gpu_mem 0.45 \
           --probe_effect 0.05 --out readout_math500_gate01_s0.json

`--arm` per folder: `lora` --arm ctrl, `lora-r20` --arm ctrl --rank 20, `gate01` --arm wide1, `gate02` --arm wide, `warmup` --arm ctrl. `to-pt` needs `torch` and `safetensors`;
the readout needs the vLLM environment `env/vllm.txt` (vLLM 0.31.0, which the gate plugin checks) and one GPU with
enough memory for the Qwen3.5-4B engine (the paper used 96 GB, two readouts per GPU).

## Contents

| Folder | Paper name | Code name | What it is |
|---|---|---|---|
| `lora/seed0` to `seed2` | LoRA | `ctrl` | LoRA only; beta = sigmoid(b) in (0, 1) |
| `lora-r20/seed0` to `seed2` | LoRA-r20 | `ctrlpm` | LoRA rank 20, about as many trainable parameters as the gated arms; no gate |
| `gate01/seed0` to `seed2` | Gate(0,1) | `wide1` | LoRA plus the β gate with ceiling 1: beta in [0, 1] |
| `gate02/seed0` to `seed2` | Gate(0,2) | `wide` | LoRA plus the β gate with ceiling 2: beta in [0, 2) |
| `warmup/seed0` | shared warm-up | `ctrl`, 40 steps | LoRA after 40 GRPO steps with the correctness reward alone; every arm starts here |

Each folder holds `adapter.safetensors` and `adapter_meta.json` (record, arm, seed, step, LoRA rank and alpha, gate
ceiling, base revision, sha256 of the source checkpoint). `SHA256SUMS` covers every file:
`sha256sum -c SHA256SUMS` after a full download.

## Training

GRPO (no KL term) with the correctness-gated length penalty of Arora and Zanette (alpha 0.2) on reasoning tokens, 150 steps of 16 prompts x 8 samples from 6,965 GSM8K training problems, starting from the shared 40-step warm-up. LoRA rank 16 (alpha 32; LoRA-r20: rank 20, alpha 40, widened from the warm-up) on `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `out_proj` of every Gated DeltaNet layer; AdamW, lr 2e-5 for LoRA and gate; training cap 512 reasoning tokens, answer cap 1,280. One NVIDIA RTX PRO 6000 Blackwell Server Edition per run; rollouts on vLLM 0.31.0 with the gate plugin.

## Evaluation

Read under the paper's locked protocol: vendor sampling (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5), thinking budgets 256 to 8,192 tokens, area = anytime accuracy (area under the accuracy-budget curve), normalized over log2 budgets, length = mean reasoning tokens at the 8,192 budget. Means of 3 seeds.

| Arm | MATH-500 area | MATH-500 length | GSM8K length | Fresh set area | Fresh set length |
|---|---|---|---|---|---|
| LoRA | 0.786 | 4,223 | 1,666 | 0.285 | 7,424 |
| LoRA-r20 | 0.792 | 4,003 | 1,296 | 0.297 | 7,325 |
| Gate(0,1) | 0.852 | 2,202 | 297 | 0.395 | 6,094 |
| Gate(0,2) | 0.849 | 2,153 | 261 | 0.407 | 5,981 |

MATH-500 (HuggingFaceH4/MATH-500) and GSM8K test (openai/gsm8k, 1,319 items): 2 samples per item. Fresh set: 130 competition problems never used in training or selection (AIME 2024, AIME 2025, AIME 2026, AMC 2023), 8 samples per item, read once. The pre-registered fresh-set hypotheses hold at 4B: Gate(0,1) - LoRA area +0.110 (95% interval [0.089, 0.132], 3 of 3 seeds) and length -1,329 tokens ([-1,585, -1,081], 3 of 3).

Mechanism (GSM8K validation slice, 500 items x 1 sample): removing the trained gate from Gate(0,1) at inference raises the mean length from 307 to 665 tokens; the trained gate alone on the warm-up LoRA gives 970 tokens against 2,916 for the warm-up; scaling the gate matrix from 0 to 2 shortens reasoning monotonically in all 3 seeds.

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
