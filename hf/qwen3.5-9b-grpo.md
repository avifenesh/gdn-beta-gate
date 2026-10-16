---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
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

# β gate adapters: Qwen3.5-9B, GRPO

Trained adapters for the paper "Beta Gate: Shorter RL Reasoning in Qwen3.5 Without Negative Eigenvalues" (Avi Fenesh, Tiyuvta.ai, 2026; Results section, Qwen3.5-9B; code release v1.2.0). The β gate is a zero-initialized
matrix `W` per Gated DeltaNet layer that moves the delta-rule write strength `beta` toward its floor or its ceiling,
`tau = tanh(W x)`. These are the adapters of the paper's 9B record: Qwen3.5-9B trained with GRPO on GSM8K under a correctness-gated length penalty, in two arms (LoRA, and LoRA plus the β gate with ceiling 1) with 3 seeds each, plus the shared warm-up both arms start from. The gated arm is more accurate than LoRA on AIME 2025 at every thinking budget and reasons shorter; on GSM8K validation it reasons 63 to 79% shorter at the same accuracy.

Code, receipts and the claims checker: https://github.com/avifenesh/gdn-beta-gate. Write-up: https://huggingface.co/blog/Avifenesh/beta-gate-qwen35. arXiv: to be added.

Base model: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Apache-2.0).
7 adapters, 347 MB in total.

## How to load

These are not PEFT adapter folders and `peft` cannot load them: the gated arm needs the patched Gated DeltaNet
forward from the code repository. `library_name` is `pytorch` because the files are plain PyTorch tensors for that
code, not a PEFT or transformers repository. Each `adapter.safetensors` holds the
trainable tensors under the training code's names: PEFT LoRA tensors (adapter name `default`) and, for the gated arm,
one float32 `gate_w` matrix per GDN layer (24 layers). `scripts/adapters.py to-pt` turns a file back into the `.pt` checkpoint
that the code's `--ckpt` and `--init_from` flags read; the round trip is bit for bit.

    hf download Avifenesh/gdn-beta-gate-qwen3.5-9b-grpo --include "gate01/seed0/*" --local-dir adapters
    git clone https://github.com/avifenesh/gdn-beta-gate && cd gdn-beta-gate
    python scripts/adapters.py to-pt ../adapters/gate01/seed0/adapter.safetensors gate01_s0.pt
    uv run python scripts/rebuild_data.py                       # test files under runs/ml7/data, checked by sha256
    python src/gdn_gate/ml7/readout_vllm.py --model $WG_HF_CACHE/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
           --data runs/ml7/data/test_aime2025.jsonl --arm wide1 --ckpt gate01_s0.pt \
           --caps 256,512,1024,2048,4096,8192 --answer_max 1280 --samples 8 --seed 20261006 --gpu_mem 0.35 \
           --probe_effect 0.05 --rank 16 --out readout_aime2025_gate01_s0.json

`--arm` per folder: `lora` --arm ctrl, `gate01` --arm wide1, `warmup` --arm ctrl. `to-pt` needs `torch` and `safetensors`;
the readout needs the vLLM environment `env/vllm.txt` (vLLM 0.31.0, which the gate plugin checks) and one GPU with
enough memory for the Qwen3.5-9B engine (the record used one NVIDIA DGX Spark GB10, 128 GB unified memory, one readout per unit).

## Contents

| Folder | Paper name | Code name | What it is |
|---|---|---|---|
| `lora/seed0` to `seed2` | LoRA | `ctrl` | LoRA only; beta = sigmoid(b) in (0, 1) |
| `gate01/seed0` to `seed2` | Gate(0,1) | `wide1` | LoRA plus the β gate with ceiling 1: beta in [0, 1] |
| `warmup/seed0` | shared warm-up | `ctrl`, 40 steps | LoRA after 40 GRPO steps with the correctness reward alone; both arms start here |

No `lora-r20` or `gate02` folder: the 9B record trained neither the parameter-matched LoRA nor the β gate with
ceiling 2.

Each folder holds `adapter.safetensors` and `adapter_meta.json` (record, arm, seed, step, LoRA rank and alpha, gate
ceiling, base revision, sha256 of the source checkpoint). `SHA256SUMS` covers every file:
`sha256sum -c SHA256SUMS` after a full download.

## Training

The 4B record's recipe on Qwen3.5-9B. GRPO (no KL term) with the correctness-gated length penalty of Arora and Zanette (alpha 0.2) on reasoning tokens, 150 steps of 16 prompts x 8 samples from 6,965 GSM8K training problems, starting from the shared 40-step warm-up. LoRA rank 16 (alpha 32) on `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `out_proj` of every Gated DeltaNet layer; AdamW, lr 2e-5 for LoRA and gate; training cap 512 reasoning tokens, answer cap 1,280; training micro-row 4,096 tokens (a memory setting; the update does not depend on it). One NVIDIA DGX Spark GB10 per run, 10.8 to 12.0 hours of logged training steps per 150-step run; rollouts on vLLM 0.31.0 with the gate plugin.

## Evaluation

Read under the paper's locked protocol: vendor sampling (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5), thinking budgets 256 to 8,192 tokens, area = anytime accuracy (area under the accuracy-budget curve), normalized over log2 budgets, length = mean reasoning tokens at the 8,192 budget. Means of 3 seeds.

| Arm | AIME 2025 area | AIME 2025 accuracy at 8,192 | AIME 2025 length | GSM8K validation area | GSM8K validation length |
|---|---|---|---|---|---|
| untouched Qwen3.5-9B | 0.122 | 0.225 | 8,032 | 0.956 | 2,870 |
| LoRA | 0.169 | 0.275 | 7,869 | 0.968 | 1,825 |
| Gate(0,1) | 0.272 | 0.482 | 7,210 | 0.971 | 502 |

AIME 2025 (MathArena/aime_2025, AIME I and II, 30 problems): 8 samples per problem, read once, pre-registered with the 4B fresh-set rule. Both hypotheses hold at 9B: Gate(0,1) - LoRA area +0.103 (95% interval [0.061, 0.147], 3 of 3 seeds) and length -659 tokens ([-1,036, -323], 3 of 3). Accuracy at the 8,192 budget is +0.207 ([0.115, 0.306]), and Gate(0,1) is above LoRA at every budget in every seed. Most traces still reach the 8,192 budget (share cut off: LoRA 0.87, Gate(0,1) 0.71).

GSM8K validation slice (500 items x 2 samples; no test set): Gate(0,1) reasons shorter than LoRA in every seed (-1,387, -1,189 and -1,394 tokens, 63 to 79% shorter) with the same accuracy at the 8,192 budget (0.976 for both arms). 9B is near its ceiling on GSM8K, which is why the record read AIME 2025.

Not read at 9B: MATH-500, the GSM8K test set and the rest of the fresh set. No mechanism tests ran at 9B.

Every number maps to a receipt in the code repository: `results/ml7_9b/aime2025/analysis_fresh_9b.json` (AIME 2025; `scripts/rerun_analysis.py` reruns it from the released readouts) and `results/ml7_9b/step9_*.json` (validation).

## Limitations

- Research artifacts for reproducing and probing the paper's results. They are not tuned for deployment or safety
  and inherit the base model's limitations.
- Two arms at 9B (no Gate(0,2), no LoRA-r20), 3 seeds per arm, GSM8K training prompts only, thinking budgets up to
  8,192 tokens. One test set of 30 problems: the item interval is wide.
- AIME 2025 (February 2025) predates the Qwen3.5 release and may be in the base model's training data; that would
  raise every arm alike.
- The RL results cover one model family (Qwen3.5) at 2B, 4B and 9B.
- Readouts are sampled; a rerun reproduces the protocol and, within seed and item noise, the numbers, not the bytes.
- Trained and read with vLLM 0.31.0, PyTorch 2.13 and transformers 5.17 on NVIDIA DGX Spark (GB10) only.

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
