#!/usr/bin/env python3
"""Initial trainable tensors exactly as train_sft.py creates them for an arm and seed (torch.manual_seed(seed), library
load on GPU 0 in bf16, the arm's gate, PEFT LoRA r16), saved under train_sft.py's names. Built on the GPU like
train_sft.py: PEFT initialises lora_A on the base layer's device, so a CPU build draws different values. train_sft_fast.py --init_from
starts a fast-stack run from this file, so a fast run has the same initial LoRA draw as the train_sft.py run would.
Usage: dump_init.py MODEL ARM SEED OUT"""

import os
import sys
from pathlib import Path

sys.modules["causal_conv1d"] = None
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "broad"))
import torch  # noqa: E402

import arms  # noqa: E402

model_path, arm, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
torch.manual_seed(seed)
model, _ = arms.load_model(model_path, arm, device_map={"": 0}, dtype=torch.bfloat16)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = arms.add_lora(model, 16)
state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
torch.save(state, out)
print(len(state), "tensors ->", out)
