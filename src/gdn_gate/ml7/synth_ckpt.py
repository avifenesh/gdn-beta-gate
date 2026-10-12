#!/usr/bin/env python3
"""A synthetic Qwen3.5 trainable-state file in train_grpo.py's key format (peft LoRA r16 on the four GDN projections
plus one fp32 gate_w per GDN layer), for smoke-testing the readout's plugin path before real step-7 checkpoints
exist. Not a model: small random LoRA (B ~ N(0, b_std)) and a gate W sized so tanh(W x) is O(0.5), which makes the
wide arm produce beta > 1 (a gate that engaged) and the wide1 arm never.
Usage: synth_ckpt.py MODEL_DIR OUT.pt [--rank 16] [--seed 0]
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--b_std", type=float, default=2e-3)
a = ap.parse_args()
md = Path(a.model)
idx = json.loads((md / "model.safetensors.index.json").read_text())["weight_map"]
cfg = json.loads((md / "config.json").read_text())
tc = cfg.get("text_config", cfg)
H, NV = tc["hidden_size"], tc["linear_num_value_heads"]
g = torch.Generator().manual_seed(a.seed)
state = {}
pat = re.compile(r"layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_a|in_proj_b|out_proj)\.weight$")
for k, f in sorted(idx.items()):
    m = pat.search(k)
    if not m or "mtp" in k or "visual" in k:
        continue
    with safe_open(str(md / f), framework="pt", device="cpu") as fh:
        out_f, in_f = fh.get_slice(k).get_shape()
    lid, p = m.group(1), m.group(2)
    pre = f"base_model.model.model.layers.{lid}.linear_attn.{p}"
    state[f"{pre}.lora_A.default.weight"] = torch.randn(a.rank, in_f, generator=g) / in_f ** 0.5
    state[f"{pre}.lora_B.default.weight"] = torch.randn(out_f, a.rank, generator=g) * a.b_std
    state[f"base_model.model.model.layers.{lid}.linear_attn.gate_w.weight"] = torch.randn(NV, H, generator=g) * (0.5 / H ** 0.5)
torch.save(state, a.out)
n_gate = sum("gate_w" in k for k in state)
print(json.dumps({"out": a.out, "tensors": len(state), "gate_layers": n_gate, "hidden": H, "num_v_heads": NV}))
