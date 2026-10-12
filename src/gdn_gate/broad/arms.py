"""Beta-range arms for the broad stream: wide (beta in (0,2)), wide1 (beta in (0,1), the same added
parameters), ctrl (LoRA only), plus loading a gated checkpoint with its gate stripped.

patch.py (src/retrofit) is imported, never edited. Its source rewrite turns
`beta = b.sigmoid()` into `beta = self._wg_beta(b, hidden_states)`; each arm binds its own
`_wg_beta` on the layer, so wide and wide1 run the identical library forward and differ in one
function only.

  wide   beta = s + t * (2 - s if t >= 0 else s)    range (0, 2)   (patch.py, unchanged)
  wide1  beta = s + t * (1 - s if t >= 0 else s)    range (0, 1)   (this file)
  ctrl   beta = s                                    library forward, no added parameters
with s = sigmoid(b), t = tanh(W x) per value head, W zero-initialised (fp32, hidden -> num_v_heads).
Both gated arms add exactly the same parameters (one W per GDN layer, same shape, same init, same lr)
and both give beta = s exactly at W = 0. The only difference is the ceiling: 2 for wide, 1 for wide1.
For wide1, t >= 0 gives s + t (1 - s) <= 1 and t < 0 gives s (1 + t) >= 0, so 1 - beta >= 0 always: the
delta-rule eigenvalue never goes negative. check_wide1_bound() proves it numerically with a red arm.

Strip: a gated arm's LoRA loaded without its W (beta = s, library forward), the same-machine twin that
asks whether a gated arm's scores need the gate at inference.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

if os.environ.get("WG_NO_FLA"):  # CPU tests: make transformers take its torch GDN path (fla is Triton, GPU only)
    sys.modules["fla"] = None

RETRO = Path(__file__).resolve().parents[2] / "retrofit"
if str(RETRO) not in sys.path:
    sys.path.insert(0, str(RETRO))
import patch  # noqa: E402  (src/retrofit/patch.py, imported unchanged)

ARMS = ("wide", "wide1", "ctrl")
CEIL = {"wide": 2.0, "wide1": 1.0}
LORA_TARGETS = r".*layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_a|in_proj_b|out_proj)"


def _wg_beta1(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """wide1: patch._wg_beta with ceiling 1 instead of 2 (same W, same tanh, same casts)."""
    s = b.sigmoid()
    t = torch.tanh(self.gate_w(x.to(self.gate_w.weight.dtype)))
    sf = s.float()
    coef = torch.where(t >= 0, 1.0 - sf, sf)
    beta = (sf + t.float() * coef).to(s.dtype)
    stats = getattr(self, "_wg_stats", None)
    if stats is not None:
        with torch.no_grad():
            # wide1 has no beta > 1 by construction; record the share at the ceiling instead (beta >= 0.999)
            stats.append(((beta.float() >= 0.999).float().mean().item(), t.abs().float().mean().item()))
    return beta


def install(model, arm: str) -> list[nn.Parameter]:
    """Install the arm's gate on every GDN layer of a loaded Qwen3.5 model. Returns the new parameters."""
    assert arm in ARMS, arm
    if arm == "ctrl":
        return []
    params = patch.install(model, beta=True, decay=False)  # wide forward + W on every layer
    if arm == "wide1":
        for layer in patch.gdn_layers(model):
            layer._wg_beta = types.MethodType(_wg_beta1, layer)
    for layer in patch.gdn_layers(model):
        layer._wg_arm = arm
    return params


def beta_of(layer, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return layer._wg_beta(b, x)


def check_wide1_bound(device: str = "cpu", n: int = 4096, seed: int = 0) -> dict:
    """Fails (AssertionError) if wide1 can produce beta > 1 or beta < 0, and fails if the check cannot see a
    violation (red arm: the same probe on wide must find beta > 1). Probes saturating W (t = +-1 exactly in
    fp32), random W, and s across the full range including s = 1 - 2^-24 and bf16 inputs."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for arm in ("wide1", "wide"):
        layer = types.SimpleNamespace()
        hid, heads = 32, 8
        layer.gate_w = nn.Linear(hid, heads, bias=False).to(device=device, dtype=torch.float32)
        fn = _wg_beta1 if arm == "wide1" else patch._wg_beta
        worst_hi, worst_lo = -1.0, 2.0
        for scale in (0.0, 0.1, 1.0, 10.0, 1e4):  # 1e4 saturates tanh to exactly +-1
            with torch.no_grad():
                layer.gate_w.weight.copy_(torch.randn(heads, hid, generator=g) * scale)
            x = torch.randn(n, hid, generator=g).to(device)
            for dtype in (torch.float32, torch.bfloat16):
                b = torch.cat([torch.randn(n - 4, heads, generator=g) * 8,
                               torch.tensor([[40.0] * heads, [-40.0] * heads, [16.6] * heads, [0.0] * heads])])
                b = b.to(device=device, dtype=dtype)
                beta = fn(layer, b, x.to(dtype)).float()
                worst_hi, worst_lo = max(worst_hi, beta.max().item()), min(worst_lo, beta.min().item())
        out[arm] = {"beta_max": worst_hi, "beta_min": worst_lo}
    assert out["wide1"]["beta_max"] <= 1.0 and out["wide1"]["beta_min"] >= 0.0, f"wide1 left [0,1]: {out}"
    assert out["wide"]["beta_max"] > 1.0, f"red arm: the probe never saw wide exceed 1, so it cannot fail: {out}"
    out["pass"] = True
    return out


def add_lora(model, rank: int):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():
        if "gate_w" in n:
            p.requires_grad_(True)
    return model


def load_model(model_path: str, arm: str, device_map=None, dtype=torch.bfloat16, attn: str | None = None):
    from transformers import AutoModelForCausalLM

    kw = dict(dtype=dtype, device_map=device_map if device_map is not None else {"": 0})
    if attn:
        kw["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    new = install(model, arm)
    return model, new


def load_trained(model_path: str, ckpt: str, arm: str, rank: int, strip: bool = False, device_map=None,
                 dtype=torch.bfloat16):
    """Rebuild base + LoRA (+ gate) from a trainable-state file. strip=True: a gated arm's LoRA alone, the
    library forward (beta = s); the dropped W tensors are counted and returned so a caller can assert them."""
    assert not (strip and arm == "ctrl"), "ctrl has no gate to strip"
    model, _ = load_model(model_path, "ctrl" if strip else arm, device_map=device_map, dtype=dtype)
    model = add_lora(model, rank)
    state = torch.load(ckpt, map_location="cpu")
    params = dict(model.named_parameters())
    dropped = [k for k in state if "gate_w" in k] if strip else []
    keep = {k: v for k, v in state.items() if k not in dropped}
    missing = [k for k in keep if k not in params]
    assert not missing, missing[:5]
    want = [k for k, p in params.items() if p.requires_grad]
    unloaded = [k for k in want if k not in keep]
    assert not unloaded, f"trainable tensors absent from the checkpoint: {unloaded[:5]}"
    with torch.no_grad():
        for k, v in keep.items():
            params[k].copy_(v.to(params[k].dtype))
    if strip:
        assert dropped, "strip asked on a checkpoint with no gate tensors"
    return model.eval(), {"arm": arm, "strip": strip, "dropped_gate_tensors": len(dropped), "loaded": len(keep)}


def core(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model, m.lm_head


if __name__ == "__main__":
    import json

    print(json.dumps(check_wide1_bound(os.environ.get("DEV", "cpu"))))
