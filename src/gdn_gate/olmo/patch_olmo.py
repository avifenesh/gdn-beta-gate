"""Beta-range arms for Olmo-Hybrid Gated DeltaNet layers (P6), inference lesions and training gates.

Olmo-Hybrid ships with linear_allow_neg_eigval true: the library computes
    beta = b_proj(h).sigmoid(); beta = beta * 2.0          (beta in (0, 2), eigenvalue 1 - beta in (-1, 1))
The installed transformers forward is copied at runtime with exactly that block replaced by
    beta = self._wg_beta(self.b_proj(hidden_states), hidden_states)
Everything else (conv, decay, kernel dispatch to fla, cache handling, the accelerate decorator) stays the library's
own code, pinned by the version and source asserts below. No kernel changes.

Per-layer beta modes (inference lesions), s = sigmoid(b):
  stock   2 s                 the shipped model, bit-identical to the library forward
  half    s                   what linear_allow_neg_eigval=false gives: every write strength halved
  erase   min(2 s, 1)         erase-only clamp: identical wherever 2 s <= 1, caps the rest at a full erase
Head mask (on top of the mode): heads in the mask use min(2 s, 1), the others keep the mode's beta. A per-layer
clamp is mode erase on one layer; a top-heads clamp is a head mask on the heads ranked by how often 2 s > 1.

Training gate (the lane's zero-init widened gate, patch.py's form with a range R):
  base = the mode's beta (stock for R = 2, erase for R = 1), t = tanh(W x) per head, W zero-initialised,
  beta = base + t * (R - base if t >= 0 else base)
so beta = base exactly at W = 0 and beta stays in [0, R] for any W.
"""

from __future__ import annotations

import inspect
import textwrap
import types

import torch
import torch.nn as nn
import transformers
from transformers.models.olmo_hybrid import modeling_olmo_hybrid as M

PINNED_TRANSFORMERS = "5.17.0"
TARGET = "beta = self.b_proj(hidden_states).sigmoid()\n    if self.allow_neg_eigval:\n        beta = beta * 2.0\n"
REPLACEMENT = "beta = self._wg_beta(self.b_proj(hidden_states), hidden_states)\n"
MODES = ("stock", "half", "erase")

_FORWARD = None


def _build_forward():
    assert transformers.__version__ == PINNED_TRANSFORMERS, transformers.__version__
    src = textwrap.dedent(inspect.getsource(M.OlmoHybridGatedDeltaNet.forward))
    assert src.count(TARGET) == 1, "upstream Olmo-Hybrid GDN forward changed"
    assert src.startswith('@force_accelerate_hooks("conv1d")\n') and "force_accelerate_hooks" in M.__dict__
    src = src.replace(TARGET, REPLACEMENT)
    ns: dict = {}
    exec(compile(src, "<wg_olmo_forward>", "exec"), M.__dict__, ns)
    return ns["forward"]


def _wg_beta(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    s = b.sigmoid()
    stock = s * 2.0  # the library's exact expression and dtype
    mode = self._ng_mode
    if mode == "stock":
        beta = stock
    elif mode == "half":
        beta = s
    elif mode == "erase":
        beta = stock.clamp(max=1.0)
    else:
        raise ValueError(mode)
    mask = self._ng_head_mask
    if mask is not None:
        beta = torch.where(mask, stock.clamp(max=1.0), beta)
    w = getattr(self, "gate_w", None)
    if w is not None:
        t = torch.tanh(w(x.to(w.weight.dtype))).float()
        bf = beta.float()
        coef = torch.where(t >= 0, self._ng_range - bf, bf)
        beta = (bf + t * coef).to(stock.dtype)
    st = self._ng_stats
    if st is not None:
        with torch.no_grad():
            flat_stock = stock.reshape(-1, stock.shape[-1])
            flat_beta = beta.reshape(-1, beta.shape[-1])
            st["tokens"] += flat_stock.shape[0]
            st["stock_gt1"] += (flat_stock > 1.0).sum(0).double()
            st["beta_gt1"] += (flat_beta > 1.0).sum(0).double()
            st["beta_sum"] += flat_beta.double().sum(0)
            st["stock_sum"] += flat_stock.double().sum(0)
    return beta


def gdn_layers(model) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, M.OlmoHybridGatedDeltaNet)]


def checkpoint_is_nope(name_or_path: str) -> bool:
    """True when the checkpoint's own config.json declares no RoPE: `rope_parameters: null` (Think-SFT) or
    `rope_parameters: {"rope_theta": null}` (Base, Instruct-SFT), with no top-level rope_theta."""
    import json

    from transformers.utils import cached_file

    if not name_or_path:  # built in memory from a config object (CPU tests): no checkpoint to read
        return False
    raw = json.load(open(cached_file(name_or_path, "config.json")))
    if "rope_parameters" not in raw or raw.get("rope_theta") is not None:
        return False
    rp = raw["rope_parameters"]
    return rp is None or rp.get("rope_theta") is None


def fix_position_encoding(model) -> str:
    """transformers 5.17 loads Think-SFT's `rope_parameters: null` as {"rope_theta": 1e4} (RotaryEmbeddingConfigMixin.
    convert_rope_params_to_dict fills default_theta), so OlmoHybridModel builds a rotary_emb and the full-attention
    layers run RoPE on a NoPE checkpoint (the modeling code's own comment: released checkpoints set it to None for
    NoPE; Base and Instruct-SFT write {"rope_theta": null} and load correctly). Measured 2026-10-08 on Think-SFT:
    NoPE NLL 2.052 vs RoPE 2.331 on wikitext-103 (4k windows), 1.862 vs 1.910 on MATH-500 chat. vLLM 0.31.0 already
    runs NoPE. Drops the rotary embedding when the checkpoint declares NoPE; returns "nope" or "rope"."""
    if checkpoint_is_nope(model.config._name_or_path):
        model.model.rotary_emb = None
        return "nope"
    return "rope" if model.model.rotary_emb is not None else "nope"


def install(model) -> list[nn.Module]:
    """Swap every GDN layer's forward for the patched copy, mode stock (bit-identical to the library), after
    fix_position_encoding (NoPE checkpoints run without RoPE)."""
    global _FORWARD
    if _FORWARD is None:
        _FORWARD = _build_forward()
    model._ng_position = fix_position_encoding(model)
    layers = gdn_layers(model)
    assert layers, "no Olmo-Hybrid GDN layers"
    for layer in layers:
        assert layer.allow_neg_eigval, "the arms are defined relative to the shipped (0, 2) model"
        layer._ng_mode = "stock"
        layer._ng_head_mask = None
        layer._ng_stats = None
        layer._wg_beta = types.MethodType(_wg_beta, layer)
        layer.forward = types.MethodType(_FORWARD, layer)
    return layers


def set_arm(model, arm: str, layer: int | None = None, heads: dict[int, list[int]] | None = None) -> dict:
    """Inference arm on every GDN layer. arm: stock | half | erase | layer (erase on GDN layer `layer`, stock
    elsewhere) | heads (stock everywhere, min(2s, 1) on heads[gdn_idx]). Returns a description for receipts."""
    layers = gdn_layers(model)
    for i, L in enumerate(layers):
        L._ng_head_mask = None
        if arm in MODES:
            L._ng_mode = arm
        elif arm == "layer":
            assert layer is not None and 0 <= layer < len(layers)
            L._ng_mode = "erase" if i == layer else "stock"
        elif arm == "heads":
            L._ng_mode = "stock"
            hs = (heads or {}).get(i, [])
            if hs:
                m = torch.zeros(L.num_v_heads, dtype=torch.bool, device=L.b_proj.weight.device)
                m[torch.tensor(hs)] = True
                L._ng_head_mask = m
        else:
            raise ValueError(arm)
    desc = {"arm": arm}
    if arm == "layer":
        desc.update(gdn_index=layer, model_layer=layers[layer].layer_idx)
    if arm == "heads":
        desc.update(n_heads=sum(len(v) for v in (heads or {}).values()),
                    heads={str(k): v for k, v in sorted((heads or {}).items()) if v})
    return desc


def install_gate(model, beta_range: float) -> list[nn.Parameter]:
    """Training gate: zero-init W per GDN layer, range R in {2, 1}. R = 2 keeps the stock base (identity at W = 0);
    R = 1 puts every layer in erase mode first (base in (0, 1]) so beta never exceeds 1."""
    assert beta_range in (1.0, 2.0), beta_range
    params = []
    for L in gdn_layers(model):
        assert hasattr(L, "_ng_mode"), "call install() first"
        L._ng_mode = "stock" if beta_range == 2.0 else "erase"
        L._ng_range = float(beta_range)
        w = nn.Linear(L.hidden_size, L.num_v_heads, bias=False)
        nn.init.zeros_(w.weight)
        L.gate_w = w.to(device=L.b_proj.weight.device, dtype=torch.float32)
        params.append(L.gate_w.weight)
    return params


def enable_stats(model, on: bool = True) -> None:
    for L in gdn_layers(model):
        if on:
            dev = L.b_proj.weight.device
            z = lambda: torch.zeros(L.num_v_heads, dtype=torch.float64, device=dev)  # noqa: E731
            L._ng_stats = {"tokens": 0, "stock_gt1": z(), "beta_gt1": z(), "beta_sum": z(), "stock_sum": z()}
        else:
            L._ng_stats = None


def collect_stats(model) -> dict:
    """Per GDN layer and head: fraction of tokens with stock beta > 1, with the arm's beta > 1, mean betas."""
    out = {"model_layer": [], "tokens": [], "stock_gt1": [], "beta_gt1": [], "beta_mean": [], "stock_mean": []}
    for L in gdn_layers(model):
        st = L._ng_stats
        if not st or not st["tokens"]:
            continue
        n = st["tokens"]
        out["model_layer"].append(L.layer_idx)
        out["tokens"].append(n)
        for k_out, k in (("stock_gt1", "stock_gt1"), ("beta_gt1", "beta_gt1"), ("beta_mean", "beta_sum"),
                         ("stock_mean", "stock_sum")):
            out[k_out].append((st[k] / n).tolist())
    return out


def rank_heads(stats: dict, k: int) -> dict[int, list[int]]:
    """Top-k (gdn layer, head) pairs by stock beta > 1 frequency. Ties broken by (layer, head) order."""
    pairs = [(f, li, h) for li, row in enumerate(stats["stock_gt1"]) for h, f in enumerate(row)]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    out: dict[int, list[int]] = {}
    for _, li, h in pairs[:k]:
        out.setdefault(li, []).append(h)
    return {li: sorted(v) for li, v in out.items()}


def random_heads(stats: dict, k: int, seed: int, exclude: dict[int, list[int]] | None = None) -> dict[int, list[int]]:
    """k random (layer, head) pairs (matched-count control for the top-heads clamp), optionally avoiding a set."""
    import random

    ex = {(li, h) for li, hs in (exclude or {}).items() for h in hs}
    pairs = [(li, h) for li, row in enumerate(stats["stock_gt1"]) for h in range(len(row)) if (li, h) not in ex]
    pick = random.Random(seed).sample(pairs, k)
    out: dict[int, list[int]] = {}
    for li, h in pick:
        out.setdefault(li, []).append(h)
    return {li: sorted(v) for li, v in out.items()}
