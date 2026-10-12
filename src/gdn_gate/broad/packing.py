"""Sequence packing for the step 6 SFT trainer (train_sft.py --pack): same objective, no padding.

Each micro-batch is one row holding several whole examples back to back. Nothing crosses an example boundary:
  positions   RoPE position ids restart at 0 for every example (position_ids passed explicitly).
  attention   the full-attention layers run causal SDPA per example (the library's sdpa_attention_forward on
              each slice, is_causal, no mask), so no token attends to another example.
  GDN         the recurrent state restarts per example: cu_seqlens goes to fla's chunk_gated_delta_rule through
              the library's own `cu_seq_lens_q` hook; on the torch reference path (CPU tests) the wrapper runs
              the reference kernel once per example.
  conv        the short causal conv (width 4) sees zeros before every example, exactly as an unpacked row's left
              padding does. The library's torch fallback ignores sequence boundaries, so the wrapper inserts
              K - 1 zero columns before each example, convolves without padding and gathers the outputs back
              (same arithmetic as one example per row). backend 'fla' uses fla's varlen Triton conv instead;
              'cc1d' passes seq_idx to the causal_conv1d CUDA package when transformers resolved it (the
              package resets at seq_idx changes; opt-in only, held back lane-wide since 2026-10-07); default
              and 'auto' are torch.
Loss: the same (position, target) pairs as the padded path (each example's labels shifted by one inside the
example; the last position of an example predicts nothing), the same chunked full-vocab CE sum.

Without packing kwargs every wrapper delegates to the library function unchanged, so the padded path (val loss,
battery, test_cpu) runs exactly as before. RED (a set of component names) disables one reset at a time; the
boundary check (check_pack.py) uses it as the red arm and must see a leak for every component.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import arms  # noqa: F401  (sets WG_NO_FLA handling and imports patch before the library module is used)
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

ATTN_NAME = "wg_packed_sdpa"
RED: set[str] = set()  # check_pack.py only: components whose boundary reset is disabled ("pos","attn","gdn","conv")
CONV_BACKEND = "torch"  # torch (zero-insert around the library fallback) | fla (varlen Triton) | cc1d (opt-in only)
COUNTS = {"attn": 0, "gdn": 0, "conv": 0}  # packed calls seen; the checks assert every path engaged

_ORIG_CONV = M.causal_conv1d_fn
_ORIG_GDN = M.torch_chunk_gated_delta_rule
_FLA: bool | None = None
_CC1D: bool | None = None


def cc1d_active() -> bool:
    """True when transformers dispatches causal_conv1d_fn to the causal_conv1d package (installed before import)."""
    global _CC1D
    if _CC1D is None:
        cv = getattr(_ORIG_CONV, "__closure__", None) or ()
        impls = [c.cell_contents for c in cv if callable(getattr(c, "cell_contents", None))]
        _CC1D = any((getattr(f, "__module__", None) or "").startswith("causal_conv1d") for f in impls)
    return _CC1D


def fla_active() -> bool:
    global _FLA
    if _FLA is None:
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
            _FLA = True
        except Exception:  # noqa: BLE001  (WG_NO_FLA sets sys.modules["fla"] = None)
            _FLA = False
    return _FLA


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    bounds = kw.pop("wg_bounds", None)
    if bounds is None or "attn" in RED:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    assert query.shape[0] == 1 and attention_mask is None, (query.shape, attention_mask is None)
    COUNTS["attn"] += 1
    outs = [sdpa_attention_forward(module, query[:, :, s:e], key[:, :, s:e], value[:, :, s:e], None, dropout=dropout,
                                   scaling=scaling, **kw)[0] for s, e in bounds]
    return torch.cat(outs, dim=1), None


def _conv(hidden_states, weight, bias=None, activation=None, **kw):
    bounds = kw.get("wg_bounds")
    if bounds is None or "conv" in RED:
        return _ORIG_CONV(hidden_states, weight, bias, activation=activation, **kw)
    COUNTS["conv"] += 1
    B, D, T = hidden_states.shape
    assert B == 1, B
    backend = "torch" if CONV_BACKEND == "auto" else CONV_BACKEND  # auto never picks cc1d (held back lane-wide)
    if backend == "cc1d":
        assert cc1d_active(), "conv backend cc1d asked but transformers does not use the causal_conv1d package"
        return _ORIG_CONV(hidden_states, weight, bias, activation=activation, seq_idx=kw["wg_seq_idx"])
    if backend == "fla":
        from fla.modules.conv import causal_conv1d

        y, _ = causal_conv1d(hidden_states.transpose(1, 2), weight, bias, activation=activation,
                             cu_seqlens=kw["cu_seq_lens_q"], cu_seqlens_cpu=kw.get("cu_seqlens_cpu"))
        return y.transpose(1, 2)
    gidx, oidx = kw["wg_conv_idx"]
    x = torch.cat([hidden_states, hidden_states.new_zeros(B, D, 1)], dim=2).index_select(2, gidx)
    out = F.conv1d(x.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias, padding=0, groups=D).index_select(2, oidx)
    if activation is not None:
        out = M.ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def _gdn(query, key, value, *args, cu_seqlens=None, **kw):
    bounds = kw.get("wg_bounds")
    if bounds is not None:
        COUNTS["gdn"] += 1
    if bounds is None or "gdn" in RED:
        kw.pop("cu_seqlens_cpu", None)
        return _ORIG_GDN(query, key, value, *args, cu_seqlens=None, **kw)
    if fla_active():
        return _ORIG_GDN(query, key, value, *args, cu_seqlens=cu_seqlens, **kw)
    # torch reference kernel (CPU tests): one call per example, zero initial state each
    assert not args and kw.get("initial_state") is None and not kw.get("output_final_state"), "training path only"
    g, beta = kw.pop("g"), kw.pop("beta")
    outs = [_ORIG_GDN(query[:, s:e], key[:, s:e], value[:, s:e], g=g[:, s:e], beta=beta[:, s:e], **kw)[0]
            for s, e in bounds]
    return torch.cat(outs, dim=1), None


def install(model) -> None:
    """Route the model's attention through the packed-aware function and wrap conv / GDN in the library module.
    Idempotent. Without packing kwargs all three delegate unchanged."""
    AttentionInterface.register(ATTN_NAME, _attention)
    # the padded path must build the exact mask it builds under "sdpa" (an unregistered name gets no mask at all,
    # which is equal for right padding in exact arithmetic but routes SDPA to a different kernel)
    AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)
    M.causal_conv1d_fn = _conv
    M.torch_chunk_gated_delta_rule = _gdn
    seen = 0
    for mod in model.modules():
        if isinstance(mod, M.Qwen3_5Attention):
            assert mod.config._attn_implementation in ("sdpa", ATTN_NAME), mod.config._attn_implementation
            mod.config._attn_implementation = ATTN_NAME
            seen += 1
    assert seen, "no Qwen3_5Attention module found"


def groups(exs: list, micro_tokens: int) -> list[list]:
    """Split one optimizer step's examples into packed rows of at most micro_tokens real tokens: as few rows as
    fit, balanced by longest-first assignment to the least-loaded row (lower peak memory than first-fit)."""
    total = sum(len(e.ids) for e in exs)
    n = max(1, -(-total // micro_tokens))
    while True:
        rows, load = [[] for _ in range(n)], [0] * n
        ok = True
        for e in sorted(exs, key=lambda e: -len(e.ids)):
            i = min(range(n), key=lambda j: load[j])
            if load[i] + len(e.ids) > micro_tokens and load[i] > 0:
                ok = False
                break
            rows[i].append(e)
            load[i] += len(e.ids)
        if ok:
            return [r for r in rows if r]
        n += 1


def batch(exs: list, dev: torch.device, conv_k: int = 4) -> dict:
    """Packed inputs for one row: ids, targets, position ids and the boundary kwargs the wrappers read."""
    lens = [len(e.ids) for e in exs]
    T = sum(lens)
    ids = torch.tensor([t for e in exs for t in e.ids], dtype=torch.long)
    tgt = torch.tensor([t for e in exs for t in (e.labels[1:] + [-100])], dtype=torch.long)
    if "pos" in RED:
        pos = torch.arange(T)
    else:
        pos = torch.cat([torch.arange(n) for n in lens])
    cu = torch.tensor([0] + lens, dtype=torch.long).cumsum(0)
    bounds = [(int(cu[i]), int(cu[i + 1])) for i in range(len(lens))]
    gidx, oidx, p = [], [], 0
    for s, e in bounds:  # zero column (index T) K-1 times, then the example; outputs sit K-1 before each token
        gidx += [T] * (conv_k - 1)
        p += conv_k - 1
        gidx += range(s, e)
        oidx += range(p - (conv_k - 1), p - (conv_k - 1) + (e - s))
        p += e - s
    seq_idx = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lens)])[None].to(dev)
    kw = {"cu_seq_lens_q": cu.to(dev, torch.int32), "cu_seqlens_cpu": cu, "wg_bounds": bounds, "wg_seq_idx": seq_idx,
          "wg_conv_idx": (torch.tensor(gidx, device=dev), torch.tensor(oidx, device=dev))}
    return {"ids": ids[None].to(dev), "pos": pos[None].to(dev), "tgt": tgt.to(dev), "kw": kw, "T": T}


def hidden(model, b: dict) -> torch.Tensor:
    dec, _ = arms.core(model)
    return dec(input_ids=b["ids"], position_ids=b["pos"], attention_mask={"full_attention": None, "linear_attention": None},
               use_cache=False, **b["kw"]).last_hidden_state[0]


def loss_sum(model, exs, dev):
    """Packed twin of train_sft.sft_loss_sum: sum of next-token CE over label positions and their count."""
    _, head = arms.core(model)
    b = batch(exs, dev, conv_k=_conv_k(model))
    h = hidden(model, b)
    sel = b["tgt"] != -100
    hs = h[sel]
    y = b["tgt"][sel]

    def head_ce(x, t):
        return F.cross_entropy(head(x).float(), t, reduction="sum")

    total = sum(checkpoint(head_ce, hs[s : s + 1024], y[s : s + 1024], use_reentrant=False)
                for s in range(0, hs.shape[0], 1024))
    return total, int(y.numel())


def _conv_k(model) -> int:
    for mod in model.modules():
        if isinstance(mod, M.Qwen3_5GatedDeltaNet):
            return mod.conv_kernel_size
    return 4
