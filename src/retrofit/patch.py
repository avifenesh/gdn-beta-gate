"""Widened-beta retrofit for Qwen3.5 Gated DeltaNet layers.

The installed transformers forward is copied at runtime with exactly one line
changed: `beta = b.sigmoid()` becomes `beta = self._wg_beta(b, hidden_states)`.
Everything else (conv, gates, kernel dispatch to fla, cache handling) stays the
library's own code, pinned by the version assert below. One exception: inspect.getsource
unwraps the @force_accelerate_hooks("conv1d") decorator, so the patched forward runs without it.
That decorator only matters under accelerate offloading; this lane loads on one GPU
(device_map={"": 0}), and G0 shows the patched forward bit-identical to the library's at W = 0.
The cached single-token decode path also uses the widened beta but is not exercised or tested here.

Widened arm:  t = tanh(W x) per head, W zero-initialised,
              beta = s + t * (2 - s if t >= 0 else s),  s = sigmoid(b)
so beta = s exactly at W = 0, beta -> 2 as t -> 1, beta -> 0 as t -> -1, for
every token whatever its pretrained s. Control arm: the library forward, untouched.

Decay-gate arms (wideg = widened beta + decay gate, ctrlg = decay gate only; owner-approved horizon test
2026-09-29, beyond the card's method): the line computing the log-decay g is replaced as well, see _wg_g.

wide1 arm (the control for "gate parameters without the negative range", 2026-10-10): the wide arm's forward, W,
init and learning rate, with the ceiling at 1 instead of 2,
              beta = s + t * (1 - s if t >= 0 else s)
so beta = s exactly at W = 0 and beta stays in [0, 1] (the delta-rule eigenvalue 1 - beta never goes negative).
_wg_beta1 is the same function as gdn_gate/broad/arms._wg_beta1 (the step 6 definition); the CPU check
gdn_gate/w1main/test_w1main_cpu.py proves the two agree bit for bit. Its stats slot records the share of beta at
the ceiling (beta >= 0.999), as arms.py does, since wide1 has no beta > 1 by construction.
"""

from __future__ import annotations

import inspect
import textwrap
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

PINNED_TRANSFORMERS = "5.17.0"
TARGET_LINE = "beta = b.sigmoid()"
TARGET_G = "g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)"

# arm -> (widened beta gate, decay gate); wide1 = the widened gate with ceiling 1 (CEIL)
ARMS = {"ctrl": (False, False), "wide": (True, False), "wideg": (True, True), "ctrlg": (False, True),
        "wide1": (True, False)}
CEIL = {"wide": 2.0, "wideg": 2.0, "wide1": 1.0}


def _build_forward(beta: bool = True, decay: bool = False):
    assert transformers.__version__ == PINNED_TRANSFORMERS, transformers.__version__
    src = textwrap.dedent(inspect.getsource(M.Qwen3_5GatedDeltaNet.forward))
    assert src.count(TARGET_LINE) == 1 and src.count(TARGET_G) == 1, "upstream forward changed"
    if beta:
        src = src.replace(TARGET_LINE, "beta = self._wg_beta(b, hidden_states)")
    if decay:
        src = src.replace(TARGET_G, "g = self._wg_g(a, hidden_states)")
    ns: dict = {}
    exec(compile(src, "<wg_forward>", "exec"), M.__dict__, ns)
    return ns["forward"]


_FORWARDS: dict = {}


def _wg_beta(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    s = b.sigmoid()
    t = torch.tanh(self.gate_w(x.to(self.gate_w.weight.dtype)))
    sf = s.float()
    coef = torch.where(t >= 0, 2.0 - sf, sf)
    beta = (sf + t.float() * coef).to(s.dtype)
    stats = getattr(self, "_wg_stats", None)
    if stats is not None:
        with torch.no_grad():
            stats.append(((beta.float() > 1.0).float().mean().item(), t.abs().float().mean().item()))
    return beta


def _wg_beta1(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """wide1: _wg_beta with ceiling 1 instead of 2 (same W, same tanh, same casts)."""
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


def _wg_g(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Decay gate (horizon extension, not part of the card's method): the library's log-decay g0 <= 0 times
    (1 - u), u = tanh(W_a x) per head, W_a zero-initialised. alpha' = alpha^(1-u): exact at u = 0 (times 1.0),
    alpha' -> 1 as u -> 1, alpha' -> alpha^2 as u -> -1, with gradient -g0 at u = 0."""
    g0 = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    u = torch.tanh(self.wg_wa(x.to(self.wg_wa.weight.dtype))).float()
    stats = getattr(self, "_wg_gstats", None)
    if stats is not None:
        with torch.no_grad():
            stats.append(((u > 0.5).float().mean().item(), u.abs().mean().item()))
    return g0 * (1.0 - u)


def gdn_layers(model) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, M.Qwen3_5GatedDeltaNet)]


def install(model, beta: bool = True, decay: bool = False) -> list[nn.Parameter]:
    """Add the zero-init gates to every GDN layer. Returns the new parameters."""
    if not (beta or decay):
        return []
    key = (beta, decay)
    if key not in _FORWARDS:
        _FORWARDS[key] = _build_forward(beta, decay)
    params = []
    for layer in gdn_layers(model):
        dev = layer.in_proj_b.weight.device
        if beta:
            w = nn.Linear(layer.hidden_size, layer.num_v_heads, bias=False)
            nn.init.zeros_(w.weight)
            layer.gate_w = w.to(device=dev, dtype=torch.float32)
            layer._wg_beta = types.MethodType(_wg_beta, layer)
            params.append(layer.gate_w.weight)
        if decay:
            wa = nn.Linear(layer.hidden_size, layer.num_v_heads, bias=False)
            nn.init.zeros_(wa.weight)
            layer.wg_wa = wa.to(device=dev, dtype=torch.float32)
            layer._wg_g = types.MethodType(_wg_g, layer)
            params.append(layer.wg_wa.weight)
        layer.forward = types.MethodType(_FORWARDS[key], layer)
    return params


def install_widened(model) -> list[nn.Parameter]:
    """The card's widened-beta arm."""
    return install(model, beta=True, decay=False)


def install_for_arm(model, arm: str) -> list[nn.Parameter]:
    beta, decay = ARMS[arm]
    params = install(model, beta=beta, decay=decay)
    if arm == "wide1":  # the same forward and W as wide; only the bound beta function differs
        for layer in gdn_layers(model):
            layer._wg_beta = types.MethodType(_wg_beta1, layer)
    return params


def enable_stats(model, on: bool = True) -> None:
    for layer in gdn_layers(model):
        if hasattr(layer, "gate_w"):
            layer._wg_stats = [] if on else None
        if hasattr(layer, "wg_wa"):
            layer._wg_gstats = [] if on else None


def collect_stats(model) -> dict:
    frac, mag = [], []
    for layer in gdn_layers(model):
        st = getattr(layer, "_wg_stats", None) or []
        if st:
            frac.append(sum(a for a, _ in st) / len(st))
            mag.append(sum(b for _, b in st) / len(st))
    ufrac, umag = [], []
    for layer in gdn_layers(model):
        st = getattr(layer, "_wg_gstats", None) or []
        if st:
            ufrac.append(sum(a for a, _ in st) / len(st))
            umag.append(sum(b for _, b in st) / len(st))
    return {"beta_gt1_frac_per_layer": frac, "abs_t_per_layer": mag,
            "decay_u_gt05_frac_per_layer": ufrac, "abs_u_per_layer": umag}
