"""β gate for Qwen3.5 Gated DeltaNet layers under vLLM 0.31.0, installed in-process (no site-packages edit).

Arms (same arithmetic as retrofit/patch.py and gdn_gate/broad/arms.py, s = sigmoid(b) rounded to b's
dtype, t = tanh(W x) in fp32, W fp32 [num_v_heads, hidden], x = the normalised hidden state fed to the layer):
  wide    beta = s + t * (2 - s if t >= 0 else s)    range (0, 2)
  wide1   beta = s + t * (1 - s if t >= 0 else s)    range (0, 1)
  stock   vLLM untouched (ctrl arm, strip twins, base)
Decay gate (retrofit arms wideg / ctrlg): g = g0 * (1 - u), u = tanh(W_a x), g0 = -exp(A_log) softplus(a + dt_bias),
the same fp32 arithmetic as patch._wg_g.

How it is wired (install() runs in the engine process: VLLM_ENABLE_V1_MULTIPROCESSING=0, before the LLM is built):
* QwenGatedDeltaNetAttention.__init__ is wrapped: each layer gets fp32 parameters gate_w (and wg_wa with the
  decay gate), zero filled, A_log is marked, the fused CUDA decode kernel is switched off for the layer, and a layer
  whose prefill backend is not Triton/FLA is refused (pass additional_config={"gdn_prefill_backend": "triton"}).
* QwenGatedDeltaNetAttention.forward_cuda is rebuilt from its own source with one inserted statement after
  `b, a = self.split_ba(ba)`: `b, a = self._wg_apply(hidden_states, b, a)`. The producer is one opaque torch
  custom op (gate_vllm::gate) so torch.compile and CUDA graphs see a single node. Its outputs are the FINAL beta
  (dtype of b) in the b slot and, with the decay gate, the FINAL fp32 g in the a slot.
* The three Triton kernels that turn (a, b) into (g, beta) for this layer type are swapped, in the
  qwen_gdn_linear_attn namespace, for source-rewritten copies with the sigmoid (and with the decay gate, the g
  formula) replaced by a passthrough; a wrapper routes marked layers to the copy and every other layer to the stock
  kernel: fused_post_conv_prep (prefill), fused_sigmoid_gating_delta_rule_update (decode, mixed batches, spec),
  fused_recurrent_gated_delta_rule_packed_decode (packed decode). The fused CUDA decode path
  (_forward_core_fused_norm_packed) and the ROCm AITER path raise for a marked layer.
* Unmerged LoRA (install(..., lora_rank=r), any arm including stock): each GDN layer gets fp32 parameters for the
  LoRA of the four trained projections (in_proj_qkv, in_proj_b, in_proj_a, out_proj), zero filled. forward_cuda adds
  scale * B(A x) to the fused in_proj_qkvz output (qkv rows only; z untouched) and to the in_proj_ba output right
  after the projections, and _output_projection adds it to out_proj's output, the arithmetic PEFT runs in the
  trainer (scale = alpha / r = 2). set_lora() copies A, B and the gate tensors in place every step: addresses stay
  static, so the compiled graph and CUDA graphs stay valid, and a sync is a copy. The fused CUDA decode path is
  switched off for these layers (its out_proj call has no LoRA). This is the RL weight sync and the readout path:
  a merged bf16 weight keeps only a few percent of a small LoRA move (delta_kept 0.013-0.040 on the first GRPO steps,
  workstation smoke 2026-10-08), so the rollout policy would lag the trained one.
* Merged weights (set_checkpoint, kept for SFT-size deltas only): W + scale B A rounded to bf16, written in place,
  with the gate tensors. Layout is checked against the base safetensors on first use (the fused vLLM parameters must
  equal the concatenated base tensors exactly); the unmerged LoRA relies on the same [qkv; z] and [b; a] layout.

Pinned: vLLM 0.31.0 (file hashes of the rewritten sources in SOURCE_MD5; the same files ship in the workstation's
0.30.1rc1.dev260 build). Any other version is refused unless WG_ALLOW_VLLM_VERSION=1.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PINNED = ("0.31.0", "0.30.1rc1.dev260+g7ba3df63c")
SOURCE_MD5 = {
    "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py": "27c70d476957",
    "third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py": "dee0c9b4814c",
    "third_party/flash_linear_attention/ops/fused_sigmoid_gating.py": "42413781831f",
    "third_party/flash_linear_attention/ops/fused_recurrent.py": "feea3e24efdb",
}
MOD_Q = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
OPS = "vllm.third_party.flash_linear_attention.ops"

FWD_ANCHOR = "b, a = self.split_ba(ba)"
LORA_ANCHOR = "ba, _ = self.in_proj_ba(hidden_states)"
OUT_ANCHOR = "output, _ = self.out_proj(core_attn_out.flatten(-2))"
LORA_SCALE = 2.0  # alpha / r with alpha = 2 r (broad/arms.add_lora)
# (module, function, [(old, new_for_beta)], [(old, new_for_decay)])
KERNELS = {
    "post_conv": (
        f"{OPS}.fused_gdn_prefill_post_conv", "fused_post_conv_prep",
        [("beta_vals = tl.sigmoid(b_vals)", "beta_vals = b_vals")],
        [("g_vals = -tl.exp(A_log_val) * sp", "g_vals = a_vals")],
    ),
    "sigmoid_gating": (
        f"{OPS}.fused_sigmoid_gating", "fused_sigmoid_gating_delta_rule_update",
        [("b_beta = tl.sigmoid(b_b.to(tl.float32))", "b_beta = b_b.to(tl.float32)")],
        [("b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x", "b_g = tl.load(p_a).to(tl.float32)")],
    ),
    "packed_decode": (
        f"{OPS}.fused_recurrent", "fused_recurrent_gated_delta_rule_packed_decode",
        [("beta_val = tl.sigmoid(b_val)", "beta_val = b_val")],
        [("g_val = -tl.exp(A_log_val) * softplus_x", "g_val = a_val")],
    ),
}
CEIL = {"wide": 2.0, "wide1": 1.0}
_STATE: dict = {"installed": None}
STATS = {"on": False, "calls": 0, "beta_gt1": 0, "beta_n": 0, "beta_max": 0.0, "beta_min": 9.0, "u_abs_sum": 0.0}
CALLS = {"gate": 0, "post_conv": 0, "sigmoid_gating": 0, "packed_decode": 0}


class GateError(RuntimeError):
    pass


def _say(msg: str) -> None:
    print(f"gate_vllm: {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ arithmetic
@contextlib.contextmanager
def _highest():
    prev = torch.get_float32_matmul_precision()
    if prev != "highest":
        torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        if prev != "highest":
            torch.set_float32_matmul_precision(prev)


def beta_ref(b: torch.Tensor, x: torch.Tensor, w: torch.Tensor, ceil: float) -> torch.Tensor:
    """patch._wg_beta (ceil 2) / arms._wg_beta1 (ceil 1), line for line."""
    s = b.sigmoid()
    with _highest():
        t = torch.tanh(F.linear(x.to(w.dtype), w))
    sf = s.float()
    coef = torch.where(t >= 0, ceil - sf, sf)
    return (sf + t.float() * coef).to(s.dtype)


def g_ref(a, x, wa, A_log, dt_bias) -> torch.Tensor:
    """patch._wg_g, line for line (fp32 result)."""
    g0 = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    with _highest():
        u = torch.tanh(F.linear(x.to(wa.dtype), wa)).float()
    return g0 * (1.0 - u)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


@torch.library.custom_op("gate_vllm::gate", mutates_args=())
def gate_op(b: torch.Tensor, a: torch.Tensor, x: torch.Tensor, w: torch.Tensor, wa: torch.Tensor,
            A_log: torch.Tensor, dt_bias: torch.Tensor, ceil: float, beta_on: bool,
            decay_on: bool) -> tuple[torch.Tensor, torch.Tensor]:
    CALLS["gate"] += 1
    nb = beta_ref(b, x, w, ceil).contiguous() if beta_on else b.contiguous().clone()
    na = g_ref(a, x, wa, A_log, dt_bias).contiguous() if decay_on else a.contiguous().clone()
    if STATS["on"] and not _capturing():
        with torch.no_grad():
            if beta_on:
                bf = nb.float()
                STATS["beta_gt1"] += int((bf > 1.0).sum())
                STATS["beta_n"] += bf.numel()
                STATS["beta_max"] = max(STATS["beta_max"], float(bf.max())) if bf.numel() else STATS["beta_max"]
                STATS["beta_min"] = min(STATS["beta_min"], float(bf.min())) if bf.numel() else STATS["beta_min"]
            STATS["calls"] += 1
    return nb, na


@gate_op.register_fake
def _gate_fake(b, a, x, w, wa, A_log, dt_bias, ceil, beta_on, decay_on):
    return torch.empty_like(b, memory_format=torch.contiguous_format), (
        torch.empty(a.shape, dtype=torch.float32, device=a.device) if decay_on
        else torch.empty_like(a, memory_format=torch.contiguous_format))


# ------------------------------------------------------------------------------------------------ source rewrite
def _vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    return Path(list(spec.submodule_search_locations)[0])


def source_hashes() -> dict:
    root = _vllm_root()
    return {k: hashlib.md5((root / k).read_bytes()).hexdigest()[:12] for k in SOURCE_MD5}


def rewrite_forward(src: str) -> str:
    src = textwrap.dedent(src)
    lines = src.rstrip("\n").split("\n")
    out, n, nl = [], 0, 0
    for ln in lines:
        out.append(ln)
        ind = ln[: len(ln) - len(ln.lstrip())]
        if ln.strip() == FWD_ANCHOR:
            out.append(ind + 'if getattr(self, "_wg", False):')
            out.append(ind + "    b, a = self._wg_apply(hidden_states, b, a)")
            n += 1
        if ln.strip() == LORA_ANCHOR:
            out.append(ind + 'if getattr(self, "_wg_lora", False):')
            out.append(ind + "    mixed_qkvz, ba = self._wg_lora_in(hidden_states, mixed_qkvz, ba)")
            nl += 1
    if n != 1 or nl != 1:
        raise GateError(f"forward_cuda anchors {FWD_ANCHOR!r} / {LORA_ANCHOR!r} found {n} / {nl} times, expected 1")
    return "\n".join(out) + "\n"


def rewrite_output_projection(src: str) -> str:
    src = textwrap.dedent(src)
    if src.count(OUT_ANCHOR) != 1:
        raise GateError(f"_output_projection anchor {OUT_ANCHOR!r} found {src.count(OUT_ANCHOR)} times, expected 1")
    lines = []
    for ln in src.rstrip("\n").split("\n"):
        if ln.strip() == OUT_ANCHOR:
            ind = ln[: len(ln) - len(ln.lstrip())]
            lines += [ind + "_wg_y = core_attn_out.flatten(-2)", ind + "output, _ = self.out_proj(_wg_y)",
                      ind + 'if getattr(self, "_wg_lora", False):',
                      ind + "    output = self._wg_lora_out(_wg_y, output)"]
        else:
            lines.append(ln)
    return "\n".join(lines) + "\n"


def rewrite_kernel(src: str, subs) -> str:
    for old, new in subs:
        c = src.count(old)
        if c != 1:
            raise GateError(f"kernel rewrite target found {c} times, expected 1: {old!r}")
        src = src.replace(old, new)
    return src


def _cache_dir() -> Path:
    p = Path(os.environ.get("WG_CACHE_DIR", str(Path.home() / ".cache/gdn_gate/vllm/gen")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_rewritten(modname: str, subs, tag: str):
    orig = importlib.import_module(modname)
    text = f"# gate_vllm passthrough copy of {modname} (generated)\n" + rewrite_kernel(Path(orig.__file__).read_text(), subs)
    # relative imports inside the copy must resolve against the original package
    text = re.sub(r"^from \.(\w)", lambda m: f"from {modname.rsplit('.', 1)[0]}.{m.group(1)}", text, flags=re.M)
    text = re.sub(r"^from \. import", f"from {modname.rsplit('.', 1)[0]} import", text, flags=re.M)
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = _cache_dir() / f"wg_{tag}_{h}.py"
    if not path.exists():
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, path)
    name = f"gate_vllm_gen_{tag}_{h}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _a_log_of(args, kwargs, pos):
    if "A_log" in kwargs:
        return kwargs["A_log"]
    return args[pos] if len(args) > pos else None


def static_check() -> dict:
    """CPU: every rewrite target exists exactly once in the installed vLLM and the rewritten sources compile."""
    root = _vllm_root()
    import ast

    qsrc = (root / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py").read_text()
    tree = ast.parse(qsrc)
    seg = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "QwenGatedDeltaNetAttention":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "forward_cuda":
                    assert not fn.decorator_list, "forward_cuda is decorated upstream"
                    seg = textwrap.dedent(" " * fn.col_offset + ast.get_source_segment(qsrc, fn))
    assert seg is not None, "forward_cuda not found"
    compile(rewrite_forward(seg), "<fwd>", "exec")
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "QwenGatedDeltaNetAttention":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_output_projection":
                    seg2 = textwrap.dedent(" " * fn.col_offset + ast.get_source_segment(qsrc, fn))
                    compile(rewrite_output_projection(seg2), "<outproj>", "exec")
    out = {"vllm_root": str(root), "hashes": source_hashes(), "expected": SOURCE_MD5}
    for tag, (mod, fn, bsubs, gsubs) in KERNELS.items():
        rel = mod.replace("vllm.", "", 1).replace(".", "/") + ".py"
        src = (root / rel).read_text()
        compile(rewrite_kernel(src, bsubs + gsubs), rel, "exec")
        out[tag] = "ok"
    out["hash_match"] = out["hashes"] == SOURCE_MD5
    return out


# ------------------------------------------------------------------------------------------------ install
def install(arm: str, decay: bool = False, lora_rank: int = 0) -> None:
    """arm in {wide, wide1, stock}; decay adds the retrofit decay gate (wideg = wide + decay, ctrlg = stock + decay);
    lora_rank > 0 adds the unmerged LoRA of that rank on every GDN layer (set_lora), for any arm."""
    key = (arm, decay, lora_rank)
    if _STATE["installed"] is not None:
        if _STATE["installed"] != key:
            raise GateError(f"already installed as {_STATE['installed']}, asked {key} (one arm per process)")
        return
    if arm not in ("wide", "wide1", "stock"):
        raise GateError(f"unknown arm {arm}")
    set_vllm_cache_root(arm, decay, lora_rank)
    if arm == "stock" and not decay and not lora_rank:
        _STATE["installed"] = key
        return
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise GateError("set VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing vllm (the patch lives in this process)")
    import vllm

    if vllm.__version__ not in PINNED and os.environ.get("WG_ALLOW_VLLM_VERSION") != "1":
        raise GateError(f"pinned to vLLM {PINNED}, found {vllm.__version__}")
    hs = source_hashes()
    if hs != SOURCE_MD5 and os.environ.get("WG_ALLOW_VLLM_VERSION") != "1":
        raise GateError(f"rewritten vLLM sources changed: {hs} vs {SOURCE_MD5}")
    beta_on = arm != "stock"
    gate_on = beta_on or decay
    ceil = CEIL.get(arm, 2.0)
    Q = importlib.import_module(MOD_Q)
    cls = Q.QwenGatedDeltaNetAttention

    # kernels (gate arms only; a LoRA-only engine keeps the stock kernels)
    for tag, (mod, fn, bsubs, gsubs) in (KERNELS.items() if gate_on else ()):
        subs = (bsubs if beta_on else []) + (gsubs if decay else [])
        patched = getattr(_load_rewritten(mod, subs, f"{tag}_{int(beta_on)}{int(decay)}"), fn)
        orig = getattr(Q, fn)
        pos = {"post_conv": 3, "sigmoid_gating": 0, "packed_decode": 3}[tag]  # call sites pass A_log by keyword

        def make(orig, patched, tag, pos):
            @functools.wraps(orig)
            def wrapper(*args, **kwargs):
                A = _a_log_of(args, kwargs, pos)
                if getattr(A, "_wg", False):
                    CALLS[tag] += 1
                    return patched(*args, **kwargs)
                return orig(*args, **kwargs)

            wrapper._wg_wrapped = True
            return wrapper

        setattr(Q, fn, make(orig, patched, tag, pos))

    # paths that cannot take the final beta/g
    for meth in ("_forward_core_fused_norm_packed", "_forward_core_rocm", "_forward_core_decode_aiter"):
        if hasattr(cls, meth):
            orig_m = getattr(cls, meth)

            def refuse(orig_m, meth):
                @functools.wraps(orig_m)
                def f(self, *a, **k):
                    if getattr(self, "_wg", False) or getattr(self, "_wg_lora", False):
                        raise GateError(f"{meth} bypasses the gate / LoRA insertions; it cannot serve a marked layer")
                    return orig_m(self, *a, **k)

                return f

            setattr(cls, meth, refuse(orig_m, meth))

    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if self.gqa_interleaved_layout:
            raise GateError("gqa_interleaved_layout (Qwen3-Next) is not covered by this patch")
        nv = self.num_v_heads // self.tp_size
        dev = self.A_log.device
        f32 = dict(dtype=torch.float32, device=dev)
        if gate_on:
            if self.chunk_gated_delta_rule.gdn_prefill_backend != "triton":
                raise GateError("gated layers need the Triton/FLA prefill kernel: "
                                  "LLM(..., additional_config={'gdn_prefill_backend': 'triton'})")
            self.gate_w = nn.Parameter(torch.zeros(nv, self.hidden_size, **f32), requires_grad=False)
            self.wg_wa = nn.Parameter(torch.zeros(nv if decay else 1, self.hidden_size if decay else 1, **f32),
                                          requires_grad=False)
            self.A_log._wg = True
            self._wg = True
        if lora_rank:
            if self.tp_size != 1:
                raise GateError("the unmerged LoRA covers tensor parallel size 1 only")
            r, H = lora_rank, self.hidden_size
            qkv = self.key_dim * 2 + self.value_dim
            self.wg_lA = nn.Parameter(torch.zeros(3 * r, H, **f32), requires_grad=False)  # [A_qkv; A_b; A_a]
            self.wg_lBqkv = nn.Parameter(torch.zeros(qkv, r, **f32), requires_grad=False)
            self.wg_lBb = nn.Parameter(torch.zeros(nv, r, **f32), requires_grad=False)
            self.wg_lBa = nn.Parameter(torch.zeros(nv, r, **f32), requires_grad=False)
            self.wg_lAo = nn.Parameter(torch.zeros(r, self.value_dim, **f32), requires_grad=False)
            self.wg_lBo = nn.Parameter(torch.zeros(H, r, **f32), requires_grad=False)
            self._wg_r = r
            self._wg_lora = True
        self.enable_fused_gdn_decode = False

    def _wg_apply(self, x, b, a):
        return torch.ops.gate_vllm.gate(b, a, x, self.gate_w, self.wg_wa, self.A_log, self.dt_bias,
                                          ceil, beta_on, decay)

    def _wg_lora_in(self, x, qkvz, ba):
        r = self._wg_r
        xa = F.linear(x.float(), self.wg_lA)
        dq = F.linear(xa[:, :r], self.wg_lBqkv) * LORA_SCALE
        dba = torch.cat([F.linear(xa[:, r : 2 * r], self.wg_lBb), F.linear(xa[:, 2 * r :], self.wg_lBa)], -1)
        qkvz = qkvz + F.pad(dq, (0, qkvz.shape[-1] - dq.shape[-1])).to(qkvz.dtype)
        return qkvz, ba + (dba * LORA_SCALE).to(ba.dtype)

    def _wg_lora_out(self, y, out):
        return out + (F.linear(F.linear(y.float(), self.wg_lAo), self.wg_lBo) * LORA_SCALE).to(out.dtype)

    src = inspect.getsource(cls.forward_cuda)
    new_src = rewrite_forward(src)
    ns: dict = {}
    exec(compile(new_src, str(_cache_dir() / "wg_forward_cuda.py"), "exec"), Q.__dict__, ns)
    cls.__init__ = __init__
    cls._wg_apply = _wg_apply
    cls.forward_cuda = ns["forward_cuda"]
    if lora_rank:
        ns2: dict = {}
        exec(compile(rewrite_output_projection(inspect.getsource(cls._output_projection)),
                     str(_cache_dir() / "wg_output_projection.py"), "exec"), Q.__dict__, ns2)
        cls._wg_lora_in = _wg_lora_in
        cls._wg_lora_out = _wg_lora_out
        cls._output_projection = ns2["_output_projection"]
    # the model loader must not complain about our parameters being absent from the checkpoint
    _patch_loader_skip()
    _STATE["installed"] = key
    _say(f"installed arm={arm} decay={decay} ceil={ceil} lora_rank={lora_rank} on vLLM {vllm.__version__}")


def _patch_loader_skip():
    """vLLM's default loader checks that every parameter was loaded; ours come from set_checkpoint()."""
    try:
        from vllm.model_executor.model_loader import default_loader as DL
    except Exception:
        return
    cls = DL.DefaultModelLoader
    if getattr(cls.load_weights, "_wg_wrapped", False):
        return
    orig = cls.load_weights

    def load_weights(self, model, model_config):
        orig_lw = model.load_weights

        def lw(weights):
            loaded = orig_lw(weights)
            if loaded is not None:
                for n, _ in model.named_parameters():
                    if ".wg_" in n:
                        loaded.add(n)
            return loaded

        model.load_weights = lw
        try:
            return orig(self, model, model_config)
        finally:
            del model.load_weights

    load_weights._wg_wrapped = True
    cls.load_weights = load_weights


# ------------------------------------------------------------------------------------------------ engine helpers
def engine_kwargs(gated: bool, **over) -> dict:
    kw = dict(dtype="bfloat16", enable_prefix_caching=False, mamba_ssm_cache_dtype="float32",
              additional_config={"gdn_prefill_backend": "triton"}, limit_mm_per_prompt={"image": 0, "video": 0},
              max_logprobs=256, seed=0, max_num_seqs=128)  # max_logprobs: MMLU-Pro cloze stride 4 has an item with 75 option tokens (2026-10-10). GDN needs one Mamba cache block per sequence: 256 exceeded the 247 blocks on the workstation (2026-10-08)
    kw.update(over)
    return kw


def set_vllm_cache_root(arm: str, decay: bool, lora_rank: int = 0) -> str:
    """vLLM's compile caches (the standalone/AOT torch.compile artifacts under VLLM_CACHE_ROOT/torch_compile_cache)
    are keyed by vLLM's own sources and config, not by this in-process patch: on the workstation (2026-10-08) every gated
    engine started after a stock one loaded the STOCK compiled graph (no gate op; the passthrough kernels then read
    raw logits as beta: NaN). Each (arm, decay, plugin source) therefore gets its own VLLM_CACHE_ROOT, set before
    vLLM is imported. WG_VLLM_CACHE overrides the parent directory."""
    h = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:10]
    parent = Path(os.environ.get("WG_VLLM_CACHE", str(Path.home() / ".cache/gdn_gate/vllm/vllm_cache")))
    d = parent / (f"{arm}_{int(decay)}_{h}" + (f"_l{lora_rank}" if lora_rank else ""))
    d.mkdir(parents=True, exist_ok=True)
    if "vllm" in sys.modules and os.environ.get("VLLM_CACHE_ROOT") != str(d):
        raise GateError("set_vllm_cache_root must run before vllm is imported")
    os.environ["VLLM_CACHE_ROOT"] = str(d)
    return str(d)


def get_model(llm):
    eng = llm.llm_engine
    for path in ("model_executor.driver_worker.worker.model_runner.model",
                 "engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model"):
        obj = eng
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise GateError("cannot reach the model object (is VLLM_ENABLE_V1_MULTIPROCESSING=0?)")


def gdn_layers(model) -> dict:
    Q = importlib.import_module(MOD_Q)
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, Q.QwenGatedDeltaNetAttention):
            mm = re.search(r"layers\.(\d+)\.linear_attn$", name)
            assert mm, name
            out[int(mm.group(1))] = m
    return out


LORA_KEY = re.compile(r"layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_a|in_proj_b|out_proj)\.lora_([AB])(?:\.[^.]+)?\.weight$")
GATE_KEY = re.compile(r"layers\.(\d+)\.linear_attn\.(gate_w|wg_wa)(?:\.weight)?$")


class BaseWeights:
    """CPU copies of the base tensors that LoRA touches, read once from the checkpoint safetensors."""

    def __init__(self, model_dir: str, layers):
        from safetensors import safe_open

        idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
        self.t = {}
        want = {}
        for lid in layers:
            for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                hits = [k for k in idx if k.endswith(f"layers.{lid}.linear_attn.{p}.weight") and "mtp" not in k
                        and "visual" not in k]
                assert len(hits) == 1, (lid, p, hits)
                want[(lid, p)] = hits[0]
        byf: dict = {}
        for key, name in want.items():
            byf.setdefault(idx[name], []).append((key, name))
        for f, lst in byf.items():
            with safe_open(str(Path(model_dir) / f), framework="pt", device="cpu") as fh:
                for key, name in lst:
                    self.t[key] = fh.get_tensor(name)


def _flat(obj):
    if isinstance(obj, dict) and "trainable" in obj and isinstance(obj["trainable"], dict):
        return obj["trainable"]
    return obj


@torch.no_grad()
def check_layout(model, base: BaseWeights) -> dict:
    """On a freshly loaded engine: every fused GDN parameter must equal the concatenated base tensors bit for bit
    ([qkv; z] for in_proj_qkvz, [b; a] for in_proj_ba, out_proj as is). Fails if vLLM fuses differently."""
    n = 0
    for lid, m in gdn_layers(model).items():
        W = {p: base.t[(lid, p)] for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")}
        want_qkvz = torch.cat([W["in_proj_qkv"], W["in_proj_z"]])
        want_ba = torch.cat([W["in_proj_b"], W["in_proj_a"]])
        for name, got, want in (("in_proj_qkvz", m.in_proj_qkvz.weight, want_qkvz),
                                ("in_proj_ba", m.in_proj_ba.weight, want_ba), ("out_proj", m.out_proj.weight, W["out_proj"])):
            if tuple(got.shape) != tuple(want.shape) or not torch.equal(got.detach().cpu(), want):
                raise GateError(f"layer {lid} {name}: vLLM parameter {tuple(got.shape)} != base layout {tuple(want.shape)}")
            n += 1
        m._wg_layout_ok = True
    return {"layout_tensors_checked": n, "gdn_layers": len(gdn_layers(model))}


LORA_SLOTS = {"in_proj_qkv": ("A", 0, "wg_lBqkv"), "in_proj_b": ("A", 1, "wg_lBb"),
              "in_proj_a": ("A", 2, "wg_lBa"), "out_proj": ("wg_lAo", None, "wg_lBo")}


@torch.no_grad()
def set_lora(model, state: dict | None, *, gate: bool, lora: bool = True, decay: bool = False) -> dict:
    """Write the unmerged LoRA (A, B per trained projection) and the gate tensors in place (install(lora_rank=r)).
    state None = the base model. lora False = LoRA zeroed (red arm), gate False = gate zeroed (strip / red arm).
    Missing LoRA modules are zero; a half pair, a rank or shape mismatch, or a LoRA outside the GDN projections
    raises."""
    layers = gdn_layers(model)
    state = _flat(state) if state is not None else {}
    lo: dict = {}
    gates: dict = {}
    for k, v in state.items():
        m = LORA_KEY.search(k)
        if m:
            lo.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v
            continue
        g = GATE_KEY.search(k)
        if g:
            gates[(int(g.group(1)), g.group(2))] = v
            continue
        if "lora_" in k:
            raise GateError(f"LoRA tensor outside the GDN projections: {k}")
    bad = [k for k, v in lo.items() if set(v) != {"A", "B"} or k[0] not in layers]
    if bad:
        raise GateError(f"unpaired or unknown LoRA modules: {bad[:4]}")
    n_lora = 0
    for lid, m in layers.items():
        if not getattr(m, "_wg_lora", False):
            raise GateError("set_lora needs an engine installed with lora_rank > 0")
        r, dev = m._wg_r, m.wg_lA.device
        for p, (aslot, ai, bslot) in LORA_SLOTS.items():
            pair = lo.get((lid, p)) if lora else None
            Bp = getattr(m, bslot)
            Ap = m.wg_lA[ai * r : (ai + 1) * r] if aslot == "A" else getattr(m, aslot)
            if pair is None:
                Ap.zero_()
                Bp.zero_()
                continue
            A, B = pair["A"].float(), pair["B"].float()
            if tuple(A.shape) != tuple(Ap.shape) or tuple(B.shape) != tuple(Bp.shape):
                raise GateError(f"layer {lid} {p}: LoRA A {tuple(A.shape)} B {tuple(B.shape)} vs engine "
                                  f"{tuple(Ap.shape)} {tuple(Bp.shape)} (rank {r})")
            Ap.copy_(A.to(dev, non_blocking=True))
            Bp.copy_(B.to(dev, non_blocking=True))
            n_lora += 1
        if hasattr(m, "gate_w"):
            gw = gates.get((lid, "gate_w")) if gate else None
            if gate and gw is None and _STATE["installed"][0] != "stock":
                raise GateError(f"layer {lid}: gated arm asked but the state has no gate_w")
            m.gate_w.copy_(gw.float().to(dev) if gw is not None else torch.zeros_like(m.gate_w))
            if decay:
                ga = gates.get((lid, "wg_wa"))
                if ga is None:
                    raise GateError(f"layer {lid}: decay gate asked but the state has no wg_wa")
                m.wg_wa.copy_(ga.float().to(dev))
        elif gate and any(k[0] == lid for k in gates):
            raise GateError(f"layer {lid}: the state carries gates but this engine is stock")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"lora_modules": n_lora, "gate_layers": sum(1 for k in gates if k[1] == "gate_w") if gate else 0}


@torch.no_grad()
def set_checkpoint(model, base: BaseWeights, state: dict | None, *, gate: bool, decay: bool, scale: float = 2.0,
                   rounding: str = "nearest", seed: int = 20261008, check_layout: bool = True) -> dict:
    """Write base + scale*B@A into the fused projections and the gate tensors, in place.
    state None = the base model (no LoRA, zero gates). gate False = the LoRA alone (strip twin / ctrl).
    Returns a receipt: tensors written, delta_kept (projection of the bf16 move on the intended delta)."""
    layers = gdn_layers(model)
    state = _flat(state) if state is not None else {}
    lora: dict = {}
    gates: dict = {}
    for k, v in state.items():
        m = LORA_KEY.search(k)
        if m:
            lora.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v.float()
            continue
        g = GATE_KEY.search(k)
        if g:
            gates[(int(g.group(1)), g.group(2))] = v.float()
            continue
        if "lora_" in k:
            raise GateError(f"LoRA tensor outside the GDN projections: {k}")
    bad = [k for k, v in lora.items() if set(v) != {"A", "B"} or k[0] not in layers]
    if bad:
        raise GateError(f"unpaired or unknown LoRA modules: {bad[:4]}")
    gen = torch.Generator().manual_seed(seed)
    num = den = 0.0
    written = 0
    for lid, m in layers.items():
        W = {p: base.t[(lid, p)] for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")}
        qkv_rows, nv = W["in_proj_qkv"].shape[0], W["in_proj_b"].shape[0]
        dev = m.in_proj_qkvz.weight.device
        if check_layout and not getattr(m, "_wg_layout_ok", False):
            raise GateError(f"layer {lid}: call check_layout(model, base) on the fresh engine first")
        merged = {}
        for p in ("in_proj_qkv", "in_proj_b", "in_proj_a", "out_proj"):
            w = W[p]
            if (lid, p) in lora:
                d = scale * (lora[(lid, p)]["B"] @ lora[(lid, p)]["A"])
                if d.shape != w.shape:
                    raise GateError(f"layer {lid} {p}: delta {tuple(d.shape)} vs weight {tuple(w.shape)}")
                s = w.float() + d
                if rounding == "stochastic":
                    bits = s.contiguous().view(torch.int32)
                    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, generator=gen)
                    nw = ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)
                else:
                    nw = s.to(torch.bfloat16)
                num += float(((nw.float() - w.float()) * d).sum())
                den += float(d.pow(2).sum())
                merged[p] = nw
                written += 1
            else:
                merged[p] = w
        m.in_proj_qkvz.weight[:qkv_rows].copy_(merged["in_proj_qkv"].to(dev))
        m.in_proj_qkvz.weight[qkv_rows:].copy_(W["in_proj_z"].to(dev))
        m.in_proj_ba.weight[:nv].copy_(merged["in_proj_b"].to(dev))
        m.in_proj_ba.weight[nv:].copy_(merged["in_proj_a"].to(dev))
        m.out_proj.weight.copy_(merged["out_proj"].to(dev))
        if hasattr(m, "gate_w"):
            gw = gates.get((lid, "gate_w")) if gate else None
            if gate and gw is None and _STATE["installed"][0] != "stock":
                raise GateError(f"layer {lid}: gated arm asked but the checkpoint has no gate_w")
            m.gate_w.copy_(gw.to(dev) if gw is not None else torch.zeros_like(m.gate_w))
            if decay:
                ga = gates.get((lid, "wg_wa"))
                if ga is None:
                    raise GateError(f"layer {lid}: decay gate asked but the checkpoint has no wg_wa")
                m.wg_wa.copy_(ga.to(dev))
        elif gate and any(k[0] == lid for k in gates):
            raise GateError(f"layer {lid}: the checkpoint carries gates but this engine is stock")
    torch.cuda.synchronize()
    return {"lora_tensors_merged": written, "delta_kept": (num / den) if den else 1.0, "rounding": rounding,
            "gate_layers": sum(1 for k in gates if k[1] == "gate_w") if gate else 0,
            "decay_layers": sum(1 for k in gates if k[1] == "wg_wa") if decay else 0}


if __name__ == "__main__":
    print(json.dumps(static_check(), indent=1))
