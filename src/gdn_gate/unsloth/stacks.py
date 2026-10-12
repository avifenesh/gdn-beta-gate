"""Training stacks for the step 6 speed comparison (owner direction 2026-10-08: use the established fast stacks
where they beat our hand-written loop). Each stack builds the same trainable model: Qwen3.5-4B-Base, the arm's
wg gate (arms.py, which imports src/retrofit/patch.py unchanged), LoRA r16 alpha 32 on every GDN
layer's in_proj_qkv, in_proj_a, in_proj_b, out_proj, gradient checkpointing, bf16 weights, and packed rows whose
boundaries reset positions, attention, GDN state and conv (packing.py).

  ours     train_sft.py --pack: library model, our packing wrappers, our chunked CE (decoder + head).
  unsloth  Unsloth 2026.10.2 FastLanguageModel load and get_peft_model (its patches, compiled modules, fused loss),
           then our gate on every GDN layer and our packing wrappers; loss through the Unsloth model forward with
           labels (--loss model) or our chunked CE (--loss ours).
  trl      TRL 1.13 SFTTrainer, padding_free=True, use_liger_kernel=True (fused linear CE, RMSNorm, SwiGLU); the
           model is built as in 'ours' (+ Liger); TRL's padding-free collator gives position ids only, so a decoder
           pre-hook derives the example boundaries from them and adds our packing kwargs (without it GDN state and
           the conv cross examples: the 'trlstock' red arm in check_fast.py).

Unsloth must be imported before transformers: callers do `stacks.early(path)` before importing anything else.
The gate's forward is built from the pristine library source before Unsloth loads a model (prebuild_gate), and
gate_proof() shows that the forward running on every GDN layer is that one and that its _wg_beta is called.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
BROAD = HERE.parent / "broad"
for p in (str(HERE), str(BROAD)):
    if p not in sys.path:
        sys.path.insert(0, p)

PATHS = ("ours", "unsloth", "trl")
CPU = os.environ.get("WG_FAST_CPU") == "1"  # CPU plumbing test only (tiny model, torch GDN, no Liger, no Unsloth)
if CPU:
    os.environ["WG_NO_FLA"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
_EARLY = {"done": False}


def early(path: str) -> None:
    """Process start, before transformers is imported anywhere."""
    if _EARLY["done"]:
        return
    if os.environ.get("WG_ALLOW_CC1D") != "1":  # causal_conv1d held back lane-wide (2026-10-07), as in train_sft runs
        sys.modules["causal_conv1d"] = None
    assert "transformers" not in sys.modules, "stacks.early must run before transformers is imported"
    if path == "unsloth":
        os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")
        os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")  # offline unit, no download counter
        import unsloth  # noqa: F401
    _EARLY["done"] = True


def lib_gdn_source_sha() -> dict:
    """sha256 of the installed library's Qwen3_5GatedDeltaNet.forward text read from the file (not the live class,
    which a stack may have replaced) and of the live class forward's source."""
    import ast

    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    txt = Path(M.__file__).read_text()
    tree = ast.parse(txt)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5GatedDeltaNet":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "forward":
                    fn = ast.get_source_segment(txt, sub)
    live = inspect.getsource(M.Qwen3_5GatedDeltaNet.forward)
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]  # noqa: E731
    return {"file_forward": h(fn or ""), "live_forward": h(live),
            "live_is_file": " ".join((fn or "").split()) in " ".join(live.split()) if fn else False}


def prebuild_gate() -> dict:
    """Build patch.py's widened forward now (from the library source) and cache it where patch.install looks."""
    import patch

    rec = lib_gdn_source_sha()
    key = (True, False)
    if key not in patch._FORWARDS:
        patch._FORWARDS[key] = patch._build_forward(True, False)
    rec["gate_forward_code"] = patch._FORWARDS[key].__code__.co_filename
    return rec


def core(model):
    """(text decoder, lm_head) for a PEFT-wrapped CausalLM or ConditionalGeneration model."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    dec = m.model
    if hasattr(dec, "language_model"):
        dec = dec.language_model
    return dec, m.lm_head


def gdn_layers(model):
    import patch

    return patch.gdn_layers(model)


def build(path: str, model_path: str, arm: str, rank: int = 16, gc: str = "unsloth", max_seq: int = 32768,
          attn: str | None = None, dtype=None, seed: int = 0):
    """Returns (model, info). The model is in train mode with LoRA (+ gate W) trainable and packing installed."""
    import torch

    import arms
    import packing

    info = {"path": path, "arm": arm}
    dtype = dtype or torch.bfloat16
    if path == "unsloth":
        info["gate_prebuild"] = prebuild_gate()
        from unsloth import FastLanguageModel

        model, _tok = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=max_seq, dtype=dtype, load_in_4bit=False,
            load_in_8bit=False, full_finetuning=False, local_files_only=True,
            attn_implementation="sdpa")  # our packed attention is per-example SDPA (Unsloth may pick flex for head_dim 256)
        info["loaded_class"] = type(model).__name__
        info["gdn_after_load"] = lib_gdn_source_sha()
        new = arms.install(model, arm)  # uses the prebuilt forward from patch._FORWARDS
        model = FastLanguageModel.get_peft_model(
            model, r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=arms.LORA_TARGETS, bias="none",
            use_gradient_checkpointing=(gc if gc != "true" else True), random_state=seed, use_rslora=False)
        # Unsloth seeds its own LoRA init with random_state (same distribution as PEFT's: kaiming-uniform A, zero B),
        # so its init draw differs from train_sft.py's for the same seed; see init_from in bench_fast/train_sft_fast
        for n, p in model.named_parameters():
            if "gate_w" in n:
                p.requires_grad_(True)
        info["n_gate_params"] = len(new)
    else:
        model, new = arms.load_model(model_path, arm, device_map={"": "cpu"} if CPU else {"": 0},
                                     dtype=torch.float32 if CPU else dtype, attn=attn)
        if path == "trl" and not CPU:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

            apply_liger_kernel_to_qwen3_5(model=model, rms_norm=True, swiglu=True, fused_linear_cross_entropy=True)
            info["liger"] = True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = arms.add_lora(model, rank)
        info["n_gate_params"] = len(new)
    packing.install(model)
    packing.CONV_BACKEND = "torch"  # train_sft.py default (--conv_backend torch)
    model.train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    info["n_lora"] = sum(p.numel() for n, p in named if "gate_w" not in n)
    info["n_w"] = sum(p.numel() for n, p in named if "gate_w" in n)
    info["n_lora_tensors"] = sum(1 for n, _ in named if "lora_" in n)
    assert (info["n_w"] > 0) == (arm != "ctrl"), info
    return model, info


def canon(name: str) -> str:
    """Parameter name without the stack's wrapper prefix: 'layers.N.linear_attn....lora_A.default.weight'."""
    i = name.find("layers.")
    return name[i:] if i >= 0 else name


def load_trainable(model, state: dict) -> int:
    """Copy a train_sft checkpoint (ours' names) into any stack's model by canonical name. Returns tensors loaded;
    asserts every trainable tensor got a value."""
    import torch

    src = {canon(k): v for k, v in state.items()}
    n = 0
    with torch.no_grad():
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            c = canon(k)
            assert c in src, f"trainable tensor {k} absent from the state"
            p.copy_(src[c].to(device=p.device, dtype=p.dtype))
            n += 1
    return n


def trainable_named(model):
    return [(canon(n), p) for n, p in model.named_parameters() if p.requires_grad]


def gate_proof(model, run_once) -> dict:
    """Which forward runs on each GDN layer, and how often _wg_beta is called during run_once()."""
    calls = {"n": 0}
    layers = gdn_layers(model)
    def innermost(f):  # the exec'd source keeps the library's @force_accelerate_hooks (functools.wraps): unwrap it
        f = getattr(f, "__func__", f)
        chain = [f.__code__.co_filename]
        while hasattr(f, "__wrapped__"):
            f = f.__wrapped__
            chain.append(f.__code__.co_filename)
        return chain[-1], len(chain)

    fwd_files = sorted({innermost(l.forward)[0] for l in layers})
    instance_bound = all("forward" in l.__dict__ for l in layers)
    origs = []
    for l in layers:
        if hasattr(l, "_wg_beta"):
            f = l._wg_beta

            def wrapped(b, x, _f=f):
                calls["n"] += 1
                return _f(b, x)

            origs.append((l, f))
            l._wg_beta = wrapped
    try:
        run_once()
    finally:
        for l, f in origs:
            l._wg_beta = f
    beta_fns = sorted({getattr(f, "__func__", f).__name__ for _, f in origs})
    return {"gdn_layers": len(layers), "forward_code_files": fwd_files, "instance_bound": instance_bound,
            "beta_calls": calls["n"],
            "beta_fns": beta_fns, "arms": sorted({getattr(l, "_wg_arm", "none") for l in layers})}


class LazyMix:
    """bench_pack.LazyMix (same rows and order as sft_data.Mixture, messages read per row group on demand)."""

    def __init__(self, path):
        import numpy as np
        import pyarrow.parquet as pq

        import sft_data

        self.ids = pq.read_table(path, columns=["id"]).column("id").to_pylist()
        self.hold = np.array([sft_data.is_holdout(str(i)) for i in self.ids])
        self.train_rows, self.val_rows = np.flatnonzero(~self.hold), np.flatnonzero(self.hold)
        self.pf = pq.ParquetFile(path)
        self.rg = np.cumsum([0] + [self.pf.metadata.row_group(i).num_rows for i in range(self.pf.num_row_groups)])
        self.cache: dict = {}

    def msgs(self, row):
        import numpy as np

        g = int(np.searchsorted(self.rg, row, side="right") - 1)
        if g not in self.cache:
            if len(self.cache) > 8:
                self.cache.clear()
            self.cache[g] = self.pf.read_row_group(g, columns=["messages"]).column("messages")
        return json.loads(self.cache[g][row - self.rg[g]].as_py())


# ---------------------------------------------------------------- loss paths (one packed row of whole examples)

def packed_labels(exs):
    """Unshifted labels for a model forward over one packed row: each example's labels with its first token masked,
    so no position predicts across a boundary (the model shifts by one inside). Same (position, target) pairs as
    packing.batch's tgt."""
    out = []
    for e in exs:
        lab = list(e.labels)
        lab[0] = -100
        out += lab
    return out


def loss_model(model, exs, dev, n_lab):
    """Step-normalised loss of one packed row through the stack's own CausalLM forward (its fused CE). Returns
    (loss already divided by n_lab, label-token sum for logging)."""
    import torch

    import packing

    b = packing.batch(exs, dev, conv_k=packing._conv_k(model))
    lab = torch.tensor(packed_labels(exs), dtype=torch.long, device=dev)[None]
    out = model(input_ids=b["ids"], position_ids=b["pos"], labels=lab, num_items_in_batch=n_lab, use_cache=False,
                attention_mask={"full_attention": None, "linear_attention": None}, **b["kw"])
    return out.loss, int((lab[0, 1:] != -100).sum())


def loss_ours_row(model, exs, dev):
    """packing.loss_sum with stacks.core (works for ConditionalGeneration models too)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    import packing

    dec, head = core(model)
    b = packing.batch(exs, dev, conv_k=packing._conv_k(model))
    h = dec(input_ids=b["ids"], position_ids=b["pos"], attention_mask={"full_attention": None, "linear_attention": None},
            use_cache=False, **b["kw"]).last_hidden_state[0]
    sel = b["tgt"] != -100
    hs, y = h[sel], b["tgt"][sel]

    def head_ce(x, t):
        return F.cross_entropy(head(x).float(), t, reduction="sum")

    total = sum(checkpoint(head_ce, hs[s: s + 1024], y[s: s + 1024], use_reentrant=False)
                for s in range(0, hs.shape[0], 1024))
    return total, int(y.numel())


def boundary_hook(model):
    """Forward pre-hook on the text decoder for the TRL path: TRL's padding-free collator emits input_ids and
    position_ids (restarting at 0 per example) without boundary kwargs. Rebuild them from position_ids and add our
    packing kwargs, as packing.batch would for the same row. Returns the hook handle."""
    import torch

    import packing

    dec, _ = core(model)
    K = packing._conv_k(model)

    def pre(mod, args, kwargs):
        if kwargs.get("wg_bounds") is not None:
            return args, kwargs
        pos = kwargs.get("position_ids")
        ids = kwargs.get("input_ids", args[0] if args else None)
        if pos is None or ids is None or ids.shape[0] != 1:
            return args, kwargs
        p = pos[0] if pos.dim() == 2 else pos.reshape(-1, pos.shape[-1])[0]
        starts = torch.nonzero(p == 0).flatten().tolist()
        T = int(p.shape[0])
        assert starts and starts[0] == 0, "packed row must start an example at position 0"
        lens = [b - a for a, b in zip(starts, starts[1:] + [T])]

        class _E:
            def __init__(self, n):
                self.ids = [0] * n
                self.labels = [-100] * n

        b = packing.batch([_E(n) for n in lens], ids.device, conv_k=K)
        kwargs.update(b["kw"])
        kwargs["attention_mask"] = {"full_attention": None, "linear_attention": None}
        return args, kwargs

    return dec.register_forward_pre_hook(pre, with_kwargs=True)
