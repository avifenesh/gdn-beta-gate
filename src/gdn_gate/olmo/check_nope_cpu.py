#!/usr/bin/env python3
"""CPU proof of patch_olmo.fix_position_encoding (2026-10-08 vLLM-vs-HF Olmo diff).

transformers 5.17 loads `rope_parameters: null` (Olmo-Hybrid-Think-SFT-7B's config.json) as {"rope_theta": 1e4},
so the full-attention layers run RoPE; `{"rope_theta": null}` (Base, Instruct-SFT) loads as NoPE. vLLM 0.31.0 runs
NoPE for both. A random tiny Olmo-Hybrid saved three ways, same weights:
  A  rope_parameters {"rope_theta": null}   (Base style, NoPE in the library)
  B  rope_parameters null                   (Think-SFT style)
  C  rope_parameters {"rope_theta": 10000}  (a real RoPE config)
Checks:
  red arm   B loaded by the library alone has a rotary embedding and its logits differ from A (the bug)
  fixed     after patch_olmo.install, B has no rotary embedding and its logits equal A bit for bit
  no-op     install leaves A NoPE and C RoPE (C still differs from A): only declared-NoPE checkpoints change
  real      checkpoint_is_nope on the local Olmo-Hybrid Think-SFT, Base, Instruct-SFT snapshots (all True)
usage: python check_nope_cpu.py [--out JSON]
"""
import argparse
import os
import glob
import inspect
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_olmo as P  # noqa: E402
from patch_olmo import M  # noqa: E402

# CPU: run the library's torch reference kernels (fla's are triton/CUDA)
M.torch_chunk_gated_delta_rule = inspect.unwrap(M.torch_chunk_gated_delta_rule)
M.causal_conv1d_fn = inspect.unwrap(M.causal_conv1d_fn)
HUB = os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
rec, ok = {"checks": []}, True


def check(name, cond, **kw):
    global ok
    ok &= bool(cond)
    rec["checks"].append({"check": name, "pass": bool(cond), **kw})
    print(("PASS " if cond else "FAIL ") + name, json.dumps(kw) if kw else "", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    cfg = M.OlmoHybridConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=4, layer_types=["linear_attention"] * 3 + ["full_attention"], linear_num_key_heads=4,
        linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=True, pad_token_id=0, max_position_embeddings=512, rope_parameters={"rope_theta": None})
    base = M.OlmoHybridForCausalLM(cfg).float().eval()
    x = torch.randint(1, 256, (1, 48))
    tmp = Path(tempfile.mkdtemp(prefix="nope_cpu_", dir=str(Path.home() / ".cache")))
    dirs = {}
    for tag, rp in (("A", {"rope_theta": None}), ("B", None), ("C", {"rope_theta": 10000.0})):
        d = tmp / tag
        base.save_pretrained(d)
        raw = json.load(open(d / "config.json"))
        raw.pop("rope_theta", None)
        raw["rope_parameters"] = rp
        json.dump(raw, open(d / "config.json", "w"))
        dirs[tag] = str(d)

    def load(tag):
        return AutoModelForCausalLM.from_pretrained(dirs[tag], dtype=torch.float32).eval()

    def logits(m):
        with torch.no_grad():
            return m(input_ids=x).logits

    mA, mB, mC = load("A"), load("B"), load("C")
    rec["library_rope_parameters"] = {t: str(m.config.rope_parameters) for t, m in (("A", mA), ("B", mB), ("C", mC))}
    lA_raw, lB_raw = logits(mA), logits(mB)
    dB_raw = float((lB_raw - lA_raw).abs().max())
    check("red arm: library alone runs RoPE on rope_parameters null (B has rotary_emb, logits differ from A)",
          mB.model.rotary_emb is not None and dB_raw > 1e-3, max_abs_logit_diff=dB_raw)
    for m in (mA, mB, mC):
        P.install(m)
    lA, lB, lC = logits(mA), logits(mB), logits(mC)
    check("fixed: install drops RoPE on B, logits equal A bit for bit",
          mB.model.rotary_emb is None and mB._ng_position == "nope" and torch.equal(lB, lA),
          position=mB._ng_position, max_abs_logit_diff=float((lB - lA).abs().max()))
    check("no-op on A: stays NoPE, logits unchanged", mA._ng_position == "nope" and torch.equal(lA, lA_raw))
    dC = float((lC - lA).abs().max())
    check("no-op on C: a real RoPE config keeps RoPE", mC._ng_position == "rope" and mC.model.rotary_emb is not None
          and dC > 1e-3, max_abs_logit_diff_vs_A=dC)
    for name, want in (("Olmo-Hybrid-Think-SFT-7B", True), ("Olmo-Hybrid-7B", True), ("Olmo-Hybrid-Instruct-SFT-7B", True)):
        snaps = sorted(glob.glob(f"{HUB}/models--allenai--{name}/snapshots/*/config.json"))
        if not snaps:
            check(f"real: {name} snapshot present", False)
            continue
        got = P.checkpoint_is_nope(str(Path(snaps[-1]).parent))
        check(f"real: checkpoint_is_nope({name}) == {want}", got == want, got=got)
    rec["pass"] = ok
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    import shutil

    shutil.rmtree(tmp)
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
