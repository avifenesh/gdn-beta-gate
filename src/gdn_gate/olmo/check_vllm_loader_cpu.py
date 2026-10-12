#!/usr/bin/env python3
"""CPU proof of the vLLM Olmo-Hybrid conv1d loader fix (vllm_arm.patch_conv1d_loader).

Runs vLLM's own AutoWeightsLoader with OlmoHybridModel.hf_to_vllm_mapper over the real checkpoint's
{q,k,v}_conv1d tensors (every GDN layer) into a module tree with vLLM's parameter names, each fused conv1d
parameter carrying the loader made by olmo_gdn_linear_attn._make_fused_conv1d_weight_loader.
  red arm   unpatched factory: the load must raise the TypeError seen on the Spark (dims[None])
  fixed     patched factory: every fused parameter must equal cat(q, k, v) bit for bit, 3 loader calls per layer
  context   which stacked targets the mapper produces, and which of their vLLM classes define load_weights (those
            read the shard id from the tensor attribute; conv1d is a ColumnParallelLinear, which does not)
Needs a vLLM venv whose utils.py, linear.py, olmo_hybrid.py and olmo_gdn_linear_attn.py match the Spark's 0.31.0
(md5 recorded in the receipt). Usage: python check_vllm_loader_cpu.py --model DIR [--model DIR2] --out JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def run_model(model_dir, G, O, AutoWeightsLoader, VA):
    import torch
    from safetensors import safe_open
    from torch import nn

    cfg = json.load(open(Path(model_dir) / "config.json"))
    kd = cfg["linear_key_head_dim"] * cfg["linear_num_key_heads"]
    vd = cfg["linear_value_head_dim"] * cfg["linear_num_value_heads"]
    K = cfg["linear_conv_kernel_dim"]
    st = Path(model_dir) / "model.safetensors"
    with safe_open(str(st), framework="pt") as f:
        names = list(f.keys())
        conv_names = sorted(n for n in names if "_conv1d." in n)
        ref = {n: f.get_tensor(n) for n in conv_names}
    layers = sorted({int(n.split(".")[2]) for n in conv_names})
    nl = max(int(n.split(".")[2]) for n in names if n.startswith("model.layers.")) + 1

    def tree():
        root = nn.Module()
        mods = []
        for i in range(nl):
            m = nn.Module()
            if i in layers:
                la = nn.Module()
                conv = nn.Module()
                conv.weight = nn.Parameter(torch.full((2 * kd + vd, 1, K), float("nan"), dtype=torch.bfloat16),
                                           requires_grad=False)
                conv.weight.weight_loader = G._make_fused_conv1d_weight_loader([kd, kd, vd], 1, 0)
                la.conv1d = conv
                m.linear_attn = la
            mods.append(m)
        root.layers = nn.ModuleList(mods)
        return root

    def weights():  # names relative to OlmoHybridModel, as AutoWeightsLoader passes them to its load_weights
        for n in conv_names:
            yield n[len("model."):], ref[n].clone()

    mapper = O.OlmoHybridModel.hf_to_vllm_mapper
    out = {"model": str(model_dir), "n_tensors": len(names), "gdn_layers": len(layers), "conv_tensors": len(conv_names),
           "dims": {"key_dim": kd, "value_dim": vd, "kernel": K}}

    # stacked targets over the whole checkpoint
    tgt = {}
    for n in names:
        r = mapper._map_name_with_shard(n)
        if r is not None and r[1] is not None:
            tgt.setdefault(r[0].rsplit(".", 2)[-2], set()).add(str(r[1]))
    out["stacked_targets"] = {k: sorted(v) for k, v in sorted(tgt.items())}

    # red arm: unpatched factory (restored temporarily)
    patched = G._make_fused_conv1d_weight_loader
    G._make_fused_conv1d_weight_loader = VA._STATE["conv_factory"]
    try:
        AutoWeightsLoader(tree()).load_weights(weights(), mapper=mapper)
        out["red_unpatched"] = "loaded (no error)"
        red_ok = False
    except TypeError as e:
        out["red_unpatched"] = f"TypeError: {e}"
        red_ok = "NoneType" in str(e)
    finally:
        G._make_fused_conv1d_weight_loader = patched

    # fixed
    VA.LOADER["n"] = 0
    VA.LOADER["shards"] = {}
    root = tree()
    loaded = AutoWeightsLoader(root).load_weights(weights(), mapper=mapper)
    bad = []
    for i in layers:
        p = root.layers[i].linear_attn.conv1d.weight.data
        exp = torch.cat([ref[f"model.layers.{i}.linear_attn.{x}_conv1d.weight"] for x in "qkv"], 0)
        if p.shape != exp.shape or not torch.equal(p, exp):
            bad.append(i)
    # a loader that ignored the shard id (offset 0 every time) or swapped q and k would fail the equality above;
    # make sure q and k actually differ so a q/k swap is detectable
    i0 = layers[0]
    qk_differ = not torch.equal(ref[f"model.layers.{i0}.linear_attn.q_conv1d.weight"],
                                ref[f"model.layers.{i0}.linear_attn.k_conv1d.weight"])
    out.update({"loaded_names": len(loaded), "loader_calls": VA.LOADER["n"],
                "loader_shards": {str(k): v for k, v in sorted(VA.LOADER["shards"].items())},
                "layers_mismatch": bad, "q_k_differ": qk_differ})
    checks = {
        "red arm: unpatched loader raises the Spark TypeError": red_ok,
        "fixed: every fused conv1d equals cat(q,k,v) bit for bit": not bad,
        "fixed: 3 loader calls per GDN layer": VA.LOADER["n"] == 3 * len(layers),
        "fixed: shard ids 0,1,2 each once per layer": VA.LOADER["shards"] == {0: len(layers), 1: len(layers), 2: len(layers)},
        "q and k conv differ (swap detectable)": qk_differ,
        "only conv1d among stacked targets lacks a module load_weights": True,  # filled by caller
    }
    out["checks"] = checks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(HERE))
    import vllm
    import vllm_arm as VA
    from vllm.model_executor.layers import linear as L
    from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as G
    from vllm.model_executor.models import olmo_hybrid as O
    from vllm.model_executor.models import utils as U

    VA.patch_conv1d_loader(G)
    # which vLLM classes back each stacked target, and whether they read the shard id themselves
    cls = {"qkv_proj": L.QKVParallelLinear, "in_proj_qkvg": L.MergedColumnParallelLinear,
           "gate_up_proj": L.MergedColumnParallelLinear, "conv1d": L.ColumnParallelLinear}
    has_lw = {k: callable(getattr(c, "load_weights", None)) for k, c in cls.items()}
    files = {m.__name__: md5(m.__file__) for m in (U, L, O, G)}
    rec = {"vllm": vllm.__version__, "files_md5": files, "target_class_has_load_weights": has_lw, "models": []}
    ok = True
    for m in a.model:
        r = run_model(m, G, O, U.AutoWeightsLoader, VA)
        lacking = sorted(k for k in r["stacked_targets"] if not has_lw.get(k, False))
        r["stacked_targets_without_load_weights"] = lacking
        r["checks"]["only conv1d among stacked targets lacks a module load_weights"] = lacking == ["conv1d"]
        for k, v in r["checks"].items():
            print(("PASS " if v else "FAIL ") + f"{Path(m).parent.parent.name}: {k}", flush=True)
            ok &= bool(v)
        rec["models"].append(r)
    rec["pass"] = ok
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["cmd"] = " ".join(sys.argv)
    rec["host"] = os.uname().nodename
    json.dump(rec, open(a.out, "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
