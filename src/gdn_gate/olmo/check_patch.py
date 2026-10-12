#!/usr/bin/env python3
"""Proof that each patch_olmo arm changes exactly what it should, and nothing when it should not.

Beta is captured at the kernel boundary (a recorder wrapped around the module's chunk/recurrent delta-rule entry
points) and b = b_proj(h) is captured by a forward hook on every GDN layer, so the expected beta of each arm is
recomputed here from b, independently of patch_olmo's code. Checks:
  library    patched forward in stock mode: logits and kernel betas bit-identical to the unpatched library
  per arm    kernel betas equal this file's expectation exactly; red arm: the same betas compared against the
             stock expectation (stock: against the erase one) must differ, and the arm must touch at least one
             position where 2 s > 1 (non-vacuous); logits must differ from stock for every lesion arm
  gate       R = 2 at W = 0 is bit-identical to stock (logits too), R = 1 at W = 0 equals erase; with a large
             random W, R = 1 never exceeds 1 while R = 2 with the same W does (red arm for the bound); W gets a
             nonzero gradient at W = 0
--tiny runs a random 4-layer Olmo-Hybrid on CPU through the library's torch reference kernels; --model runs the
real checkpoint on GPU through fla. Writes a JSON receipt and exits nonzero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_olmo as P  # noqa: E402
from patch_olmo import M  # noqa: E402


class Recorder:
    def __init__(self, model, cpu_ref: bool):
        self.events = []
        self.orig = {n: getattr(M, n) for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")}
        for n, f in self.orig.items():
            impl = f.__wrapped__ if cpu_ref else f

            def rec(*a, _impl=impl, **kw):
                beta = kw["beta"] if "beta" in kw else a[4]
                self.events.append(("beta", beta.detach().clone()))
                return _impl(*a, **kw)

            setattr(M, n, rec)
        self.hooks = [L.b_proj.register_forward_hook(self._hook(i)) for i, L in enumerate(P.gdn_layers(model))]

    def _hook(self, i):
        def h(mod, inp, out):
            self.events.append(("b", i, out.detach().clone()))
        return h

    def pairs(self):
        """[(gdn index, b, beta)] in call order."""
        out, last = [], None
        for e in self.events:
            if e[0] == "b":
                last = e
            else:
                assert last is not None
                out.append((last[1], last[2], e[1]))
                last = None
        self.events = []
        return out

    def close(self):
        for n, f in self.orig.items():
            setattr(M, n, f)
        for h in self.hooks:
            h.remove()


def expect(arm, i, b, layer=None, heads=None):
    s = b.sigmoid()
    stock = s * 2.0
    erase = stock.clamp(max=1.0)
    if arm == "stock":
        return stock
    if arm == "half":
        return s
    if arm == "erase":
        return erase
    if arm == "layer":
        return erase if i == layer else stock
    if arm == "heads":
        hs = (heads or {}).get(i, [])
        if not hs:
            return stock
        m = torch.zeros(b.shape[-1], dtype=torch.bool, device=b.device)
        m[torch.tensor(hs)] = True
        return torch.where(m, erase, stock)
    raise ValueError(arm)


def run(model, ids, rec):
    with torch.no_grad():
        logits = model(input_ids=ids).logits.float()
    return logits, rec.pairs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--text_ids", default="", help="optional .pt of [B, T] token ids for the real-model check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    if a.tiny:
        cfg = M.OlmoHybridConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=32,
            linear_conv_kernel_dim=4, linear_allow_neg_eigval=True, pad_token_id=0, max_position_embeddings=512)
        model = M.OlmoHybridForCausalLM(cfg).float().eval()
        for L in P.gdn_layers(model):  # spread b so both sides of 2 s = 1 occur
            torch.nn.init.normal_(L.b_proj.weight, std=0.5)
        ids = torch.randint(1, 256, (2, 96))
        dev = "cpu"
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
        ids = torch.load(a.text_ids)[:2, :1024].long().cuda() if a.text_ids else torch.randint(1000, 50000, (2, 1024)).cuda()
        dev = "cuda"
    # only the gate's W may need gradients: base weights frozen (a 7B bf16 grad set would not fit the workstation)
    model.requires_grad_(False)
    checks, ok = [], True

    def check(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        checks.append({"check": name, "pass": bool(cond), **info})
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info) if info else "", flush=True)

    rec = Recorder(model, cpu_ref=a.tiny)
    lib_logits, lib_pairs = run(model, ids, rec)
    layers = P.install(model)
    stock_logits, stock_pairs = run(model, ids, rec)
    check("library: stock logits bit-identical", torch.equal(lib_logits, stock_logits))
    check("library: stock kernel betas bit-identical",
          len(lib_pairs) == len(stock_pairs) == len(layers)
          and all(torch.equal(x[2], y[2]) for x, y in zip(lib_pairs, stock_pairs)), n_layers=len(layers))
    n_gt1 = sum(int((p[1] > 0).sum()) for p in stock_pairs)
    n_all = sum(p[1].numel() for p in stock_pairs)
    check("input has positions with 2s > 1", n_gt1 > 0, frac=n_gt1 / max(1, n_all))

    # head ranking from the stock run (stats) for the heads arm
    P.enable_stats(model, True)
    run(model, ids, rec)
    stats = P.collect_stats(model)
    P.enable_stats(model, False)
    nh = layers[0].num_v_heads
    k = max(1, len(layers) * nh // 10)
    top = P.rank_heads(stats, k)
    mid = len(layers) // 2
    arms = [("stock", {}), ("half", {}), ("erase", {}), ("layer", {"layer": 0}), ("layer", {"layer": mid}),
            ("heads", {"heads": top})]
    for arm, kw in arms:
        desc = P.set_arm(model, arm, **kw)
        logits, pairs = run(model, ids, rec)
        tag = json.dumps(desc)[:120]
        exact = all(torch.equal(beta, expect(arm, i, b, **kw)) for i, b, beta in pairs) and len(pairs) == len(layers)
        check(f"{arm} {kw.get('layer', '')}: betas equal expectation", exact, desc=tag)
        if arm == "stock":
            red = any(not torch.equal(beta, expect("erase", i, b)) for i, b, beta in pairs)
            check("stock red arm: differs from erase expectation", red)
            check("stock logits equal library", torch.equal(logits, lib_logits))
            continue
        red = any(not torch.equal(beta, expect("stock", i, b)) for i, b, beta in pairs)
        check(f"{arm} {kw.get('layer', '')} red arm: differs from stock expectation", red)
        check(f"{arm} {kw.get('layer', '')}: logits differ from stock", not torch.equal(logits, stock_logits),
              max_abs=float((logits - stock_logits).abs().max()))
        if arm in ("layer", "heads"):
            untouched = []
            for i, b, beta in pairs:
                st = expect("stock", i, b)
                if arm == "layer" and i != kw["layer"]:
                    untouched.append(torch.equal(beta, st))
                if arm == "heads":
                    hs = kw["heads"].get(i, [])
                    keep = [h for h in range(nh) if h not in hs]
                    untouched.append(torch.equal(beta[..., keep], st[..., keep]))
            check(f"{arm} {kw.get('layer', '')}: untouched layers/heads exactly stock", all(untouched))
            touched = sum(int((b > 0).sum()) for i, b, _ in pairs if (arm == "layer" and i == kw["layer"]))
            if arm == "heads":
                touched = sum(int((b[..., kw["heads"].get(i, [])] > 0).sum()) for i, b, _ in pairs)
            check(f"{arm} {kw.get('layer', '')}: clamps at least one 2s > 1 position", touched > 0, n=touched)
    P.set_arm(model, "stock")

    # training gate
    for R, base_arm in ((2.0, "stock"), (1.0, "erase")):
        for L in layers:
            if hasattr(L, "gate_w"):
                del L.gate_w
        params = P.install_gate(model, R)
        logits, pairs = run(model, ids, rec)
        check(f"gate R={R} W=0: betas equal {base_arm}",
              all(torch.equal(beta, expect(base_arm, i, b)) for i, b, beta in pairs))
        if R == 2.0:
            check("gate R=2 W=0: logits bit-identical to library", torch.equal(logits, lib_logits))
        # gradient reaches W at W = 0
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(input_ids=ids[:1, :256]).logits.float()  # short slice: backward activations fit the workstation
            out.logsumexp(-1).mean().backward()
        gn = sum(float(p.grad.norm()) for p in params if p.grad is not None)
        check(f"gate R={R}: W gets a gradient at W=0", gn > 0, grad_norm=gn)
        model.zero_grad(set_to_none=True)
        rec.pairs()
        g = torch.Generator(device="cpu").manual_seed(1)
        with torch.no_grad():
            for p in params:
                p.copy_(torch.randn(p.shape, generator=g).to(p.device) * 0.5)
        _, pairs = run(model, ids, rec)
        mx = max(float(beta.float().max()) for _, _, beta in pairs)
        mn = min(float(beta.float().min()) for _, _, beta in pairs)
        if R == 1.0:
            check("gate R=1 random W: beta never exceeds 1", mx <= 1.0 and mn >= 0.0, max=mx, min=mn)
        else:
            check("gate R=2 random W (red arm for the bound): beta exceeds 1", mx > 1.0, max=mx, min=mn)
    rec.close()
    receipt = {"pass": ok, "mode": "tiny-cpu" if a.tiny else "model-gpu", "model": a.model, "device": dev,
               "ids_shape": list(ids.shape), "checks": checks, "torch": torch.__version__,
               "transformers": P.transformers.__version__, "wall_s": round(time.time() - t0, 1),
               "top_heads_k": k, "top_heads": {str(k_): v for k_, v in top.items()}}
    if dev == "cuda":
        receipt["gpu"] = torch.cuda.get_device_name(0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(receipt, open(a.out, "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
