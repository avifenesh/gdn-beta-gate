#!/usr/bin/env python3
"""Correctness of a fast stack against our verified packed trainer, on fixed examples and fixed trained adapters.

Run once with --stack ours (writes the reference to --ref), then once per stack (unsloth, trl), each in its own
process. Every process loads the same base weights, the same trained LoRA + gate tensors (the step 6 pilot's
ckpt_step50.pt, wide arm), and scores the same N examples (stream seed 0, the first N).

Reference process (ours, library model + packing.py), as check_pack.py:
  alone    each example in its own forward (no padding, no packing): the reference values
  padded   length-sorted right-padded micro-batches (the trainer before --pack)
  packed   all examples in one packed row (train_sft.py --pack)
  per example: 'full' = label CE sum, 'head64' / 'head8' = mean next-token CE over the first 64 / 8 positions;
  gradients of the label CE sum over all examples, split into the LoRA tensors and the gate W tensors.
Stack process (all examples in one packed row through the stack's own path):
  metrics  the stack's decoder (its patches) on the packed row; unsloth: packing.batch kwargs; trl: input_ids and
           position_ids only, boundaries rebuilt by stacks.boundary_hook (what TRL's padding-free collator gives)
  total    the label CE sum through the stack's training loss (unsloth: its CausalLM forward with labels and
           num_items_in_batch=1, its fused CE; trl: the Liger fused-linear-CE forward), compared with sum(alone)
  grads    from that training loss
Bounds, calibrated in the reference run (no absolute tolerance):
  metric m:  bound(m) = 2 x max(dev(padded, m), dev(packed, m)) + 1e-6,  dev = max_example |x - alone| / |alone|
  total:     bound = 2 x max(|sum padded - sum alone|, |sum packed - sum alone|) / |sum alone| + 1e-6
  grads g in {lora, gate}: bound(g) = 2 x max(rel(padded), rel(packed)) + 1e-5, rel = |g - g_alone| / |g_alone|
  green: every metric, the total and both gradient groups within bound
Red arms (in the stack process; each must exceed its bound by >= 10x somewhere among the metrics/total/grads):
  nogate   the gate patch not applied (library forward on every GDN layer, beta = sigmoid(b)); W gets no gradient
  clamp1   wide clamped at 1 (the wide1 beta function swapped in on the same W)
  trlstock (trl only) TRL's stock padding-free input with no boundary kwargs (GDN state and conv cross examples)
Plus gate_proof: the forward running on each GDN layer is patch.py's rewrite and _wg_beta is called.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--stack", required=True, choices=["ours", "unsloth", "trl"])
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--adapters", required=True)
ap.add_argument("--ref", required=True, help="reference .pt (written by --stack ours, read by the others)")
ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
ap.add_argument("--n_ex", type=int, default=24)
ap.add_argument("--wscale", type=float, default=1.0, help="multiply the loaded gate W (both reference and stack runs): "
                "a stronger gate makes the clamp1 red arm visible above bf16 noise")
ap.add_argument("--out", required=True)
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stacks  # noqa: E402

stacks.early(args.stack)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import packing  # noqa: E402
import sft_data  # noqa: E402
import train_sft  # noqa: E402

METRICS = ("full", "head64", "head8")
CPU = stacks.CPU
dev = torch.device("cpu" if CPU else "cuda:0")
DT = getattr(torch, args.dtype)


def metrics(head, h, e):
    tgt = torch.tensor(e.labels[1:] + [-100], device=dev)
    nxt = torch.tensor(e.ids[1:] + [-100], device=dev)
    sel = tgt != -100
    out = {"full": float(F.cross_entropy(head(h[sel]).float(), tgt[sel], reduction="sum")) if bool(sel.any()) else 0.0}
    for k, n in (("head64", 64), ("head8", 8)):
        m = min(n, len(e.ids) - 1)
        out[k] = float(F.cross_entropy(head(h[:m]).float(), nxt[:m], reduction="mean")) if m > 0 else 0.0
    return out


def dev_of(a, ref):
    return {m: max(abs(x[m] - y[m]) / max(abs(y[m]), 1e-6) for x, y in zip(a, ref)) for m in METRICS}


def grads_of(model, fn):
    model.zero_grad(set_to_none=True)
    fn()
    out = {}
    for name, p in stacks.trainable_named(model):
        grp = "gate" if "gate_w" in name else "lora"
        g = p.grad.float().flatten().cpu() if p.grad is not None else torch.zeros(p.numel())
        out.setdefault(grp, []).append((name, g))
    model.zero_grad(set_to_none=True)
    return {k: torch.cat([g for _, g in sorted(v)]) for k, v in out.items()}


def rel(g, ref):
    return {k: float((g[k] - ref[k]).norm() / ref[k].norm()) if k in g else float("inf") for k in ref}


def build_ours_ref(model, exs, tok):
    dec, head = stacks.core(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16 and not CPU)
    with torch.no_grad(), ac:
        alone = [metrics(head, dec(input_ids=torch.tensor(e.ids, device=dev)[None]).last_hidden_state[0], e) for e in exs]
        padded = {}
        for grp in sft_data.micro_batches(exs, 4096):
            ids, _, att = train_sft.pad(grp, pad_id)
            h = dec(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
            for i, e in enumerate(grp):
                padded[e.row] = metrics(head, h[i, : len(e.ids)], e)
        padded = [padded[e.row] for e in exs]
        packed = packed_metrics(model, exs, hook=False)

    def alone_bwd():
        for e in exs:
            with ac:
                s, c = train_sft.sft_loss_sum(model, [e], pad_id, dev)
            if c:
                s.backward()

    def padded_bwd():
        for grp in sft_data.micro_batches(exs, 4096):
            with ac:
                s, c = train_sft.sft_loss_sum(model, grp, pad_id, dev)
            if c:
                s.backward()

    def packed_bwd():
        with ac:
            s, _ = stacks.loss_ours_row(model, exs, dev)
        s.backward()

    ga, gpad, gpk = grads_of(model, alone_bwd), grads_of(model, padded_bwd), grads_of(model, packed_bwd)
    with torch.no_grad(), ac:
        tot_pk = float(stacks.loss_ours_row(model, exs, dev)[0])
    return {"alone": alone, "padded": padded, "packed": packed, "g_alone": ga, "g_padded": gpad, "g_packed": gpk,
            "total_packed": tot_pk}


def packed_metrics(model, exs, hook: bool):
    """One packed row through the stack's decoder. hook=True: input_ids + position_ids only (TRL collator output),
    boundary kwargs from stacks.boundary_hook if installed."""
    dec, head = stacks.core(model)
    b = packing.batch(exs, dev, conv_k=packing._conv_k(model))
    if hook:
        h = dec(input_ids=b["ids"], position_ids=b["pos"], use_cache=False).last_hidden_state[0]
    else:
        h = dec(input_ids=b["ids"], position_ids=b["pos"], attention_mask={"full_attention": None, "linear_attention": None},
                use_cache=False, **b["kw"]).last_hidden_state[0]
    return [metrics(head, h[s:e], ex) for (s, e), ex in zip(b["kw"]["wg_bounds"], exs)]


def stack_loss(model, exs, hook: bool):
    """Label CE sum of all examples through the stack's own training loss on one packed row."""
    import torch as T

    b = packing.batch(exs, dev, conv_k=packing._conv_k(model))
    lab = T.tensor(stacks.packed_labels(exs), dtype=T.long, device=dev)[None]
    if hook:  # TRL padding-free: input_ids, position_ids, labels
        out = model(input_ids=b["ids"], position_ids=b["pos"], labels=lab, num_items_in_batch=1, use_cache=False)
    else:
        out = model(input_ids=b["ids"], position_ids=b["pos"], labels=lab, num_items_in_batch=1, use_cache=False,
                    attention_mask={"full_attention": None, "linear_attention": None}, **b["kw"])
    return out.loss


def run_stack(model, exs, ref, hook: bool):
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16 and not CPU)
    packing.COUNTS.update(attn=0, gdn=0, conv=0)
    with torch.no_grad(), ac:
        pm = packed_metrics(model, exs, hook)
        total = float(stack_loss(model, exs, hook))
    counts = dict(packing.COUNTS)

    def bwd():
        with ac:
            l = stack_loss(model, exs, hook)
        l.backward()

    g = grads_of(model, bwd)
    s_alone = sum(m["full"] for m in ref["alone"])
    return {"dev": dev_of(pm, ref["alone"]), "total_rel": abs(total - s_alone) / abs(s_alone),
            "grad_rel": rel(g, ref["g_alone"]), "engaged": counts, "total": total}


def main():
    t0 = time.time()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    s = sft_data.Stream(stacks.LazyMix(args.data), sft_data.Renderer(tok), 0, 4096)
    exs = [s.next() for _ in range(args.n_ex)]
    if args.stack == "unsloth":
        import arms

        model, info = stacks.build("unsloth", args.model, "wide", gc="true", dtype=DT)
    else:
        import arms

        model, new = arms.load_model(args.model, "wide", device_map={"": "cpu"} if CPU else {"": 0}, dtype=DT)
        info = {"path": args.stack}
        if args.stack == "trl" and not CPU:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

            apply_liger_kernel_to_qwen3_5(model=model, rms_norm=True, swiglu=True, fused_linear_cross_entropy=True)
            info["liger"] = True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = arms.add_lora(model, 16)
        packing.install(model)
        packing.CONV_BACKEND = "torch"
    model.train()
    state = torch.load(args.adapters, map_location="cpu")
    state = {k: (v * args.wscale if "gate_w" in k else v) for k, v in state.items()}
    n_loaded = stacks.load_trainable(model, state)
    res = {"args": vars(args), "info": info, "adapters_loaded": n_loaded, "n_ex": len(exs),
           "lens": [len(e.ids) for e in exs], "torch": torch.__version__, "gpu": None if CPU else torch.cuda.get_device_name(0),
           "param_dtypes": sorted({str(p.dtype) for p in model.parameters()})}
    if args.stack == "ours":
        ref = build_ours_ref(model, exs, tok)
        ref["exs_hash"] = hash(tuple(tuple(e.ids) for e in exs))
        torch.save(ref, args.ref)
        d_pad, d_pk = dev_of(ref["padded"], ref["alone"]), dev_of(ref["packed"], ref["alone"])
        s_al = sum(m["full"] for m in ref["alone"])
        s_pad = sum(m["full"] for m in ref["padded"])
        res["ref"] = {"dev_padded": d_pad, "dev_packed": d_pk, "grad_rel_padded": rel(ref["g_padded"], ref["g_alone"]),
                      "grad_rel_packed": rel(ref["g_packed"], ref["g_alone"]),
                      "total_rel_padded": abs(s_pad - s_al) / s_al, "total_rel_packed": abs(ref["total_packed"] - s_al) / s_al,
                      "grad_norm_alone": {k: float(v.norm()) for k, v in ref["g_alone"].items()}}
        res["verdict"] = "REF"
    else:
        ref = torch.load(args.ref)
        d_pad, d_pk = dev_of(ref["padded"], ref["alone"]), dev_of(ref["packed"], ref["alone"])
        bound = {m: 2 * max(d_pad[m], d_pk[m]) + 1e-6 for m in METRICS}
        s_al = sum(m["full"] for m in ref["alone"])
        tb = 2 * max(abs(sum(m["full"] for m in ref["padded"]) - s_al), abs(ref["total_packed"] - s_al)) / s_al + 1e-6
        rp, rk = rel(ref["g_padded"], ref["g_alone"]), rel(ref["g_packed"], ref["g_alone"])
        gb = {k: 2 * max(rp[k], rk[k]) + 1e-5 for k in ref["g_alone"]}
        res["bounds"] = {"metric": bound, "total": tb, "grad": gb}
        hook = args.stack == "trl"
        hh = stacks.boundary_hook(model) if hook else None
        dec, _ = stacks.core(model)
        b = packing.batch(exs[:4], dev, conv_k=packing._conv_k(model))

        def once():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16 and not CPU):
                if hook:
                    dec(input_ids=b["ids"], position_ids=b["pos"], use_cache=False)
                else:
                    dec(input_ids=b["ids"], position_ids=b["pos"], use_cache=False,
                        attention_mask={"full_attention": None, "linear_attention": None}, **b["kw"])

        res["gate_proof"] = stacks.gate_proof(model, once)
        green = run_stack(model, exs, ref, hook)

        def ratios(r):
            out = {f"metric_{m}": r["dev"][m] / bound[m] for m in METRICS}
            out["total"] = r["total_rel"] / tb
            out.update({f"grad_{k}": r["grad_rel"][k] / gb[k] for k in gb})
            return out

        green["ratio_to_bound"] = ratios(green)
        green["ok"] = all(v <= 1.0 for v in green["ratio_to_bound"].values())
        res["green"] = green
        red = {}
        # nogate: library forward on every GDN layer (instance attribute removed: the class forward runs, beta = s)
        import arms
        import patch

        layers = patch.gdn_layers(model)
        saved = [(l, l.__dict__.get("forward")) for l in layers]
        for l in layers:
            l.__dict__.pop("forward", None)
        r = run_stack(model, exs, ref, hook)
        for l, f in saved:
            if f is not None:
                l.forward = f
        r["ratio_to_bound"] = ratios(r)
        red["nogate"] = r
        # clamp1: wide1's beta function on the same W
        saved = [(l, l._wg_beta) for l in layers]
        import types as _t

        for l in layers:
            l._wg_beta = _t.MethodType(arms._wg_beta1, l)
        r = run_stack(model, exs, ref, hook)
        for l, f in saved:
            l._wg_beta = f
        r["ratio_to_bound"] = ratios(r)
        red["clamp1"] = r
        if hook:
            hh.remove()
            r = run_stack(model, exs, ref, hook)
            r["ratio_to_bound"] = ratios(r)
            red["trlstock"] = r
            hh = stacks.boundary_hook(model)
        for k, r in red.items():
            r["max_ratio"] = max(v for v in r["ratio_to_bound"].values() if v == v)
            r["detected"] = r["max_ratio"] >= 10
        res["red"] = red
        why = []
        gp = res["gate_proof"]
        if gp["forward_code_files"] != ["<wg_forward>"] or not gp["instance_bound"]:
            why.append(f"GDN forward is not patch.py's rewrite: {gp['forward_code_files']}")
        if gp["beta_calls"] < gp["gdn_layers"]:
            why.append(f"_wg_beta called {gp['beta_calls']} times for {gp['gdn_layers']} GDN layers")
        if not all(green["engaged"][k] > 0 for k in ("attn", "gdn", "conv")):
            why.append(f"packing wrappers not engaged in the stack: {green['engaged']}")
        if not green["ok"]:
            why.append(f"stack beyond bound: {green['ratio_to_bound']}")
        for k, r in red.items():
            if not r["detected"]:
                why.append(f"red arm {k} not detected (max ratio {r['max_ratio']:.3g})")
        res["why"] = why
        res["verdict"] = "PASS" if not why else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_gib"] = 0.0 if CPU else round(torch.cuda.max_memory_allocated() / 2**30, 2)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1, default=str)[:12000])
    sys.exit(0 if res["verdict"] in ("PASS", "REF") else 1)


if __name__ == "__main__":
    main()
