#!/usr/bin/env python3
"""Step 2 lesion: held-out NLL of Olmo-Hybrid under each beta arm (patch_olmo), per domain and position bucket.

For every arm the same fixed chunk slice is scored (prefix of each domain in prep_nll.py's file), so arm-vs-stock
differences are paired by chunk. Per chunk we keep the summed NLL in position buckets, which lets the analysis
bootstrap over chunks. Head ranking for the top-heads arm comes from a calibration slice taken from the END of
each domain (disjoint from the scored prefix), counted under the stock model.

Arms (comma list in --arms): stock, half, erase, layers (24 runs: erase on one GDN layer), heads:K (top-K heads by
stock beta > 1 frequency), rand:K:SEED (K random heads, matched-count control). Resumable: arms already in
<out>/arms.jsonl are skipped. Writes one JSON line per arm plus <out>/calib.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_olmo as P  # noqa: E402

BUCKETS = [(0, 128), (128, 512), (512, 2048)]  # position of the predicted token, 1-based


@torch.no_grad()
def chunk_nll(model, ids):
    """ids [B, L] -> per-sequence summed NLL per position bucket [B, nb] and token counts [nb]."""
    m = model
    h = m.model(input_ids=ids).last_hidden_state[:, :-1]
    tgt = ids[:, 1:]
    B, T, D = h.shape
    nll = torch.empty(B, T, device=ids.device, dtype=torch.float32)
    for b in range(B):
        for s in range(0, T, 1024):
            lg = m.lm_head(h[b, s:s + 1024]).float()
            nll[b, s:s + 1024] = F.cross_entropy(lg, tgt[b, s:s + 1024], reduction="none")
    pos = torch.arange(1, T + 1, device=ids.device)  # position of the predicted token
    out, cnt = [], []
    for a, z in BUCKETS:
        sel = (pos >= a) & (pos < z)
        out.append(nll[:, sel].sum(1))
        cnt.append(int(sel.sum()))
    return torch.stack(out, 1).double().cpu(), cnt


def score(model, data, n_chunks, batch):
    res, t0, ntok = {}, time.time(), 0
    for dom, x in data.items():
        x = x[:n_chunks]
        rows = []
        for s in range(0, x.shape[0], batch):
            ids = x[s:s + batch].long().cuda()
            r, cnt = chunk_nll(model, ids)
            rows.append(r)
            ntok += ids.numel()
        res[dom] = {"sum": torch.cat(rows).tolist(), "count": cnt}
    torch.cuda.synchronize()
    return res, ntok, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="stock,half,erase")
    ap.add_argument("--n_chunks", type=int, default=256, help="scored chunks per domain (prefix)")
    ap.add_argument("--n_chunks_sweep", type=int, default=128, help="per-layer and head arms: chunks per domain")
    ap.add_argument("--calib_chunks", type=int, default=32, help="calibration chunks per domain, from the end")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = torch.load(a.data)
    meta_data = raw.pop("meta")
    data = {d: v[:, : a.seq] for d, v in raw.items()}
    for d, v in data.items():
        assert v.shape[0] >= a.n_chunks + a.calib_chunks, (d, v.shape)
    done = set()
    if (out / "arms.jsonl").exists():
        done = {json.loads(l)["name"] for l in open(out / "arms.jsonl")}
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    P.install(model)
    env = {"torch": torch.__version__, "transformers": P.transformers.__version__, "gpu": torch.cuda.get_device_name(0),
           "host": platform.node(), "model": a.model, "data": a.data, "data_meta": meta_data,
           "git": subprocess.run(["git", "-C", str(Path(__file__).parent), "describe", "--always", "--dirty"],
                                 capture_output=True, text=True).stdout.strip(),
           "argv": sys.argv}
    calib_path = out / "calib.json"
    if calib_path.exists():
        calib = json.load(open(calib_path))
    else:
        P.set_arm(model, "stock")
        P.enable_stats(model, True)
        cal = {d: v[-a.calib_chunks:] for d, v in data.items()}
        _, ntok, sec = score(model, cal, a.calib_chunks, a.batch)
        calib = {"stats": P.collect_stats(model), "tokens": ntok, "seconds": sec, "env": env}
        P.enable_stats(model, False)
        json.dump(calib, open(calib_path, "w"))
        print("CALIB", json.dumps({"tokens": ntok, "seconds": round(sec, 1), "tok_per_s": round(ntok / sec)}),
              flush=True)
    stats = calib["stats"]
    n_gdn = len(P.gdn_layers(model))
    plan = []
    for spec in a.arms.split(","):
        if spec in P.MODES:
            plan.append((spec, spec, {}, a.n_chunks))
        elif spec == "layers":
            plan += [(f"layer{i:02d}", "layer", {"layer": i}, a.n_chunks_sweep) for i in range(n_gdn)]
        elif spec.startswith("heads:"):
            k = int(spec.split(":")[1])
            plan.append((f"heads{k}", "heads", {"heads": P.rank_heads(stats, k)}, a.n_chunks_sweep))
        elif spec.startswith("rand:"):
            _, k, seed = spec.split(":")
            plan.append((f"rand{k}s{seed}", "heads", {"heads": P.random_heads(stats, int(k), int(seed))},
                         a.n_chunks_sweep))
        else:
            raise ValueError(spec)
    for name, arm, kw, n in plan:
        if name in done:
            print("skip", name, flush=True)
            continue
        desc = P.set_arm(model, arm, **kw)
        P.enable_stats(model, True)
        res, ntok, sec = score(model, data, n, a.batch)
        st = P.collect_stats(model)
        P.enable_stats(model, False)
        tok_w = [t for t in st["tokens"]]
        frac = sum(sum(r) / len(r) * t for r, t in zip(st["beta_gt1"], tok_w)) / sum(tok_w)
        rec = {"name": name, "desc": desc, "n_chunks": n, "seq": a.seq, "buckets": BUCKETS, "tokens": ntok,
               "seconds": round(sec, 1), "tok_per_s": round(ntok / sec, 1), "beta_gt1_frac": frac,
               "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "domains": res}
        summ = {d: round(sum(map(sum, r["sum"])) / (len(r["sum"]) * sum(r["count"])), 5) for d, r in res.items()}
        with open(out / "arms.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("ARM", name, json.dumps({"nll": summ, "tok_per_s": rec["tok_per_s"], "beta_gt1": round(frac, 4)}),
              flush=True)
    json.dump(env, open(out / "env.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
