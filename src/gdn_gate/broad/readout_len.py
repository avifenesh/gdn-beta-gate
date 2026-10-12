#!/usr/bin/env python3
"""Step 7 readout: accuracy vs reasoning tokens at several reasoning budgets on held-out MATH-500.

Budget semantics: the budget counts reasoning tokens only (tokens before </think>). Phase 1 samples once per item
with the Qwen3.5 card's thinking-mode shape (temperature 1.0, top_p 0.95, top_k 20, presence penalty 1.5), up to the
largest budget plus --answer_max tokens. For a budget c, an item whose reasoning closed within c tokens keeps the
phase-1 answer (sampling up to c is the same draw at every budget, so this is exactly what a run at budget c would
have produced, given the sampled prefix); an item still thinking at c is cut at c, closed with "\\n</think>\\n\\n"
(forced) and its answer sampled in phase 2. So one long sample per item serves every budget, and smaller budgets
are exact prefixes of it.

Reports per budget: accuracy, mean reasoning tokens used (min(think, c)), the forced (truncated) share, and silent
stops (the model ended inside its thinking). Per-item records go to the JSON for paired seed-level analysis.

Usage: readout_len.py --model M --data math500.jsonl --out FILE [--ckpt PT --arm A [--strip]] [--caps 512,...,8192]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import arms
import rl_common as R


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="none")
    ap.add_argument("--arm", default="ctrl", choices=arms.ARMS)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--strip", action="store_true")
    ap.add_argument("--caps", default="512,1024,2048,4096,8192")
    ap.add_argument("--answer_max", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gen_batch", type=int, default=64)
    ap.add_argument("--sample_seed", type=int, default=20261006)
    ap.add_argument("--mem_frac", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    dev = torch.device("cuda:0" if a.device == "cuda" else a.device)
    if dev.type == "cuda" and a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    dmap = {"": 0} if dev.type == "cuda" else {"": "cpu"}
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    if a.ckpt == "none":
        model, _ = arms.load_model(a.model, a.arm, device_map=dmap, dtype=dt)
        info = {"arm": a.arm, "untouched": True}
    else:
        model, info = arms.load_trained(a.model, a.ckpt, a.arm, a.rank, strip=a.strip, device_map=dmap, dtype=dt)
    model.eval()
    th = R.Thinker(tok)
    rows = R.load_jsonl(a.data)[:: a.stride][: a.limit or None]
    caps = sorted(int(c) for c in a.caps.split(","))
    big = caps[-1]
    prompts = [th.prompt(r["problem"]) for r in rows]

    torch.manual_seed(a.sample_seed)
    t1 = time.time()
    ph1 = []
    for s in range(0, len(prompts), a.gen_batch):
        ph1 += th._sample(model, prompts[s : s + a.gen_batch], big + a.answer_max, dev, R.VENDOR)
    t_ph1 = time.time() - t1
    recs = []
    for r, p, g in zip(rows, prompts, ph1):
        closed = th.end_think in g
        k = g.index(th.end_think) if closed else len(g)
        silent = (not closed) and bool(g) and g[-1] in th.stops
        think = g[:k] if closed else (g[:-1] if silent else g)
        recs.append({"id": r["id"], "gold": r["answer"], "p": p, "g": g, "think_len": len(think), "closed": closed,
                     "silent_stop": silent, "think": think})

    res = {c: {} for c in caps}
    t2 = time.time()
    for c in caps:
        need = []
        for i, x in enumerate(recs):
            if x["closed"] and x["think_len"] <= c:
                text = tok.decode(x["g"][x["think_len"] :], skip_special_tokens=True)
                ended = x["g"][-1] in th.stops
                res[c][x["id"]] = {"correct": int(R.correct(text, x["gold"])), "reasoning": x["think_len"], "forced": 0,
                                   "answer_ended": int(ended)}
            else:
                need.append(i)
        for s in range(0, len(need), a.gen_batch):
            idx = need[s : s + a.gen_batch]
            ctx = [recs[i]["p"] + recs[i]["think"][:c] + th.force for i in idx]
            outs = th._sample(model, ctx, a.answer_max, dev, R.VENDOR,
                              prior=[recs[i]["think"][:c] + th.force for i in idx])
            for i, o in zip(idx, outs):
                x = recs[i]
                text = tok.decode(o, skip_special_tokens=True)
                res[c][x["id"]] = {"correct": int(R.correct(text, x["gold"])), "reasoning": min(x["think_len"], c),
                                   "forced": 1, "answer_ended": int(bool(o) and o[-1] in th.stops)}
    t_ph2 = time.time() - t2
    summ = {}
    for c in caps:
        v = list(res[c].values())
        summ[c] = {"acc": sum(x["correct"] for x in v) / len(v), "reasoning_mean": sum(x["reasoning"] for x in v) / len(v),
                   "forced_frac": sum(x["forced"] for x in v) / len(v), "n": len(v)}
    out = {"args": vars(a), "load": info, "summary": summ, "items": {str(c): res[c] for c in caps},
           "think_len": {x["id"]: x["think_len"] for x in recs}, "silent_stops": sum(x["silent_stop"] for x in recs),
           "unclosed_at_max": sum(not x["closed"] for x in recs), "phase1_seconds": t_ph1, "phase2_seconds": t_ph2,
           "phase1_tokens": sum(len(x["g"]) for x in recs), "load_seconds": t1 - t0,
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({"summary": summ, "silent_stops": out["silent_stops"], "phase1_s": round(t_ph1, 1),
                      "phase2_s": round(t_ph2, 1)}), flush=True)


if __name__ == "__main__":
    main()
