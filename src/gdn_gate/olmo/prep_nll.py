#!/usr/bin/env python3
"""Held-out NLL text for the Olmo-Hybrid lesion: four domains, Olmo-tokenized, packed into fixed-length chunks.

Domains (documents separated by EOS, packed, chunk order fixed per domain):
  web   HuggingFaceFW/fineweb-edu sample-10BT, skipping the first --skip_web documents (the lane's replay builder,
        prep_replay_clean.py, reads this stream from its start, so the eval text is disjoint from any replay)
  code  codeparrot/codeparrot-clean-valid, skipping the first --skip_code files (same reason)
  wiki  wikimedia/wikipedia 20231101.en
  math  HuggingFaceTB/finemath finemath-4plus
Output: {domain: int32 [N, L]} plus a meta record. The full spec is about 20M tokens (5M per domain at the default);
the runner takes a fixed prefix slice of chunks per domain, so a small slice is a subset of the full set.
"""
import argparse
import json
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--tokens_per_domain", type=int, default=5_000_000)
ap.add_argument("--skip_web", type=int, default=200_000)
ap.add_argument("--skip_code", type=int, default=20_000)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.model)
eos = tok.eos_token_id

SOURCES = {
    "web": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", args.skip_web),
    "code": ("codeparrot/codeparrot-clean-valid", None, "content", args.skip_code),
    "wiki": ("wikimedia/wikipedia", "20231101.en", "text", 0),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus", "text", 0),
}
out, meta = {}, {"chunk": args.chunk, "tokens_per_domain": args.tokens_per_domain, "sources": {}}
for dom, (name, cfg, field, skip) in SOURCES.items():
    t0 = time.time()
    ds = load_dataset(name, cfg, split="train", streaming=True) if cfg else load_dataset(name, split="train", streaming=True)
    if skip:
        ds = ds.skip(skip)
    buf, ndoc = [], 0
    for row in ds:
        t = row[field]
        if not t or not t.strip():
            continue
        buf += tok.encode(t, add_special_tokens=False) + [eos]
        ndoc += 1
        if len(buf) >= args.tokens_per_domain:
            break
    n = len(buf) // args.chunk
    out[dom] = torch.tensor(buf[: n * args.chunk], dtype=torch.int32).view(n, args.chunk)
    meta["sources"][dom] = {"dataset": name, "config": cfg, "field": field, "skip_docs": skip, "docs": ndoc,
                            "chunks": n, "seconds": round(time.time() - t0, 1)}
    print(dom, json.dumps(meta["sources"][dom]), flush=True)
out["meta"] = meta
torch.save(out, args.out)
print("saved", args.out)
