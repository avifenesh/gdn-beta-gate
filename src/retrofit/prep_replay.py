#!/usr/bin/env python3
"""Pack WikiText-103 (raw) into 512-token chunks for the replay stream and the held-out NLL guard."""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--train_tokens", type=int, default=20_000_000)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.model)
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
out = {}
for split, cap in (("train", args.train_tokens), ("test", None)):
    buf = []
    for row in ds[split]:
        t = row["text"]
        if not t.strip():
            continue
        buf += tok.encode(t, add_special_tokens=False)
        if cap and len(buf) >= cap:
            break
    n = len(buf) // 512
    out[split] = torch.tensor(buf[: n * 512], dtype=torch.int64).view(n, 512).to(torch.int32)
    print(split, out[split].shape)
torch.save(out, args.out)
