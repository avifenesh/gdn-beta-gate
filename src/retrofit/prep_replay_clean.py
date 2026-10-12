#!/usr/bin/env python3
"""Clean replay corpus: FineWeb-Edu (English web, normal formatting) plus Python source (and, with --he_frac,
Hebrew web text from FineWeb-2 heb_Hebr), packed into 512-token
chunks in prep_replay.py's format ({'train','test'} int32 [N, 512]).

Why: the WikiText-103 replay taught the model WikiText's tokenized formatting (" , ", " @-@ ", lines that start
with a space). Main-run HumanEval completions had correct bodies followed by "\\n def ..." that the harness could
not cut, and pass@1 fell from 50 to about 27. Documents are separated by the EOS token.
"""
import argparse
import itertools

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--train_tokens", type=int, default=16_000_000)
ap.add_argument("--test_tokens", type=int, default=400_000)
ap.add_argument("--code_frac", type=float, default=0.25)
ap.add_argument("--he_frac", type=float, default=0.0, help="share of Hebrew web text (FineWeb-2 heb_Hebr, ODC-BY)")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.model)
eos = tok.eos_token_id


def stream(name, config, field, split="train"):
    ds = load_dataset(name, config, split=split, streaming=True) if config else load_dataset(name, split=split, streaming=True)
    for row in ds:
        t = row[field]
        if t and t.strip():
            yield t


def take(gen, n_tokens):
    buf = []
    for t in gen:
        buf += tok.encode(t, add_special_tokens=False) + [eos]
        if len(buf) >= n_tokens:
            break
    return buf


web = stream("HuggingFaceFW/fineweb-edu", "sample-10BT", "text")
code = stream("codeparrot/codeparrot-clean-valid", None, "content")
heweb = stream("HuggingFaceFW/fineweb-2", "heb_Hebr", "text") if args.he_frac > 0 else None
out = {}
for split, total in (("test", args.test_tokens), ("train", args.train_tokens)):  # test first: disjoint docs
    n_code, n_he = int(total * args.code_frac), int(total * args.he_frac)
    a, b = take(web, total - n_code - n_he), take(code, n_code)
    h = take(heweb, n_he) if heweb is not None else []
    # interleave 512-token chunks of each source so replay batches mix them
    ca = [a[i : i + 512] for i in range(0, len(a) - 511, 512)]
    cb = [b[i : i + 512] for i in range(0, len(b) - 511, 512)]
    ch = [h[i : i + 512] for i in range(0, len(h) - 511, 512)]
    g = torch.Generator().manual_seed(0 if split == "train" else 1)
    chunks = torch.tensor(ca + cb + ch, dtype=torch.int32)
    out[split] = chunks[torch.randperm(len(chunks), generator=g)]
    print(split, out[split].shape, "web chunks", len(ca), "code chunks", len(cb), "hebrew chunks", len(ch), flush=True)
torch.save(out, args.out)
print("sample:", repr(tok.decode(out["train"][0][:80].tolist())))
