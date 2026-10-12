#!/usr/bin/env python3
"""MMLU-Pro scored without letters (cloze): pick the option whose text has the highest mean log-likelihood given
the question. Tests whether the letter-scored drop is letter-format interference (the tasks use letter tokens as
labels and as inputs) or lost knowledge. Stratified subset (every 4th question by default) for speed.

Usage: mmlu_cloze.py --model M --run_dir D [--ckpt trainable.pt] [--stride 4]; writes <run_dir>/mmlu_cloze.json
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

import general_eval as G


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="trainable.pt")
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model, arm = G.build(a.model, a.run_dir, ckpt=a.ckpt)
    dec, head = model.get_base_model().model, model.get_base_model().lm_head
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [ds[i] for i in range(0, len(ds), a.stride)]
    correct = {}
    for r in rows:
        ctx = tok.encode(f"The following is a question about {r['category']}.\n\nQuestion: {r['question']}\nAnswer:",
                         add_special_tokens=False)[-2000:]
        opts = [tok.encode(" " + o, add_special_tokens=False)[:200] for o in r["options"]]
        seqs = [ctx + o for o in opts]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = dec(input_ids=ids.cuda()).last_hidden_state
        scores = []
        for i, o in enumerate(opts):
            pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(o), device=h.device)
            lp = F.log_softmax(head(h[i, pos]).float(), -1)
            scores.append(lp.gather(-1, torch.tensor(o, device=h.device)[:, None]).mean().item())
        correct[r["question_id"]] = int(int(torch.tensor(scores).argmax()) == r["answer_index"])
    out = {"arm": arm, "n": len(correct), "cloze_acc": sum(correct.values()) / len(correct), "items": correct}
    json.dump(out, open(os.path.join(a.run_dir, "mmlu_cloze.json"), "w"))
    print(json.dumps({"arm": arm, "n": out["n"], "cloze_acc": round(out["cloze_acc"], 4)}))


if __name__ == "__main__":
    main()
