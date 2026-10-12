#!/usr/bin/env python3
"""Step 3 (4B part): re-score one banked Qwen3.5-4B retrofit checkpoint on one GPU, as trained or as its strip twin.

Model: base + LoRA (+ zero-init gates) rebuilt from <run_dir>/<ckpt> by general_eval.build (unchanged lane code).
Strip twin (--strip, eval only): every widened-beta gate weight (gate_w) zeroed, so beta = sigmoid(b) exactly, the
LoRA and any decay gate (wg_wa, wideg arm) kept. The receipt checks that the twin's beta > 1 fraction is 0 and,
for the trained arm, that the gate is engaged (beta > 1 fraction > 0).

Battery (each part timed, written as it finishes, resumable):
  tier1   train.eval_tasks at lengths 64/256/1024, 128 sequences, test seed 10_000, called as train.py's final eval
  nll     train.eval_nll on 256 x 512 test chunks (randperm seed 1234) of BOTH replay corpora: WikiText-103 and clean
  letter  general_eval.mmlu_pro: MMLU-Pro test, all 12,032, next-token letter logit
  he      general_eval.humaneval: 164 problems, greedy, pass@1
  cloze   MMLU-Pro cloze, the same scoring as retrofit/mmlu_cloze.py (mean option log-likelihood, bf16 autocast,
          the same prompt and truncation), every --cloze_stride-th question (1 = all 12,032), item-level output
Usage: rescore_4b.py --run_dir D --name N --out_dir O [--strip] [--ckpt trainable.pt] [--parts tier1,nll,letter,he,cloze]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "retrofit"))
import general_eval as G  # noqa: E402
import patch  # noqa: E402
import tasks  # noqa: E402
import train  # noqa: E402

MODEL = os.path.join(os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
                     "models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/")
_RUNS = os.environ.get("WG_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "runs"))
REPLAYS = {"wikitext": os.path.join(_RUNS, "retrofit/replay_wikitext103.pt"),
           "clean": os.path.join(_RUNS, "retrofit/box/replay_clean.pt")}


@torch.no_grad()
def cloze(model, tok, stride):
    """retrofit/mmlu_cloze.py's scoring, unchanged, returning item-level correctness."""
    from datasets import load_dataset

    dec, head = model.get_base_model().model, model.get_base_model().lm_head
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [ds[i] for i in range(0, len(ds), stride)]
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
    return sum(correct.values()) / len(correct), correct


def strip(model):
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gate_w.weight"):
                p.zero_()
                n += 1
    return n


def gate_norm(model):
    return sum(p.float().norm().item() ** 2 for n, p in model.named_parameters() if n.endswith("gate_w.weight")) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="trainable.pt")
    ap.add_argument("--strip", action="store_true")
    ap.add_argument("--parts", default="tier1,nll,letter,he,cloze")
    ap.add_argument("--cloze_stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke: MMLU/HumanEval item limit, tier-1 n_seq 8")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"{a.name}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    model, arm = G.build(MODEL, a.run_dir, ckpt=a.ckpt)
    res.update({"name": a.name, "run_dir": a.run_dir, "ckpt": a.ckpt, "arm": arm, "strip": a.strip,
                "model": MODEL, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "argv": sys.argv,
                "git_describe": subprocess.run(["git", "-C", HERE, "describe", "--always", "--dirty"],
                                               capture_output=True, text=True).stdout.strip()})
    res.setdefault("seconds", {})["build"] = time.time() - t0
    res["gate_norm_trained"] = gate_norm(model)
    if a.strip:
        n = strip(model)
        assert n == 24, n  # 24 GDN layers in Qwen3.5-4B
        res["stripped_gates"] = n
        res["gate_norm_after_strip"] = gate_norm(model)
    model.eval()

    def save():
        json.dump(res, open(out_path + ".tmp", "w"))
        os.replace(out_path + ".tmp", out_path)

    parts = a.parts.split(",")
    if "tier1" in parts and "tier1" not in res:
        t0 = time.time()
        V = tasks.Vocab(tok)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        patch.enable_stats(model, True)
        r = train.eval_tasks(model, V, [64, 256, 1024], 8 if a.limit else 128, 8, pad_id)
        st = patch.collect_stats(model)
        patch.enable_stats(model, False)
        model.eval()
        summ = {}
        for t in tasks.ALL_TASKS:
            for L in (64, 256, 1024):
                summ[f"{t}@{L}"] = sum(r[f"{t}@{L}"]) / L
        res["tier1"] = {"summary": summ, "raw": r, "wg_stats": st}
        if st["beta_gt1_frac_per_layer"]:
            res["tier1"]["beta_gt1_frac_mean"] = sum(st["beta_gt1_frac_per_layer"]) / len(st["beta_gt1_frac_per_layer"])
        res["seconds"]["tier1"] = time.time() - t0
        save()
        print("tier1", json.dumps({k: round(v, 4) for k, v in summ.items()}), res["tier1"].get("beta_gt1_frac_mean"),
              flush=True)
    if "nll" in parts and "nll" not in res:
        t0 = time.time()
        res["nll"] = {}
        for k, p in REPLAYS.items():
            ch = train.replay_chunks(p, "test", 8 if a.limit else 256, 512, 1234)
            v = train.eval_nll(model, ch.long(), 8)
            model.eval()
            res["nll"][k] = {"mean": sum(v) / len(v), "per_chunk": v}
        res["seconds"]["nll"] = time.time() - t0
        save()
        print("nll", {k: round(v["mean"], 5) for k, v in res["nll"].items()}, flush=True)
    if "letter" in parts and "letter" not in res:
        t0 = time.time()
        acc, items = G.mmlu_pro(model, tok, a.limit or None)
        res["letter"] = {"acc": acc, "n": len(items), "items": items}
        res["seconds"]["letter"] = time.time() - t0
        save()
        print("letter", round(acc, 4), len(items), flush=True)
    if "he" in parts and "he" not in res:
        t0 = time.time()
        acc, items = G.humaneval(model, tok, a.limit or None)
        model.eval()
        res["he"] = {"pass1": acc, "n": len(items), "items": items}
        res["seconds"]["he"] = time.time() - t0
        save()
        print("he", round(acc, 4), len(items), flush=True)
    if "cloze" in parts and "cloze" not in res:
        t0 = time.time()
        stride = a.cloze_stride if not a.limit else max(a.cloze_stride, 12032 // max(1, a.limit))
        acc, items = cloze(model, tok, stride)
        res["cloze"] = {"acc": acc, "n": len(items), "stride": stride, "items": items}
        res["seconds"]["cloze"] = time.time() - t0
        save()
        print("cloze", round(acc, 4), len(items), flush=True)
    # gate engagement checks (the twin must not reach beta > 1; a trained wide arm must)
    if "tier1" in res and arm in ("wide", "wideg"):
        f = res["tier1"].get("beta_gt1_frac_mean")
        res["check_gate"] = {"beta_gt1_frac_mean": f, "pass": (f == 0.0) if a.strip else (f is not None and f > 0.0)}
    # wide1 (beta <= 1 by construction; its stats slot is the share at the ceiling, which is not 0 at W = 0): the
    # trained gate must be non-zero, the strip twin's zero
    if arm == "wide1":
        g = res["gate_norm_after_strip"] if a.strip else res["gate_norm_trained"]
        res["check_gate"] = {"gate_norm": g, "ceiling_frac_mean": res.get("tier1", {}).get("beta_gt1_frac_mean"),
                             "pass": (g == 0.0) if a.strip else (g > 0.0)}
    res["seconds"]["total_this_call"] = time.time() - t_start
    res["smoke"] = bool(a.limit)
    save()
    print("DONE", a.name, round(time.time() - t_start), "s", flush=True)


if __name__ == "__main__":
    main()
