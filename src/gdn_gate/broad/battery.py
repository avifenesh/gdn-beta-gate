#!/usr/bin/env python3
"""Broad benchmark battery for step 6 checkpoints (and the untouched base), with per-item scores and a hashed
50/50 explore/confirm split.

Benches (frozen inputs from make_eval_data.py, --data):
  mmlu_cloze  MMLU-Pro, cloze: the option text with the highest mean log-likelihood after the question (the same
              context and scoring as retrofit/mmlu_cloze.py; letters are not used: the gate distorts them)
  arc_cloze   ARC-Challenge test, the same cloze scoring
  gsm8k       chat prompt, greedy, 512 new tokens, last "Answer:" number (else the last number) vs gold
  humaneval   retrofit/general_eval.humaneval unchanged (raw completion, greedy, pass@1)
  mbppplus    chat prompt with the first test, greedy, 512 new tokens, first python block, EvalPlus base+plus tests
  swde, fda, squad   Based recall prompts, raw continuation, greedy 48 tokens, contains-match (case-insensitive)
  ruler       synthetic niah_single / niah_multikey / vt at 4k and 16k, chat prompt plus answer prefix, greedy
  nll         per-domain next-token NLL on 64 x 1,024 tokens (wiki, web, code, math), plus gate usage per domain
Each item's split is sha256(SPLIT_SALT + "{bench}:{id}") parity: A = explore, B = confirm.

Usage: battery.py --model M --data DIR --out FILE.json [--ckpt PT --arm A --rank 16 [--strip]] [--benches ...]
--ckpt none scores the model as loaded (untouched base).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import arms
import patch
import sft_data

ALL = ("mmlu_cloze", "arc_cloze", "gsm8k", "humaneval", "mbppplus", "swde", "fda", "squad", "ruler", "nll")
MID = ("mmlu_cloze", "arc_cloze", "gsm8k", "swde", "fda", "nll")


SPLIT_SALT = bytes.fromhex("6e65676569672d62726f61643a").decode()  # fixed salt of the lab's A/B item split, kept byte-exact


def split_of(bench: str, item_id: str) -> str:
    return "A" if int(hashlib.sha256(f"{SPLIT_SALT}{bench}:{item_id}".encode()).hexdigest(), 16) % 2 == 0 else "B"


def jl(p):
    return [json.loads(l) for l in open(p)]


# ------------------------------------------------------------------ cloze
@torch.no_grad()
def cloze(model, tok, items, ctx_fn, dev, budget=16384, stride=1, limit=0):
    """Mean log-likelihood per option token; one sequence per (item, option), batched by padded tokens."""
    dec, head = arms.core(model)
    items = items[::stride][: limit or None]
    seqs = []  # (item index, option index, ctx ids, option ids)
    for qi, r in enumerate(items):
        ctx = tok.encode(ctx_fn(r), add_special_tokens=False)[-2000:]
        for oi, o in enumerate(r["options"]):
            seqs.append((qi, oi, ctx, tok.encode(" " + o, add_special_tokens=False)[:200]))
    order = sorted(range(len(seqs)), key=lambda i: -(len(seqs[i][2]) + len(seqs[i][3])))
    scores = {}
    i = 0
    while i < len(order):
        L = len(seqs[order[i]][2]) + len(seqs[order[i]][3])
        n = max(1, budget // L)
        chunk = [seqs[j] for j in order[i : i + n]]
        i += n
        ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
        for k, (_, _, c, o) in enumerate(chunk):
            ids[k, : len(c) + len(o)] = torch.tensor(c + o)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = dec(input_ids=ids.to(dev)).last_hidden_state  # right padding: causal, so pads never reach scores
        for k, (qi, oi, c, o) in enumerate(chunk):
            pos = torch.arange(len(c) - 1, len(c) - 1 + len(o), device=h.device)
            lp = F.log_softmax(head(h[k, pos]).float(), -1)
            scores[(qi, oi)] = lp.gather(-1, torch.tensor(o, device=h.device)[:, None]).mean().item()
    out = {}
    for qi, r in enumerate(items):
        s = [scores[(qi, oi)] for oi in range(len(r["options"]))]
        out[r["id"]] = int(max(range(len(s)), key=s.__getitem__) == r["answer_index"])
    return out


# ------------------------------------------------------------------ generation
@torch.no_grad()
def generate(model, tok, prompts: list[list[int]], max_new: int, dev, budget=65536, max_batch=64, stop_ids=None):
    """Greedy, left-padded, batched by prompt length. Returns decoded completions in input order."""
    stop_ids = stop_ids or [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    res = [None] * len(prompts)
    i = 0
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    while i < len(order):
        L = len(prompts[order[i]])
        n = max(1, min(max_batch, budget // (L + max_new)))
        idx = order[i : i + n]
        i += n
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for k, j in enumerate(idx):
            p = prompts[j]
            ids[k, L - len(p):] = torch.tensor(p)
            att[k, L - len(p):] = 1
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new,
                                 do_sample=False, eos_token_id=stop_ids, pad_token_id=pad)
        for k, j in enumerate(idx):
            res[j] = tok.decode(out[k, L:], skip_special_tokens=True)
    return res


def _limits(mem_gb):
    def f():
        resource.setrlimit(resource.RLIMIT_AS, (mem_gb << 30, mem_gb << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    return f


def run_program(code, timeout=30, mem_gb=8):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        open(p, "w").write(code)
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, timeout=timeout,
                               preexec_fn=_limits(mem_gb), env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"})
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm_pred(text):
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    s = m[-1] if m else (NUM.findall(text) or [""])[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def code_block(text):
    m = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    if m:
        return m[0]
    m = re.findall(r"```(?:python|py)?\n(.*)", text, re.S)
    return m[0] if m else text


# ------------------------------------------------------------------ benches
def b_mmlu(model, tok, data, dev, a):
    ctx = lambda r: f"The following is a question about {r['category']}.\n\nQuestion: {r['question']}\nAnswer:"  # noqa: E731
    return cloze(model, tok, jl(data / "mmlu_pro.jsonl"), ctx, dev, stride=a.mmlu_stride, limit=a.limit)


def b_arc(model, tok, data, dev, a):
    return cloze(model, tok, jl(data / "arc_c.jsonl"), lambda r: f"Question: {r['question']}\nAnswer:", dev, limit=a.limit)


def b_gsm8k(model, tok, data, dev, a):
    R = sft_data.Renderer(tok)
    rows = jl(data / "gsm8k.jsonl")[: a.limit or None]
    prompts = [R.prompt([{"role": "user", "content": r["question"] + "\nSolve it step by step, then give the final "
                          "answer on the last line as 'Answer: <number>'."}]) for r in rows]
    outs = generate(model, tok, prompts, 512, dev)
    res = {}
    for r, o in zip(rows, outs):
        p = gsm_pred(o)
        res[r["id"]] = int(p is not None and abs(p - float(r["answer"])) < 1e-6)
    return res


class _Rows(list):
    def select(self, idx):
        return _Rows(self[i] for i in idx)


def b_humaneval(model, tok, data, dev, a):
    """retrofit/general_eval.humaneval unchanged (prompt, stops, 384 greedy tokens, batch 16, sandbox), fed
    the frozen rows instead of a datasets download so the Sparks stay offline."""
    import general_eval

    rows = _Rows(jl(data / "humaneval.jsonl"))
    orig = general_eval.load_dataset
    general_eval.load_dataset = lambda *x, **k: rows
    try:
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, items = general_eval.humaneval(model, tok, a.limit or None)
    finally:
        general_eval.load_dataset = orig
    return items


def b_mbppplus(model, tok, data, dev, a):
    R = sft_data.Renderer(tok)
    rows = jl(data / "mbppplus.jsonl")[: a.limit or None]
    prompts = []
    for r in rows:
        tests = eval(r["test_list"]) if isinstance(r["test_list"], str) else r["test_list"]  # noqa: S307 (a list literal from the frozen dataset)
        u = (f"{r['prompt']}\nYour code should pass this test:\n{tests[0]}\n"
             "Write the Python function in a single ```python code block.")
        prompts.append(R.prompt([{"role": "user", "content": u}]))
    outs = generate(model, tok, prompts, 512, dev)
    progs = []
    for r, o in zip(rows, outs):
        imps = r["test_imports"]
        imps = eval(imps) if isinstance(imps, str) else imps  # noqa: S307
        progs.append("\n".join(imps) + "\n" + code_block(o) + "\n\n" + r["test"] + "\n")
    with cf.ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(run_program, progs))
    return {r["id"]: int(v) for r, v in zip(rows, ok)}


def b_recall(name):
    def f(model, tok, data, dev, a):
        rows = jl(data / f"{name}.jsonl")[: a.limit or None]
        prompts = [tok.encode(r["text"], add_special_tokens=False)[-6000:] for r in rows]
        outs = generate(model, tok, prompts, 48, dev, stop_ids=[tok.eos_token_id])
        return {r["id"]: int(r["value"].strip().lower() in o.lower()) for r, o in zip(rows, outs)}
    return f


def b_ruler(model, tok, data, dev, a):
    R = sft_data.Renderer(tok)
    rows = jl(data / "ruler.jsonl")
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    prompts = [R.prompt([{"role": "user", "content": r["user"]}]) + tok.encode(r["answer_prefix"], add_special_tokens=False)
               for r in rows]
    outs = generate(model, tok, prompts, 48, dev, budget=131072, max_batch=16)
    res = {}
    for r, o in zip(rows, outs):
        hit = [x for x in r["answers"] if x.lower() in o.lower()]
        res[r["id"]] = len(hit) / len(r["answers"])  # niah: 0/1; vt: share of the 5 variables named
    return res


@torch.no_grad()
def b_nll(model, tok, data, dev, a):
    dec, head = arms.core(model)
    doms = torch.load(data / "nll_domains.pt")
    res, gate = {}, {}
    for d, x in doms.items():
        x = x[: a.limit or None]
        patch.enable_stats(model, True)
        for i in range(0, x.shape[0], 8):
            ids = x[i : i + 8].to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = dec(input_ids=ids).last_hidden_state[:, :-1]
            for k in range(ids.shape[0]):
                lg = head(h[k]).float()
                res[f"{d}#{i + k}"] = F.cross_entropy(lg, ids[k, 1:]).item()
        st = patch.collect_stats(model)
        patch.enable_stats(model, False)
        if st["beta_gt1_frac_per_layer"]:
            gate[d] = sum(st["beta_gt1_frac_per_layer"]) / len(st["beta_gt1_frac_per_layer"])
    b_nll.gate = gate
    return res


BENCH = {"mmlu_cloze": b_mmlu, "arc_cloze": b_arc, "gsm8k": b_gsm8k, "humaneval": b_humaneval,
         "mbppplus": b_mbppplus, "swde": b_recall("swde"), "fda": b_recall("fda"), "squad": b_recall("squad"),
         "ruler": b_ruler, "nll": b_nll}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="none")
    ap.add_argument("--arm", default="ctrl", choices=arms.ARMS)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--strip", action="store_true")
    ap.add_argument("--benches", default="all", help="all, mid, or a comma list")
    ap.add_argument("--mmlu_stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="smoke: items per bench")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    benches = ALL if a.benches == "all" else MID if a.benches == "mid" else tuple(a.benches.split(","))
    assert all(b in BENCH for b in benches), benches
    from transformers import AutoTokenizer

    dev = torch.device("cuda:0" if a.device == "cuda" else a.device)
    dmap = {"": 0} if dev.type == "cuda" else {"": "cpu"}
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32  # fp32 only for CPU tests
    t0 = time.time()
    if a.ckpt == "none":
        model, _ = arms.load_model(a.model, a.arm, device_map=dmap, dtype=dt)
        model.eval()
        info = {"arm": a.arm, "strip": False, "untouched": True}
    else:
        model, info = arms.load_trained(a.model, a.ckpt, a.arm, a.rank, strip=a.strip, device_map=dmap, dtype=dt)
    out = {"args": vars(a), "load": info, "load_seconds": time.time() - t0, "bench": {}}
    prev = Path(a.out)
    if prev.exists():  # resumable per bench
        old = json.loads(prev.read_text())
        if old.get("args", {}).get("ckpt") == a.ckpt and old.get("load", {}).get("strip") == info.get("strip"):
            out["bench"] = old.get("bench", {})
    data = Path(a.data)
    for b in benches:
        if b in out["bench"]:
            continue
        t1 = time.time()
        items = BENCH[b](model, tok, data, dev, a)
        rec = {"score": sum(items.values()) / max(len(items), 1), "n": len(items), "seconds": time.time() - t1,
               "items": items, "split": {k: split_of(b, k) for k in items}}
        if b == "nll" and getattr(b_nll, "gate", None):
            rec["gate_beta_gt1_by_domain"] = b_nll.gate
        out["bench"][b] = rec
        Path(a.out).write_text(json.dumps(out))
        print(json.dumps({"bench": b, "score": round(rec["score"], 4), "n": rec["n"], "s": round(rec["seconds"], 1)}),
              flush=True)
    out["total_seconds"] = time.time() - t0
    out["gpu"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    Path(a.out).write_text(json.dumps(out))
    print("END", json.dumps({b: round(out["bench"][b]["score"], 4) for b in benches}), flush=True)


if __name__ == "__main__":
    main()
