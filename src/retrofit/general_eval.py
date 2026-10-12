#!/usr/bin/env python3
"""General-capability regression for one trained run (the card's "MMLU-Pro / code-eval" gate).

Rebuilds base + LoRA (+ widened gate for the wide arm) from <run_dir>/trainable.pt, then:
  MMLU-Pro (TIGER-Lab/MMLU-Pro test, all 12,032): zero-shot, the next-token logit over the option
    letters decides; accuracy and per-item correctness.
  HumanEval (openai/openai_humaneval, 164): greedy completion, pass@1; each program runs in a
    subprocess with a timeout and CPU/memory limits.
Writes <run_dir>/general.json.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import patch
import train

LETTERS = "ABCDEFGHIJ"
STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n```"]


def build(model_path, run_dir, qlora=False, ckpt="trainable.pt"):
    res = next(os.path.join(run_dir, f) for f in ("result.json", "result_sweep.json", "ckpt_meta.json")
               if os.path.exists(os.path.join(run_dir, f)))
    meta = json.load(open(res))["meta"]
    arm, rank = meta["args"]["arm"], meta["args"]["rank"]
    kw = dict(dtype=torch.bfloat16, device_map={"": 0})
    if qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    patch.install_for_arm(model, arm)
    model = train.add_lora(model, None, rank)
    state = torch.load(os.path.join(run_dir, ckpt))
    params = dict(model.named_parameters())
    missing = [k for k in state if k not in params]
    assert not missing, missing[:5]
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].dtype))
    return model.eval(), arm


@torch.no_grad()
def mmlu_pro(model, tok, limit=None, batch=8):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    if limit:
        ds = ds.select(range(limit))
    letter_ids = [tok.encode(" " + c, add_special_tokens=False) for c in LETTERS]
    assert all(len(x) == 1 for x in letter_ids), letter_ids
    letter_ids = [x[0] for x in letter_ids]
    items = []
    for r in ds:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
        prompt = (f"The following is a multiple choice question about {r['category']}. "
                  f"Answer with the letter of the correct option.\n\nQuestion: {r['question']}\n"
                  f"Options:\n{opts}\nAnswer:")
        items.append((r["question_id"], tok.encode(prompt, add_special_tokens=False)[-3000:],
                      len(r["options"]), r["answer_index"]))
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
    correct = {}
    dec = model.get_base_model().model
    head = model.get_base_model().lm_head
    for s in range(0, len(order), batch):
        chunk = [items[i] for i in order[s:s + batch]]
        L = max(len(c[1]) for c in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, c in enumerate(chunk):
            ids[i, :len(c[1])] = torch.tensor(c[1])
            att[i, :len(c[1])] = 1
        h = dec(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
        last = h[torch.arange(len(chunk)), torch.tensor([len(c[1]) - 1 for c in chunk])]
        logits = head(last).float()[:, letter_ids]
        for i, c in enumerate(chunk):
            pred = int(logits[i, :c[2]].argmax())
            correct[c[0]] = int(pred == c[3])
    return sum(correct.values()) / len(correct), correct


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def run_program(code, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        open(p, "w").write(code)
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, timeout=timeout,
                               preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


@torch.no_grad()
def humaneval(model, tok, limit=None, batch=16, max_new=384):
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(limit))
    tok.padding_side = "left"
    res = {}
    rows = list(ds)
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        enc = tok([r["prompt"] for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
        out = model.generate(input_ids=enc.input_ids.cuda(), attention_mask=enc.attention_mask.cuda(),
                             max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or 0)
        for i, r in enumerate(chunk):
            comp = tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True)
            cut = min([comp.find(st) for st in STOPS if st in comp] + [len(comp)])
            comp = comp[:cut]
            prog = r["prompt"] + comp + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n"
            res[r["task_id"]] = int(run_program(prog))
    tok.padding_side = "right"
    return sum(res.values()) / len(res), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qlora", action="store_true", help="4-bit base (smoke tests only)")
    ap.add_argument("--ckpt", default="trainable.pt", help="trainable-state file in run_dir, e.g. ckpt_step1600.pt")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    model, arm = build(a.model, a.run_dir, a.qlora, a.ckpt)
    m_acc, m_items = mmlu_pro(model, tok, a.limit or None)
    t1 = time.time()
    h_acc, h_items = humaneval(model, tok, a.limit or None)
    out = {"arm": arm, "ckpt": a.ckpt, "mmlu_pro_acc": m_acc, "mmlu_pro_n": len(m_items), "humaneval_pass1": h_acc,
           "humaneval_n": len(h_items), "mmlu_seconds": t1 - t0, "humaneval_seconds": time.time() - t1,
           "mmlu_items": m_items, "humaneval_items": h_items, "smoke": bool(a.limit or a.qlora)}
    name = "general_smoke.json" if out["smoke"] else "general.json"
    json.dump(out, open(os.path.join(a.run_dir, name), "w"))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_items")}))


if __name__ == "__main__":
    main()
