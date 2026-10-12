#!/usr/bin/env python3
"""Step 2 base battery for one Olmo-Hybrid lesion arm (venv <venv>: the pinned versions plus lm_eval).

Scorers:
  mmlu_cloze   MMLU-Pro test (12,032 at --mmlu_stride 1), cloze: the option whose text has the highest mean
               log-likelihood given the question (same prompt and truncation as retrofit/mmlu_cloze.py)
  lm-eval      arc_challenge, gsm8k, humaneval, mbpp_plus, swde, fda, squad_completion (task defaults: shots,
               greedy decoding, metrics), and RULER tasks at each --ruler_lengths length (one lm-eval run per
               length so --ruler_limit applies per length)
The arm is installed in the in-memory model (patch_olmo) and lm-eval wraps that model object, so every scorer sees
the arm. Per-item results are kept (lm-eval samples, cloze items) for paired arm-vs-stock analysis.
Writes <out>/<arm>/{mmlu_cloze.json, lmeval_<group>.json, samples_<group>/, receipt.json}; resumable per group.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_olmo as P  # noqa: E402

LM_TASKS = ["arc_challenge", "gsm8k", "humaneval", "mbpp_plus", "swde", "fda", "squad_completion"]
RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multiquery", "niah_multivalue", "ruler_vt",
               "ruler_cwe", "ruler_qa_squad"]


@torch.no_grad()
def mmlu_cloze(model, tok, stride, limit=0):
    from datasets import load_dataset

    dec, head = model.model, model.lm_head
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [ds[i] for i in range(0, len(ds), stride)]
    if limit:
        rows = rows[:limit]
    correct = {}
    for r in rows:
        ctx = tok.encode(f"The following is a question about {r['category']}.\n\nQuestion: {r['question']}\nAnswer:",
                         add_special_tokens=False)[-2000:]
        opts = [tok.encode(" " + o, add_special_tokens=False)[:200] for o in r["options"]]
        seqs = [ctx + o for o in opts]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        h = dec(input_ids=ids.cuda()).last_hidden_state
        scores = []
        for i, o in enumerate(opts):
            pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(o), device=h.device)
            lp = F.log_softmax(head(h[i, pos]).float(), -1)
            scores.append(lp.gather(-1, torch.tensor(o, device=h.device)[:, None]).mean().item())
        correct[r["question_id"]] = int(int(torch.tensor(scores).argmax()) == r["answer_index"])
    return {"n": len(correct), "cloze_acc": sum(correct.values()) / len(correct), "stride": stride, "items": correct}


def run_lmeval(lm, tasks, out_dir: Path, group: str, limit, metadata=None):
    from lm_eval import simple_evaluate

    res = simple_evaluate(model=lm, tasks=tasks, limit=limit or None, log_samples=True, confirm_run_unsafe_code=True,
                          metadata=metadata, random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                          fewshot_random_seed=1234)
    sdir = out_dir / f"samples_{group}"
    sdir.mkdir(exist_ok=True)
    for t, samples in res.get("samples", {}).items():
        with open(sdir / f"{t}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps({k: s.get(k) for k in ("doc_id", "target", "filtered_resps", "resps") if k in s}
                                   | {m: s[m] for m in s if m in ("exact_match", "acc", "acc_norm", "pass@1", "contains")
                                      or m.isdigit()}, default=str) + "\n")
    keep = {"results": res["results"], "n-samples": res.get("n-samples"), "configs_versions": res.get("versions")}
    json.dump(keep, open(out_dir / f"lmeval_{group}.json", "w"), indent=1, default=str)
    return keep["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="stock", help="stock | half | erase | layer:I | heads:K | rand:K:SEED")
    ap.add_argument("--calib", default="", help="lesion_nll calib.json (head ranking) for heads/rand arms")
    ap.add_argument("--groups", default="mmlu,lm,ruler")
    ap.add_argument("--mmlu_stride", type=int, default=1)
    ap.add_argument("--mmlu_limit", type=int, default=0)
    ap.add_argument("--lm_tasks", default=",".join(LM_TASKS))
    ap.add_argument("--lm_limit", type=int, default=0)
    ap.add_argument("--ruler_tasks", default=",".join(RULER_TASKS))
    ap.add_argument("--ruler_lengths", default="4096,8192,16384")
    ap.add_argument("--ruler_limit", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ruler_tokens", type=int, default=24576,
                    help="RULER batch = max(1, min(--batch, ruler_tokens // length)); 8 x 8192 OOMed on the 23.4 GiB workstation")
    a = ap.parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    P.install(model)
    parts = a.arm.split(":")
    if parts[0] in P.MODES:
        desc = P.set_arm(model, parts[0])
    elif parts[0] == "layer":
        desc = P.set_arm(model, "layer", layer=int(parts[1]))
    elif parts[0] in ("heads", "rand"):
        stats = json.load(open(a.calib))["stats"]
        heads = P.rank_heads(stats, int(parts[1])) if parts[0] == "heads" else P.random_heads(stats, int(parts[1]), int(parts[2]))
        desc = P.set_arm(model, "heads", heads=heads)
    else:
        raise ValueError(a.arm)
    out = Path(a.out) / a.arm.replace(":", "_")
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "receipt.json"
    rec = json.load(open(rec_path)) if rec_path.exists() else {"timing_s": {}}
    rec.update({"arm": a.arm, "desc": desc, "argv": sys.argv, "host": platform.node(),
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "transformers": P.transformers.__version__,
                "git": subprocess.run(["git", "-C", str(Path(__file__).parent), "describe", "--always", "--dirty"],
                                      capture_output=True, text=True).stdout.strip()})
    groups = a.groups.split(",")
    if "mmlu" in groups and not (out / "mmlu_cloze.json").exists():
        t0 = time.time()
        r = mmlu_cloze(model, tok, a.mmlu_stride, a.mmlu_limit)
        rec["timing_s"]["mmlu"] = round(time.time() - t0, 1)
        json.dump(r, open(out / "mmlu_cloze.json", "w"))
        json.dump(rec, open(rec_path, "w"), indent=1)
        print("MMLU", json.dumps({"n": r["n"], "acc": round(r["cloze_acc"], 4), "s": rec["timing_s"]["mmlu"]}), flush=True)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch, max_length=32768)
    if "lm" in groups and not (out / "lmeval_lm.json").exists():
        t0 = time.time()
        r = run_lmeval(lm, a.lm_tasks.split(","), out, "lm", a.lm_limit)
        rec["timing_s"]["lm"] = round(time.time() - t0, 1)
        json.dump(rec, open(rec_path, "w"), indent=1)
        print("LM", json.dumps(r, default=str)[:2000], flush=True)
    if "ruler" in groups:
        for L in [int(x) for x in a.ruler_lengths.split(",")]:
            g = f"ruler{L}"
            if (out / f"lmeval_{g}.json").exists():
                continue
            t0 = time.time()
            lm.batch_size_per_gpu = max(1, min(a.batch, a.ruler_tokens // L))
            rec.setdefault("ruler_batch", {})[g] = lm.batch_size_per_gpu
            r = run_lmeval(lm, a.ruler_tasks.split(","), out, g, a.ruler_limit,
                           metadata={"tokenizer": a.model, "max_seq_lengths": [L]})
            lm.batch_size_per_gpu = a.batch
            rec["timing_s"][g] = round(time.time() - t0, 1)
            rec["peak_mem_gib_" + g] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            json.dump(rec, open(rec_path, "w"), indent=1)
            print(g.upper(), json.dumps(r, default=str)[:1500], flush=True)
    rec["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    json.dump(rec, open(rec_path, "w"), indent=1)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
