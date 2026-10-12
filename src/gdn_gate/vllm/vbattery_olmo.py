#!/usr/bin/env python3
"""Step 2 Olmo-Hybrid Base battery on vLLM (step 4 of the vLLM switch): the same scorers and tasks as
gdn_gate/olmo/battery.py, with lm-eval's vLLM backend in place of HFLM and the arm installed by olmo/vllm_arm.py
(stock | half | erase; the layer/heads/rand lesions exist only on the HF path).

  mmlu_cloze  battery.mmlu_cloze's prompt, truncation and mean option log-likelihood, option rows only (scoring.py)
  lm-eval     arc_challenge, gsm8k, humaneval, mbpp_plus, swde, fda, squad_completion and the RULER tasks per length,
              simple_evaluate with the same seeds, limits and metadata as battery.py
Writes <out>/<arm>/{mmlu_cloze.json, lmeval_<group>.json, samples_<group>/, receipt.json} in battery.py's format
(receipt.backend = "vllm"), so analyze scripts read both. Run only after olmo p9 parity passes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "olmo"))
import vllm_arm  # noqa: E402

LM_TASKS = ["arc_challenge", "gsm8k", "humaneval", "mbpp_plus", "swde", "fda", "squad_completion"]
RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multiquery", "niah_multivalue", "ruler_vt",
               "ruler_cwe", "ruler_qa_squad"]


def mmlu_cloze(llm, tok, stride, limit=0):
    from datasets import load_dataset
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [ds[i] for i in range(0, len(ds), stride)]
    if limit:
        rows = rows[:limit]
    from scoring import cont_logprobs

    ctxs, conts, idx = [], [], []
    for qi, r in enumerate(rows):
        ctx = tok.encode(f"The following is a question about {r['category']}.\n\nQuestion: {r['question']}\nAnswer:",
                         add_special_tokens=False)[-2000:]
        for o in r["options"]:
            ctxs.append(ctx)
            conts.append(tok.encode(" " + o, add_special_tokens=False)[:200])
            idx.append(qi)
    lps = cont_logprobs(llm, ctxs, conts, SamplingParams, TokensPrompt)
    scores = [[] for _ in rows]
    for qi, lp in zip(idx, lps):
        scores[qi].append(sum(lp) / len(lp))
    correct = {r["question_id"]: int(max(range(len(sc)), key=sc.__getitem__) == r["answer_index"])
               for r, sc in zip(rows, scores)}
    return {"n": len(correct), "cloze_acc": sum(correct.values()) / len(correct), "stride": stride, "items": correct}


def _greedy_fix(VLLM):
    """lm-eval 0.4.13 VLLM bug (found 2026-10-09 on Olmo fda): normalize_gen_kwargs returns do_sample=False without a
    temperature for tasks whose generation_kwargs omit it, modify_gen_kwargs then drops do_sample, and vLLM's
    SamplingParams default temperature is 1.0, so those tasks (fda, swde, ...) were SAMPLED while the HF backend decodes
    greedily. Measured: fda 0.279 (sampled) vs 0.756 greedy vLLM vs 0.760 HF, item agreement 0.987. Restore greedy
    whenever the request did not ask for sampling."""
    orig = VLLM.modify_gen_kwargs

    def modify(gen_kwargs, eos=None, default_max_gen_toks=256):
        sample = bool(gen_kwargs.get("do_sample", False)) or float(gen_kwargs.get("temperature", 0.0) or 0.0) > 0.0
        kw, until, mx = orig(gen_kwargs, eos=eos, default_max_gen_toks=default_max_gen_toks)
        if not sample:
            kw["temperature"] = 0.0
        return kw, until, mx

    VLLM.modify_gen_kwargs = staticmethod(modify)


def run_lmeval(lm, tasks, out_dir: Path, group: str, limit, metadata=None):
    sys.path.insert(0, str(HERE.parent / "olmo"))
    import battery as OB  # the HF battery's own writer (simple_evaluate args, sample fields, file layout)

    return OB.run_lmeval(lm, tasks, out_dir, group, limit, metadata=metadata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="stock", choices=vllm_arm.ARMS)
    ap.add_argument("--groups", default="mmlu,lm,ruler")
    ap.add_argument("--mmlu_stride", type=int, default=1)
    ap.add_argument("--mmlu_limit", type=int, default=0)
    ap.add_argument("--lm_tasks", default=",".join(LM_TASKS))
    ap.add_argument("--lm_limit", type=int, default=0)
    ap.add_argument("--ruler_tasks", default=",".join(RULER_TASKS))
    ap.add_argument("--ruler_lengths", default="4096,8192,16384")
    ap.add_argument("--ruler_limit", type=int, default=100)
    ap.add_argument("--gpu_mem", type=float, default=0.80)
    ap.add_argument("--max_num_batched_tokens", type=int, default=4096)
    a = ap.parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    # per-arm vLLM cache root (see gate_vllm.set_vllm_cache_root: compile caches ignore in-process patches)
    root = Path.home() / ".cache/gdn_gate/vllm/vllm_cache" / f"olmo_{a.arm}"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(root)
    vllm_arm.install(a.arm)
    import torch
    import vllm
    from lm_eval.models.vllm_causallms import VLLM
    _greedy_fix(VLLM)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    out = Path(a.out) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "receipt.json"
    rec = json.load(open(rec_path)) if rec_path.exists() else {"timing_s": {}}
    t0 = time.time()
    lm = VLLM(pretrained=a.model, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=a.gpu_mem,
              batch_size="auto", seed=0, enable_prefix_caching=False, mamba_ssm_cache_dtype="float32",
              max_num_batched_tokens=a.max_num_batched_tokens, max_num_seqs=128)
    rec["timing_s"]["engine"] = round(time.time() - t0, 1)
    rec.update({"arm": a.arm, "backend": "vllm", "vllm": vllm.__version__, "argv": sys.argv, "host": platform.node(),
                "gpu": torch.cuda.get_device_name(0), "conv_loader_calls": vllm_arm.LOADER["n"]})
    assert vllm_arm.LOADER["n"] > 0, "conv1d loader fix never ran: the arm wrapper is not in this process"
    groups = a.groups.split(",")
    if "mmlu" in groups and not (out / "mmlu_cloze.json").exists():
        t0 = time.time()
        r = mmlu_cloze(lm.model, tok, a.mmlu_stride, a.mmlu_limit)
        rec["timing_s"]["mmlu"] = round(time.time() - t0, 1)
        json.dump(r, open(out / "mmlu_cloze.json", "w"))
        json.dump(rec, open(rec_path, "w"), indent=1)
        print("MMLU", json.dumps({"n": r["n"], "acc": round(r["cloze_acc"], 4), "s": rec["timing_s"]["mmlu"]}), flush=True)
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
            r = run_lmeval(lm, a.ruler_tasks.split(","), out, g, a.ruler_limit,
                           metadata={"tokenizer": a.model, "max_seq_lengths": [L]})
            rec["timing_s"][g] = round(time.time() - t0, 1)
            json.dump(rec, open(rec_path, "w"), indent=1)
            print(g.upper(), json.dumps(r, default=str)[:1500], flush=True)
    rec["wrapper_calls"] = vllm_arm.CALLS["n"]
    if getattr(vllm_arm, "LAYERS", None):
        rec["gdn_layers_seen"] = len(vllm_arm.LAYERS)
    json.dump(rec, open(rec_path, "w"), indent=1)


if __name__ == "__main__":
    main()
