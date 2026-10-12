#!/usr/bin/env python3
"""4B re-score parts on vLLM (vLLM venv, GPU): one engine per arm type, every checkpoint of that type scored in turn
by writing its merged LoRA and gates in place (gate_vllm.set_checkpoint). Inputs are the frozen token ids from
prep_4b_items.py; the receipts carry the same keys and item ids as rescore_4b.py's, so the two compare item by item.

  cloze   mean log-likelihood of each option's tokens after the context (prompt_logprob_token_ids from the option rows only, scoring.py),
          argmax over options vs answer (rescore_4b.cloze arithmetic)
  letter  greedy single token restricted to the item's letter ids (allowed_token_ids), i.e. the argmax of the letter
          logits (general_eval.mmlu_pro)
  he      greedy, 384 new tokens, stop strings general_eval.STOPS, program run as general_eval.run_program
  nll     mean next-token NLL per chunk (prompt_logprobs=0), both replay corpora
tier1 stays on the HF path (its labels are not the next token, so prompt logprobs cannot score it; 5 min per ckpt).

Usage: vscore_4b.py --items I.json --engine wide|wide1|stock [--decay] --job NAME=CKPT[:strip|:base] ...
                    --out_dir O [--parts cloze,letter,he,nll] [--cloze_stride K] [--model DIR]
--cloze_stride K scores every K-th cloze question (rescore_4b.py --cloze_stride: rows 0, K, 2K, ... of the MMLU-Pro
test split, which is the order prep_4b_items.py writes). The 9B re-scores use K = 4 and --model on the 9B base: its
tokenizer files are byte-identical to the 4B's, so the frozen items are the same (vscore9b_spark.sh).
"""
import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_GDN_DECODE_KERNEL", "triton")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import gate_vllm as NV  # noqa: E402


# general_eval.run_program / _limits, verbatim (general_eval imports peft, absent from the vLLM venv)
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


def prompt_lps(llm, seqs, TokensPrompt, SamplingParams):
    outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs],
                        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0), use_tqdm=False)
    res = []
    for s, o in zip(seqs, outs):
        pl = o.prompt_logprobs
        res.append([pl[i][s[i]].logprob for i in range(1, len(s))])
    return res


def part_cloze(llm, I, TP, SP, limit=0, stride=1):
    """Option logprobs from the option rows only (scoring.py): the head and vocabulary log-softmax run only on
    option tokens, as in rescore_4b.cloze. stride K keeps rows 0, K, 2K, ... (rescore_4b.cloze's selection)."""
    from scoring import cont_logprobs

    rows = I["cloze"][::stride][: limit or None]
    prompts, conts, idx = [], [], []
    for qi, r in enumerate(rows):
        for oi, o in enumerate(r["opts"]):
            prompts.append(r["ctx"])
            conts.append(o)
            idx.append((qi, oi))
    lps = cont_logprobs(llm, prompts, conts, SP, TP)
    scores = [[None] * len(r["opts"]) for r in rows]
    for (qi, oi), lp in zip(idx, lps):
        scores[qi][oi] = sum(lp) / len(lp)
    items = {}
    for r, sc in zip(rows, scores):
        items[str(r["qid"])] = int(max(range(len(sc)), key=lambda i: sc[i]) == r["answer"])
    return sum(items.values()) / len(items), items, scores


def part_letter(llm, I, TP, SP, limit=0):
    rows = I["letter"][: limit or None]
    L = I["letter_ids"]
    outs = llm.generate([TP(prompt_token_ids=r["ids"]) for r in rows],
                        [SP(max_tokens=1, temperature=0.0, allowed_token_ids=L[: r["n"]]) for r in rows],
                        use_tqdm=False)
    items = {}
    for r, o in zip(rows, outs):
        items[str(r["qid"])] = int(L.index(o.outputs[0].token_ids[0]) == r["answer"])
    return sum(items.values()) / len(items), items


def part_he(llm, tok, I, TP, SP, limit=0):
    rows = I["he"][: limit or None]
    prompts = [TP(prompt_token_ids=tok.encode(r["prompt"], add_special_tokens=False)) for r in rows]
    outs = llm.generate(prompts, SP(max_tokens=I["he_max_new"], temperature=0.0, stop=I["stops"],
                                    skip_special_tokens=True), use_tqdm=False)
    items, comps = {}, {}
    for r, o in zip(rows, outs):
        comp = o.outputs[0].text
        cut = min([comp.find(st) for st in I["stops"] if st in comp] + [len(comp)])  # no-op with stop strings
        comp = comp[:cut]
        prog = r["prompt"] + comp + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n"
        items[r["task_id"]] = int(run_program(prog))
        comps[r["task_id"]] = comp
    return sum(items.values()) / len(items), items, comps


def part_nll(llm, I, TP, SP, limit=0):
    out = {}
    for k, ch in I["nll"].items():
        ch = ch[: limit or None]
        lps = prompt_lps(llm, ch, TP, SP)
        v = [-sum(lp) / len(lp) for lp in lps]
        out[k] = {"mean": sum(v) / len(v), "per_chunk": v}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--engine", choices=("wide", "wide1", "stock"), required=True)
    ap.add_argument("--decay", action="store_true")
    ap.add_argument("--job", action="append", required=True, help="NAME=CKPT[:strip|:base]")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--parts", default="cloze,letter,he,nll")
    ap.add_argument("--gpu_mem", type=float, default=0.60)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--max_num_batched_tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cloze_stride", type=int, default=1, help="every K-th cloze question (rescore_4b.py --cloze_stride)")
    ap.add_argument("--model", help="model dir (default: the one the items were built from)")
    a = ap.parse_args()
    NV.install(a.engine, decay=a.decay)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm

    I = json.load(open(a.items))
    I["model"] = a.model or I["model"]
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    llm = LLM(model=I["model"], **NV.engine_kwargs(a.engine != "stock", gpu_memory_utilization=a.gpu_mem,
                                                    max_model_len=4096, max_num_seqs=a.max_num_seqs,
                                                    max_num_batched_tokens=a.max_num_batched_tokens))
    t_engine = time.time() - t0
    tok = AutoTokenizer.from_pretrained(I["model"])
    model = NV.get_model(llm)
    base = NV.BaseWeights(I["model"], sorted(NV.gdn_layers(model)))
    layout = NV.check_layout(model, base)
    parts = a.parts.split(",")
    for job in a.job:
        name, spec = job.split("=", 1)
        ckpt, _, mode = spec.partition(":")
        out_path = os.path.join(a.out_dir, f"{name}.json")
        res = json.load(open(out_path)) if os.path.exists(out_path) else {}
        state = None if mode == "base" else torch.load(ckpt, map_location="cpu")
        t1 = time.time()
        rec = NV.set_checkpoint(model, base, state, gate=(mode != "strip" and a.engine != "stock"), decay=a.decay)
        llm.reset_prefix_cache()
        res.update({"name": name, "ckpt": ckpt, "mode": mode or "trained", "engine": a.engine, "decay": a.decay,
                    "model": I["model"], "items": a.items,
                    "set_checkpoint": rec, "layout": layout, "host": socket.gethostname(),
                    "gpu": torch.cuda.get_device_name(0), "vllm": vllm.__version__, "torch": torch.__version__,
                    "argv": sys.argv, "smoke": bool(a.limit),
                    "git_describe": subprocess.run(["git", "-C", HERE, "describe", "--always", "--dirty"],
                                                   capture_output=True, text=True).stdout.strip()})
        sec = res.setdefault("seconds", {})
        sec["engine_build_shared"] = t_engine
        sec["set_checkpoint"] = time.time() - t1

        def save():
            json.dump(res, open(out_path + ".tmp", "w"))
            os.replace(out_path + ".tmp", out_path)

        if "cloze" in parts:
            t = time.time()
            acc, items, scores = part_cloze(llm, I, TokensPrompt, SamplingParams, a.limit, a.cloze_stride)
            res["cloze"] = {"acc": acc, "n": len(items), "stride": a.cloze_stride, "items": items, "scores": scores}
            sec["cloze"] = time.time() - t
            save()
            print(name, "cloze", round(acc, 4), round(sec["cloze"]), "s", flush=True)
        if "letter" in parts:
            t = time.time()
            acc, items = part_letter(llm, I, TokensPrompt, SamplingParams, a.limit)
            res["letter"] = {"acc": acc, "n": len(items), "items": items}
            sec["letter"] = time.time() - t
            save()
            print(name, "letter", round(acc, 4), round(sec["letter"]), "s", flush=True)
        if "he" in parts:
            t = time.time()
            acc, items, comps = part_he(llm, tok, I, TokensPrompt, SamplingParams, a.limit)
            res["he"] = {"pass1": acc, "n": len(items), "items": items, "completions": comps}
            sec["he"] = time.time() - t
            save()
            print(name, "he", round(acc, 4), round(sec["he"]), "s", flush=True)
        if "nll" in parts:
            t = time.time()
            res["nll"] = part_nll(llm, I, TokensPrompt, SamplingParams, a.limit)
            sec["nll"] = time.time() - t
            save()
            print(name, "nll", {k: round(v["mean"], 5) for k, v in res["nll"].items()}, round(sec["nll"]), "s", flush=True)
        sec["total_this_job"] = time.time() - t1
        save()
        print("DONE", name, round(time.time() - t1), "s", flush=True)


if __name__ == "__main__":
    main()
