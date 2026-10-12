#!/usr/bin/env python3
"""P9 parity: the vLLM beta arms (vllm_arm.py) against the HF arms (patch_olmo.py) on Olmo-Hybrid-Think-SFT-7B.

Prompts: the first 2 items of each think set (GPQA-D, AIME 24, AIME 25, MATH-500), rendered with the Think-SFT
embedded chat template (generation prompt opens <think>).
  --side hf      (venv with the pinned versions) per arm stock/half/erase: greedy 64-token continuation (HF generate)
                 and teacher-forced logprobs of every token of prompt + the stock continuation -> hf_ref.pt
  --side vllm    (vLLM venv, one process per arm, CUDA graphs on as in the decode runs) prompt logprobs of the same
                 token sequences (prefill path) and a greedy 64-token continuation (decode path) -> vllm_<arm>.pt
  --side compare -> JSON receipt. Distances are mean |logprob difference| over scored positions:
                 d_lesion(x) = d(HF x, HF stock); d_match(x) = d(vLLM x, HF x); d_cross(x) = d(vLLM x, HF stock).
    pass if, for x in half and erase: d_match(x) < 0.2 d_lesion(x) and d_cross(x) > 0.5 d_lesion(x) (red arm: the
    vLLM arm is not stock); for stock: d_match(stock) < 0.2 min(d_lesion); and the decode path is live: for half
    and erase the vLLM greedy continuation differs from the vLLM stock one on at least one prompt. Greedy prefix
    agreement with HF is reported, not gated (bf16 kernels differ between the stacks).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("stock", "half", "erase")
NEW = 64


def prompts(items_path, tok):
    items = [json.loads(l) for l in open(items_path)]
    pick = []
    for s in ("gpqa_diamond", "aime24", "aime25", "math500"):
        pick += [it for it in items if it["set"] == s][:2]
    return [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=True,
                                    add_generation_prompt=True, return_dict=False) for it in pick], [it["id"] for it in pick]


def side_hf(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(HERE))
    import patch_olmo as P

    tok = AutoTokenizer.from_pretrained(a.model)
    ps, ids = prompts(a.items, tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    P.install(model)
    out = {"ids": ids, "prompts": ps, "greedy": {}, "lp": {}}
    for arm in ARMS:
        P.set_arm(model, arm)
        out["greedy"][arm] = []
        for p in ps:
            x = torch.tensor([p], device="cuda")
            with torch.no_grad():
                g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=NEW,
                                   min_new_tokens=NEW, pad_token_id=tok.pad_token_id)
            out["greedy"][arm].append(g[0, len(p):].tolist())
    out["seqs"] = [p + c for p, c in zip(ps, out["greedy"]["stock"])]
    if a.seqs_from:  # score the token sequences of another reference so every distance is on the same tokens
        out["seqs"] = torch.load(a.seqs_from, weights_only=False)["seqs"]
    for arm in ARMS:
        P.set_arm(model, arm)
        out["lp"][arm] = []
        for s in out["seqs"]:
            x = torch.tensor([s], device="cuda")
            with torch.no_grad():
                lg = model(input_ids=x).logits[0, :-1].float().log_softmax(-1)
            out["lp"][arm].append(lg.gather(-1, x[0, 1:, None])[:, 0].cpu())
    out["env"] = {"torch": torch.__version__, "transformers": P.transformers.__version__,
                  "gpu": torch.cuda.get_device_name(0), "position": model._ng_position}
    torch.save(out, a.out)
    print("HF DONE", a.out, flush=True)


def side_vllm(a):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch

    sys.path.insert(0, str(HERE))
    import vllm_arm as VA

    VA.install(a.arm)
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ref = torch.load(a.ref, weights_only=False)  # our own HF-side reference file
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
              enable_prefix_caching=False, seed=0, max_num_seqs=16)
    calls_after_load = VA.CALLS["n"]
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    res = llm.generate([TokensPrompt(prompt_token_ids=s) for s in ref["seqs"]], sp_lp)
    lp = []
    for s, r in zip(ref["seqs"], res):
        pl = r.prompt_logprobs
        lp.append(torch.tensor([pl[i][s[i]].logprob for i in range(1, len(s))]))
    sp_g = SamplingParams(max_tokens=NEW, min_tokens=NEW, temperature=0.0)
    res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ref["prompts"]], sp_g)
    greedy = [list(r.outputs[0].token_ids) for r in res]
    out = {"arm": a.arm, "lp": lp, "greedy": greedy, "calls_at_load": calls_after_load, "calls_total": VA.CALLS["n"],
           "conv_loader_calls": VA.LOADER["n"],
           "env": {"vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}}
    torch.save(out, a.out)
    print("VLLM DONE", a.arm, json.dumps({"calls": VA.CALLS["n"]}), flush=True)


def side_compare(a):
    import torch

    ref = torch.load(a.ref, weights_only=False)  # our own HF-side reference file
    vl = {arm: torch.load(Path(a.vllm_dir) / f"vllm_{arm}.pt", weights_only=False) for arm in ARMS}
    plen = [len(p) for p in ref["prompts"]]

    def d(x, y):
        # positions 1.. of each sequence; skip the first 8 prompt positions (no context, large and noisy)
        return float(torch.cat([(u[8:] - v[8:]).abs() for u, v in zip(x, y)]).mean())

    hf, rec, ok = ref["lp"], {"pairs": {}, "checks": []}, True
    lesion = {x: d(hf[x], hf["stock"]) for x in ("half", "erase")}
    for x in ARMS:
        rec["pairs"][x] = {"d_match": d(vl[x]["lp"], hf[x]), "d_cross": d(vl[x]["lp"], hf["stock"]),
                           "d_lesion": lesion.get(x, 0.0), "calls_total": vl[x]["calls_total"],
                           "calls_at_load": vl[x]["calls_at_load"]}

    def check(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        rec["checks"].append({"check": name, "pass": bool(cond), **kw})
        print(("PASS " if cond else "FAIL ") + name, json.dumps(kw), flush=True)

    for x in ARMS:
        check(f"{x}: wrapper engaged", vl[x]["calls_total"] > 0, calls=vl[x]["calls_total"])
        check(f"{x}: conv1d loader fix ran for every GDN conv tensor (72)", vl[x].get("conv_loader_calls") == 72,
              conv_loader_calls=vl[x].get("conv_loader_calls"))
    for x in ("half", "erase"):
        r = rec["pairs"][x]
        check(f"{x}: vLLM matches HF arm", r["d_match"] < 0.2 * r["d_lesion"], d_match=r["d_match"], d_lesion=r["d_lesion"])
        check(f"{x}: red arm, vLLM arm is not stock", r["d_cross"] > 0.5 * r["d_lesion"], d_cross=r["d_cross"])
        diff = sum(g1 != g0 for g1, g0 in zip(vl[x]["greedy"], vl["stock"]["greedy"]))
        check(f"{x}: decode path live (greedy differs from vLLM stock)", diff > 0, prompts_differing=diff)
    check("stock: vLLM matches HF stock", rec["pairs"]["stock"]["d_match"] < 0.2 * min(lesion.values()),
          d_match=rec["pairs"]["stock"]["d_match"])

    def prefix(u, v):
        n = 0
        for p, q in zip(u, v):
            if p != q:
                break
            n += 1
        return n

    rec["greedy_prefix_vs_hf_same_arm"] = {x: [prefix(u, v) for u, v in zip(vl[x]["greedy"], ref["greedy"][x])] for x in ARMS}
    rec["greedy_prefix_vs_hf_stock"] = {x: [prefix(u, v) for u, v in zip(vl[x]["greedy"], ref["greedy"]["stock"])] for x in ARMS}
    rec["prompt_lens"] = plen
    rec["ids"] = ref["ids"]
    rec["env"] = {"hf": ref["env"], "vllm": vl["stock"]["env"]}
    rec["pass"] = ok
    json.dump(rec, open(a.out, "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["hf", "vllm", "compare"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--items", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--arm", default="stock", choices=ARMS)
    ap.add_argument("--vllm_dir", default="")
    ap.add_argument("--gpu_mem", type=float, default=0.4)
    ap.add_argument("--seqs_from", default="", help="hf side: score this reference's sequences")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    {"hf": side_hf, "vllm": side_vllm, "compare": side_compare}[a.side](a)
    print("wall_s", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
