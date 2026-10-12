#!/usr/bin/env python3
"""Per-layer beta > 1 usage of Olmo-Hybrid-7B Base by RULER task (2026-10-09, after the depth localization).

Question: does each GDN layer's share of the negative range (beta > 1, i.e. a negative eigenvalue of the state
transition) on counting prompts track the CWE cost of erasing that layer's quarter (Q1 -1.8, Q2 -2.6, Q3 -0.7,
Q4 -1.7 pt at 8k)? And do counting prompts use the range more than retrieval or QA prompts?

Stock arm, vLLM eager (graph replays skip Python, so STATS needs enforce_eager), olmo/vllm_arm.STATS per layer.
Each task runs through lm-eval exactly as the battery does (same seeds and metadata), --limit items at one length;
the counts cover every token the GDN layers see (prefill and the few generated tokens).
Writes OUT.json: per task, per GDN layer {gt1_frac}, plus the overall fraction and accuracy on the slice.
usage: usage_probe.py --model DIR --out OUT.json [--length 8192] [--limit 40] [--tasks a,b,...]"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_CACHE_ROOT", str(Path.home() / ".cache/gdn_gate/vllm/vllm_cache/olmo_stock_eager"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vllm"))
import vllm_arm  # noqa: E402

TASKS = "ruler_cwe,ruler_fwe,niah_single_1,niah_multikey_1,ruler_vt,ruler_qa_squad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    a = ap.parse_args()
    vllm_arm.install("stock")
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM
    import vbattery_olmo as VB

    VB._greedy_fix(VLLM)
    lm = VLLM(pretrained=a.model, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=a.gpu_mem,
              batch_size="auto", seed=0, enable_prefix_caching=False, mamba_ssm_cache_dtype="float32",
              max_num_batched_tokens=4096, max_num_seqs=32, enforce_eager=True)
    n_layers = 24
    out = {"model": a.model, "length": a.length, "limit": a.limit, "argv": sys.argv, "tasks": {}}
    for t in a.tasks.split(","):
        vllm_arm.STATS.update({"on": True, "beta_max": 0.0, "gt1": 0, "n": 0, "per_layer": {}})
        t0 = time.time()
        res = simple_evaluate(model=lm, tasks=[t], limit=a.limit, confirm_run_unsafe_code=True,
                              metadata={"tokenizer": a.model, "max_seq_lengths": [a.length]}, random_seed=0,
                              numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)
        vllm_arm.STATS["on"] = False
        pl = vllm_arm.STATS["per_layer"]
        assert sorted(pl) == list(range(n_layers)), f"{t}: layers seen {sorted(pl)}"
        acc = {k: v for k, v in res["results"][t].items() if isinstance(v, float)}
        out["tasks"][t] = {"gt1_frac": vllm_arm.STATS["gt1"] / max(vllm_arm.STATS["n"], 1),
                           "beta_max": vllm_arm.STATS["beta_max"], "tokens_x_heads": vllm_arm.STATS["n"],
                           "per_layer": [pl[k][0] / max(pl[k][1], 1) for k in range(n_layers)],
                           "acc": acc, "seconds": round(time.time() - t0, 1)}
        print(t, round(out["tasks"][t]["gt1_frac"], 4), [round(x, 3) for x in out["tasks"][t]["per_layer"]],
              flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    q = lambda v, g: sum(v[6 * g:6 * g + 6]) / 6  # noqa: E731
    out["quarters"] = {t: [round(q(d["per_layer"], g), 4) for g in range(4)] for t, d in out["tasks"].items()}
    json.dump(out, open(a.out, "w"), indent=1)
    print("quarters", out["quarters"])


if __name__ == "__main__":
    main()
