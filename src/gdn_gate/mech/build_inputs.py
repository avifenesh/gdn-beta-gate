"""Teacher-forcing inputs for gate_stats.py, built from saved validation traces (no model reads): prompt ids
(rl_common.Thinker.prompt, the readout's own template) + the trace's reasoning tokens, truncated to MAXT.
fixed = the warm-up's validation traces (same tokens for every checkpoint); own_<run> = each arm run's own traces."""
import glob, gzip, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "broad"))
import os
os.environ.setdefault("WG_NO_FLA", "1")
from transformers import AutoTokenizer
import rl_common as R

ROOT = Path(__file__).resolve().parents[3] / "results/ml7_4b"
SNAP = glob.glob(os.path.join(os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub")), "models--Qwen--Qwen3.5-4B/snapshots/*/"))[0]
N, MAXT = int(os.environ.get("N", 64)), int(os.environ.get("MAXT", 2048))


def take(path, th, problems):
    out = []
    for ln in gzip.open(path, "rt"):
        t = json.loads(ln)
        if t["sample"] != 0:
            continue
        out.append({"id": t["id"], "prompt": th.prompt(problems[t["id"]]), "think": t["think"][:MAXT],
                    "think_len_full": len(t["think"])})
        if len(out) == N:
            break
    return out


def main():
    tok = AutoTokenizer.from_pretrained(SNAP)
    th = R.Thinker(tok)
    problems = {json.loads(l)["id"]: json.loads(l)["problem"] for l in open(os.path.join(os.environ.get("WG_RUNS", "runs"), "ml7/data/val.jsonl"))}
    sets = {"fixed": take(ROOT / "step12_box/readout_val_warmup.json.traces.jsonl.gz", th, problems)}
    for arm in ("ctrl", "ctrlpm", "wide1", "wide"):
        for s in range(3):
            d = "step12_ctrlpm" if arm == "ctrlpm" else "step12_box"
            sets[f"own_{arm}_s{s}"] = take(ROOT / d / f"readout_val_{arm}_{s}.json.traces.jsonl.gz", th, problems)
    json.dump({"N": N, "MAXT": MAXT, "sets": sets}, open(sys.argv[1], "w"))
    for k, v in sets.items():
        print(k, len(v), sum(len(x["think"]) for x in v))


if __name__ == "__main__":
    main()
