"""Text-level statistics of already-saved readout traces (no new model reads): where the reasoning tokens go for each
arm. Inputs: the step 12 validation traces and the step 14 test traces of the 4B record (saved by readout_vllm.py).
Per trace: reasoning length, whether thinking closed inside 8,192 tokens, correctness at 8,192 (from the readout rows),
the token position where the gold answer value first appears in the reasoning (numeric golds only), the share of
reasoning spent after that point, and counts of re-check markers. Output JSON + figures."""
import glob, gzip, json, re, sys
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/ml7_4b")
DATA = Path(os.environ.get("WG_RUNS", "runs")) / "ml7/data"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mech")
TOK = Tokenizer.from_file(glob.glob(os.path.join(os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub")), "models--Qwen--Qwen3.5-4B/snapshots/*/tokenizer.json"))[0])
ARMS = ["untouched", "warmup", "ctrl", "ctrlpm", "wide1", "wide"]
COL = {"untouched": "#bbbbbb", "warmup": "#555555", "ctrl": "#7f7f7f", "ctrlpm": "#2a7bbd", "wide1": "#d4572a", "wide": "#8c3fb0"}
SETS = {"val": ("val.jsonl", ["step12_box", "step12_ctrlpm"]),
        "gsm8k": ("test_gsm8k.jsonl", ["step14", "step14_ctrlpm"]),
        "math500": ("test_math500.jsonl", ["step14", "step14_ctrlpm"])}
MARKERS = {
    "wait": r"\bwait\b",
    "alternatively": r"\balternative(ly)?\b",
    "check_verify": r"\b(double[- ]?check|re-?check|check|verify|verification|confirm)\w*",
    "hmm_actually": r"\b(hmm+|actually)\b",
    "self_correction": r"self[- ]correct",
    "lets_re": r"\blet me (re|try again|go back|recalculate|reconsider)",
}
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def num_pattern(gold):
    g = gold.strip().replace(",", "")
    if not NUM.match(g):
        return None
    if "." in g:
        g = g.rstrip("0").rstrip(".")
    # the value as a standalone number, allowing thousands separators
    body = re.escape(g)
    if len(g.lstrip("-")) > 3 and "." not in g:
        s = g.lstrip("-")
        parts = []
        while s:
            parts.insert(0, s[-3:]); s = s[:-3]
        body = ("-" if g.startswith("-") else "") + r",?".join(parts)
    return re.compile(r"(?<![\d.])" + body + r"(?:\.0+)?(?![\d])")


def load_rows(path):
    d = json.load(open(path))
    return {(r["id"], r["sample"]): r for r in d["rows"]}, d


def label_of(f):
    m = re.match(r"readout_\w+?_(untouched|warmup|ctrl|ctrlpm|wide1|wide)(?:_(\d))?\.json$", Path(f).name)
    return (m.group(1), int(m.group(2)) if m.group(2) else None) if m else None


def files_for(set_name):
    _, dirs = SETS[set_name]
    seen = {}
    for dd in dirs:
        for f in sorted(glob.glob(str(ROOT / dd / f"readout_{set_name}_*.json"))):
            lab = label_of(f)
            if lab and lab not in seen:
                seen[lab] = f
    return seen


def trace_stats(think_ids, gold_pat):
    text = TOK.decode(think_ids)
    n = len(think_ids)
    low = text.lower()
    st = {k: len(re.findall(p, low)) for k, p in MARKERS.items()}
    first = None
    if gold_pat is not None:
        m = gold_pat.search(text)
        if m:
            first = int(round(n * m.start() / max(1, len(text))))
    return n, first, st


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for set_name, (dfile, _) in SETS.items():
        gold = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(DATA / dfile)}
        pats = {k: num_pattern(v) for k, v in gold.items()}
        per_arm = defaultdict(list)
        for (arm, seed), f in files_for(set_name).items():
            rows, _ = load_rows(f)
            recs = []
            for ln in gzip.open(f + ".traces.jsonl.gz", "rt"):
                t = json.loads(ln)
                r = rows[(t["id"], t["sample"])]
                n, first, mk = trace_stats(t["think"], pats.get(t["id"]))
                c8 = r["caps"]["8192"]
                recs.append({"id": t["id"], "n": min(n, 8192), "closed": int(r["closed"] and r["think_len"] <= 8192),
                             "correct": int(c8[0]), "first": first, "numeric": pats.get(t["id"]) is not None, **mk})
            per_arm[arm].append({"seed": seed, "file": f, "recs": recs})
        out = {}
        for arm, runs in per_arm.items():
            seeds = []
            for run in runs:
                R = run["recs"]
                num = [x for x in R if x["numeric"]]
                found = [x for x in num if x["first"] is not None and x["closed"]]
                closed = [x for x in R if x["closed"]]
                cut = [x for x in R if not x["closed"]]
                s = {"seed": run["seed"], "file": run["file"], "n_traces": len(R),
                     "reasoning_mean": float(np.mean([x["n"] for x in R])),
                     "closed_share": float(np.mean([x["closed"] for x in R])),
                     "acc_8192": float(np.mean([x["correct"] for x in R])),
                     "acc_given_closed": float(np.mean([x["correct"] for x in closed])) if closed else None,
                     "acc_given_cut": float(np.mean([x["correct"] for x in cut])) if cut else None,
                     "numeric_items_share": len(num) / len(R),
                     "gold_found_share_closed_numeric": len(found) / max(1, len([x for x in num if x["closed"]])),
                     "tokens_to_first_gold": float(np.mean([x["first"] for x in found])) if found else None,
                     "tokens_after_first_gold": float(np.mean([x["n"] - x["first"] for x in found])) if found else None,
                     "share_after_first_gold": float(np.mean([(x["n"] - x["first"]) / max(1, x["n"]) for x in found])) if found else None,
                     }
                for k in MARKERS:
                    s[f"{k}_per_trace"] = float(np.mean([x[k] for x in R]))
                    s[f"{k}_per_1k"] = float(1000 * sum(x[k] for x in R) / max(1, sum(x["n"] for x in R)))
                seeds.append(s)
            keys = [k for k in seeds[0] if k not in ("seed", "file") and all(isinstance(z[k], (int, float)) for z in seeds)]
            out[arm] = {"seeds": seeds, "mean": {k: float(np.mean([z[k] for z in seeds])) for k in keys}}
        res[set_name] = out
        print("==", set_name)
        for arm in ARMS:
            if arm in out:
                m = out[arm]["mean"]
                print(f"{arm:9s} len {m['reasoning_mean']:7.0f} closed {m['closed_share']:.3f} acc8k {m['acc_8192']:.3f} "
                      f"acc|closed {m.get('acc_given_closed', float('nan')):.3f} to_gold {m.get('tokens_to_first_gold', float('nan')):6.0f} "
                      f"after_gold {m.get('tokens_after_first_gold', float('nan')):6.0f} share_after {m.get('share_after_first_gold', float('nan')):.2f} "
                      f"wait {m['wait_per_trace']:.2f} alt {m['alternatively_per_trace']:.2f} check {m['check_verify_per_trace']:.2f} "
                      f"check/1k {m['check_verify_per_1k']:.2f}")
    json.dump(res, open(OUT / "text_stats.json", "w"), indent=1)

    # figure: tokens before vs after the first appearance of the gold value, per arm, per set
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for i, sn in enumerate(SETS):
        arms = [a for a in ARMS if a in res[sn]]
        to = [res[sn][a]["mean"].get("tokens_to_first_gold", 0) for a in arms]
        af = [res[sn][a]["mean"].get("tokens_after_first_gold", 0) for a in arms]
        ax[i].bar(arms, to, color="#999999", label="before gold value first appears")
        ax[i].bar(arms, af, bottom=to, color=[COL[a] for a in arms], label="after (re-checking)")
        ax[i].set_title(f"{sn}: closed traces with a numeric gold"); ax[i].set_ylabel("reasoning tokens")
        ax[i].tick_params(axis="x", rotation=30)
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "text_tokens_before_after_gold.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    sn = "math500"
    arms = [a for a in ARMS if a in res[sn]]
    cs = [res[sn][a]["mean"]["closed_share"] for a in arms]
    ac = [res[sn][a]["mean"].get("acc_given_closed", 0) for a in arms]
    ax[0].bar(arms, cs, color=[COL[a] for a in arms]); ax[0].set_title("MATH-500: share of traces that close inside 8,192")
    ax[1].bar(arms, ac, color=[COL[a] for a in arms]); ax[1].set_title("MATH-500: accuracy of the traces that close")
    ax[1].set_ylim(0.8, 1.0)
    for a in ax: a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(OUT / "text_math500_closure.png", dpi=130)


if __name__ == "__main__":
    main()
