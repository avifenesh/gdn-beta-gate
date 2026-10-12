"""Item-matched decomposition of the MATH-500 / GSM8K accuracy gap at the 8,192 budget from the saved step 14 readout
rows (no new reads): split items by whether the ctrl trace (same seed, same sample index) closed its thinking inside
8,192, and compare each arm's accuracy on the same items. Also bins items by the untouched model's reasoning length."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("results/ml7_4b")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mech")
ARMS = ["ctrl", "ctrlpm", "wide1", "wide"]


def rows(set_name, arm, seed):
    d = ROOT / ("step14_ctrlpm" if arm == "ctrlpm" else "step14")
    f = d / (f"readout_{set_name}_{arm}_{seed}.json" if seed is not None else f"readout_{set_name}_{arm}.json")
    R = json.load(open(f))["rows"]
    return {(r["id"], r["sample"]): (int(r["caps"]["8192"][0]), int(r["closed"] and r["think_len"] <= 8192), min(r["think_len"], 8192)) for r in R}


def main():
    res = {}
    for sn in ("math500", "gsm8k"):
        base = rows(sn, "untouched", None)
        # untouched-length tertiles: per item mean over samples
        by_item = {}
        for (i, s), v in base.items():
            by_item.setdefault(i, []).append(v[2])
        ul = {i: np.mean(v) for i, v in by_item.items()}
        q = np.quantile(list(ul.values()), [1 / 3, 2 / 3])
        tert = {i: int(np.searchsorted(q, x)) for i, x in ul.items()}
        out = {"ctrl_closed": {}, "ctrl_cut": {}, "tertile_by_untouched_length": {}, "tertile_edges": q.tolist()}
        for arm in ARMS:
            a_cl, a_cut, n_cl, n_cut = [], [], [], []
            tt = {0: [], 1: [], 2: []}
            tl = {0: [], 1: [], 2: []}
            for s in range(3):
                C = rows(sn, "ctrl", s)
                A = rows(sn, arm, s)
                keys = sorted(C)
                cl = [k for k in keys if C[k][1]]
                cu = [k for k in keys if not C[k][1]]
                a_cl.append(np.mean([A[k][0] for k in cl])); a_cut.append(np.mean([A[k][0] for k in cu]) if cu else np.nan)
                n_cl.append(len(cl)); n_cut.append(len(cu))
                for t in range(3):
                    ks = [k for k in keys if tert[k[0]] == t]
                    tt[t].append(np.mean([A[k][0] for k in ks])); tl[t].append(np.mean([A[k][2] for k in ks]))
            out["ctrl_closed"][arm] = {"acc": float(np.mean(a_cl)), "per_seed": [float(x) for x in a_cl], "n_per_seed": n_cl}
            out["ctrl_cut"][arm] = {"acc": float(np.nanmean(a_cut)), "per_seed": [float(x) for x in a_cut], "n_per_seed": n_cut}
            out["tertile_by_untouched_length"][arm] = {str(t): {"acc": float(np.mean(tt[t])), "reasoning": float(np.mean(tl[t]))} for t in range(3)}
        res[sn] = out
        print("==", sn, "tertile edges (untouched tokens)", [round(x) for x in q])
        for arm in ARMS:
            print(f"{arm:7s} on ctrl-closed items acc {out['ctrl_closed'][arm]['acc']:.3f} (n {out['ctrl_closed'][arm]['n_per_seed']})"
                  f"  on ctrl-cut items acc {out['ctrl_cut'][arm]['acc']:.3f} (n {out['ctrl_cut'][arm]['n_per_seed']})  tertiles "
                  + " ".join(f"{out['tertile_by_untouched_length'][arm][str(t)]['acc']:.3f}/{out['tertile_by_untouched_length'][arm][str(t)]['reasoning']:.0f}" for t in range(3)))
    json.dump(res, open(OUT / "item_match.json", "w"), indent=1)


if __name__ == "__main__":
    main()
