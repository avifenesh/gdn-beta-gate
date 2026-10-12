#!/usr/bin/env python3
"""Training cap from the 40-step ctrl pilots (rule pre-registered in ml_pipeline/PIPELINE.md, 2026-10-08 about 21:06):
score = normalized area under accuracy vs log2 budget, pilot minus the untouched step 9 curve, paired over items
(compare_readout.compare, z_cont); 95% interval D +- 1.96 |D / z|. Largest gain wins; a cap whose interval contains the
winner's point estimate ties; ties go to the cheaper (smaller) cap.
usage: pilot_pick.py BASE.json CAP=PILOT.json [CAP=PILOT.json ...] --out OUT.json"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_readout import compare, load_rows  # noqa: E402


def main():
    argv = sys.argv[1:]
    out = argv[argv.index("--out") + 1]
    argv = argv[:argv.index("--out")]
    base, pilots = argv[0], dict(a.split("=", 1) for a in argv[1:])
    rb, caps, _ = load_rows([base])
    res = {"base": base, "rule": __doc__.split("usage")[0].strip(), "pilots": {}}
    for cap, f in sorted(pilots.items(), key=lambda kv: int(kv[0])):
        ra, c2, _ = load_rows([f])
        assert c2 == caps, (f, c2, caps)
        t = compare(ra, rb, caps)["tests"]
        a = t["norm_auc_log2"]
        D, z = a["diff_a_minus_b"], a["z"]
        half = 1.96 * abs(D / z) if z and not math.isnan(z) and not math.isinf(z) else math.inf
        res["pilots"][cap] = {"file": f, "auc_gain": D, "z": z, "ci95": [D - half, D + half],
                              "acc_gain": {k: v["diff_a_minus_b"] for k, v in t.items() if k.startswith("acc@")}}
    best = max(res["pilots"], key=lambda c: res["pilots"][c]["auc_gain"])
    bD = res["pilots"][best]["auc_gain"]
    tied = [c for c, v in res["pilots"].items() if v["ci95"][0] <= bD <= v["ci95"][1]]
    res.update(best=int(best), tied=sorted(int(c) for c in tied), cap=min(int(c) for c in tied))
    json.dump(res, open(out, "w"), indent=1)
    for c, v in res["pilots"].items():
        print(c, round(v["auc_gain"], 4), [round(x, 4) for x in v["ci95"]], round(v["z"], 2))
    print("CAP", res["cap"], "best", best, "tied", res["tied"])


if __name__ == "__main__":
    main()
