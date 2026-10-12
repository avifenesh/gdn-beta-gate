#!/usr/bin/env python3
"""Training cap from the step 9 cap-mode receipt with per-sample data (rule pre-registered in ml_pipeline/PIPELINE.md,
2026-10-08 21:10): S(cap) = E[sum |A|] / E[sum completion tokens] over groups of 8 drawn with replacement from each
item's samples, reward rl_common.length_reward (alpha 0.2), Dr. GRPO advantage A = r - mean(r). Reports the
correctness-only and length-only parts and a 95% bootstrap interval over items; ties within the interval of the best
go to the cheaper cap. Revised about 21:03 (PIPELINE.md): the cap is chosen on S_length, the length-penalty part of the
advantage per token; total S and the correctness part are reported only. usage: pick_cap.py RECEIPT.json OUT.json"""
import json
import random
import sys

sys.path.insert(0, __file__.rsplit("/ml7/", 1)[0] + "/broad")
import rl_common as R  # noqa: E402

G, DRAWS, ALPHA, SEED, BOOT = 8, 200, 0.2, 20261008, 1000


def item_stats(cor, ln, rng):
    """per item: (sum|A| full, sum|A| correctness-only, sum|A| length-only, tokens) summed over DRAWS groups"""
    k = len(cor)
    tot = [0.0, 0.0, 0.0, 0.0]
    for _ in range(DRAWS):
        idx = [rng.randrange(k) for _ in range(G)]
        c, l = [cor[i] for i in idx], [ln[i] for i in idx]
        r = R.length_reward(c, l, ALPHA)
        rc = [float(x) for x in c]
        m, mc = sum(r) / G, sum(rc) / G
        tot[0] += sum(abs(x - m) for x in r)
        tot[1] += sum(abs(x - mc) for x in rc)
        rl = [x - y for x, y in zip(r, rc)]  # the length penalty part of the reward
        ml = sum(rl) / G
        tot[2] += sum(abs(x - ml) for x in rl)
        tot[3] += sum(l)
    return tot


def main():
    rec = json.load(open(sys.argv[1]))
    grp = rec["group"]
    out = {"receipt": sys.argv[1], "rule": __doc__.split("usage")[0].strip(), "caps": {}}
    per = {}
    for cap, v in sorted(rec["caps"].items(), key=lambda kv: int(kv[0])):
        s = v["samples"]
        n = len(s["correct"]) // grp
        rng = random.Random(SEED + int(cap))
        st = []
        for i in range(n):
            sl = slice(i * grp, (i + 1) * grp)
            ln = [a + b for a, b in zip(s["think_tokens"][sl], s["tail_tokens"][sl])]
            st.append(item_stats(s["correct"][sl], ln, rng))
        per[cap] = st
        A = sum(x[0] for x in st); T = sum(x[3] for x in st)
        brng = random.Random(SEED)
        boots, bootl = [], []
        for _ in range(BOOT):
            b = [st[brng.randrange(n)] for _ in range(n)]
            boots.append(sum(x[0] for x in b) / sum(x[3] for x in b))
            bootl.append(sum(x[2] for x in b) / sum(x[3] for x in b))
        boots.sort(); bootl.sort()
        out["caps"][cap] = {"acc": v["acc"], "S_per_1k_tokens": 1000 * A / T,
                            "S_correctness_only": 1000 * sum(x[1] for x in st) / T,
                            "S_length_only": 1000 * sum(x[2] for x in st) / T,
                            "ci95": [1000 * boots[int(0.025 * BOOT)], 1000 * boots[int(0.975 * BOOT)]],
                            "ci95_length": [1000 * bootl[int(0.025 * BOOT)], 1000 * bootl[int(0.975 * BOOT)]],
                            "tokens_per_group": T / (n * DRAWS), "adv_mass_per_group": A / (n * DRAWS)}
    best = max(out["caps"], key=lambda c: out["caps"][c]["S_length_only"])
    lo = out["caps"][best]["ci95_length"][0]
    tied = [c for c in out["caps"] if out["caps"][c]["S_length_only"] >= lo]
    out["best"], out["tied_with_best"] = int(best), sorted(int(c) for c in tied)
    out["cap"] = min(tied, key=lambda c: out["caps"][c]["tokens_per_group"])
    out["cap"] = int(out["cap"])
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    for c, v in out["caps"].items():
        print(c, json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()}))
    print("CAP", out["cap"], "best", best, "tied", out["tied_with_best"])


if __name__ == "__main__":
    main()
