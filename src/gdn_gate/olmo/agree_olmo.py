#!/usr/bin/env python3
"""Item-level agreement between the HF Base battery (battery.py) and the vLLM one (vbattery_olmo.py), one arm.

Per part (mmlu_cloze, each lm-eval task and metric): items compared (by question_id / doc_id), agreement rate,
HF and vLLM scores, discordant counts (HF higher / vLLM higher) and the exact two-sided sign test (McNemar for 0/1
items), plus the rate of byte-identical generated text for generate_until tasks. mbpp_plus keeps no per-item
metric, so it is judged on its aggregate (two-proportion z test). Speed per item from each receipt's own timings;
the vLLM side is charged its engine start (load + compile) as well.
Switch rule (fixed before the run, same bar as the 4B switch, agree_4b.py): a part passes when sign-test p >= 0.05
and agreement >= 0.97 for log-likelihood parts (mmlu_cloze, arc_challenge) or >= 0.90 for greedy generation parts
(one flipped token changes a whole answer); a group is switchable when every part passes and vLLM is at least 2x
faster per item including its engine start.
usage: agree_olmo.py --hf HF_ARM_DIR --v VLLM_ARM_DIR --groups mmlu,lm [--out JSON]
"""
import argparse
import json
from math import comb, erf, sqrt
from pathlib import Path

LOGLIK = {"mmlu_cloze", "arc_challenge"}
METRICS = ("acc", "acc_norm", "exact_match", "pass@1", "contains")


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def z_p(p1, p2, n1, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    z = abs(p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def paired(name, h, v, extra=None):
    ids = [k for k in h if k in v]
    b = sum(h[k] > v[k] for k in ids)
    c = sum(h[k] < v[k] for k in ids)
    agree = sum(h[k] == v[k] for k in ids) / len(ids)
    p = sign_p(b, c)
    thr = 0.97 if name.split(":")[0] in LOGLIK else 0.90
    r = {"n": len(ids), "agree": agree, "score_hf": sum(h[k] for k in ids) / len(ids),
         "score_vllm": sum(v[k] for k in ids) / len(ids), "hf_higher": b, "vllm_higher": c, "sign_p": p,
         "agree_bar": thr, "pass": agree >= thr and p >= 0.05}
    if extra:
        r.update(extra)
    return r


def load_samples(p):
    return {json.loads(l)["doc_id"]: json.loads(l) for l in open(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--v", required=True)
    ap.add_argument("--groups", default="mmlu,lm")
    ap.add_argument("--out")
    ap.add_argument("--only", default="", help="comma list of task names to compare (others skipped)")
    ap.add_argument("--no_speed", action="store_true", help="agreement only: the speed bar was settled by an earlier run")
    a = ap.parse_args()
    H, V = Path(a.hf), Path(a.v)
    hr, vr = json.load(open(H / "receipt.json")), json.load(open(V / "receipt.json"))
    eng = vr["timing_s"].get("engine", 0.0)
    out = {"hf": str(H), "vllm": str(V), "vllm_engine_s": eng, "groups": {}}
    for g in a.groups.split(","):
        parts, n_items = {}, 0
        if g == "mmlu":
            hi, vi = json.load(open(H / "mmlu_cloze.json"))["items"], json.load(open(V / "mmlu_cloze.json"))["items"]
            parts["mmlu_cloze"] = paired("mmlu_cloze", hi, vi)
            n_h, n_v = len(hi), len(vi)
        else:
            hl, vl = json.load(open(H / f"lmeval_{g}.json")), json.load(open(V / f"lmeval_{g}.json"))
            n_h = sum(x["effective"] for x in hl["n-samples"].values())
            n_v = sum(x["effective"] for x in vl["n-samples"].values())
            only = set(x for x in a.only.split(",") if x)
            for f in sorted((H / f"samples_{g}").glob("*.jsonl")):
                t = f.stem
                if only and t not in only:
                    continue
                vf = V / f"samples_{g}" / f.name
                if not vf.exists():
                    parts[t] = {"pass": False, "error": "missing on vLLM side"}
                    continue
                hs, vs = load_samples(f), load_samples(vf)
                ids = [k for k in hs if k in vs]
                same_text = sum(hs[k].get("filtered_resps") == vs[k].get("filtered_resps") for k in ids) / len(ids)
                keys = [m for m in hs[ids[0]] if m in METRICS or m.isdigit()]
                for m in keys:
                    hv = {k: float(hs[k][m]) for k in ids}
                    vv = {k: float(vs[k][m]) for k in ids}
                    parts[f"{t}:{m}"] = paired(f"{t}:{m}", hv, vv, {"same_text": same_text})
                if not keys:  # mbpp_plus: aggregate only
                    hk = [k for k in hl["results"][t] if k.endswith(",none") and "stderr" not in k and k != "alias"]
                    m = hk[0]
                    p1, p2 = float(hl["results"][t][m]), float(vl["results"][t][m])
                    p = z_p(p1, p2, len(hs), len(vs))
                    parts[f"{t}:{m}"] = {"n": len(ids), "score_hf": p1, "score_vllm": p2, "z_p": p, "same_text": same_text,
                                         "note": "no per-item metric kept; aggregate two-proportion test",
                                         "pass": p >= 0.05}
        th, tv = hr["timing_s"].get(g), vr["timing_s"].get(g)
        sph = th / n_h if th else None
        spv = tv / n_v if tv else None
        spv_e = (tv + eng) / n_v if tv else None
        sp = sph / spv_e if sph and spv_e else None
        out["groups"][g] = {"parts": parts, "n_hf": n_h, "n_vllm": n_v, "sec_hf": th, "sec_vllm": tv,
                            "sec_per_item_hf": sph, "sec_per_item_vllm": spv, "sec_per_item_vllm_with_engine": spv_e,
                            "speedup_with_engine": sp, "speedup_no_engine": (sph / spv) if sph and spv else None,
                            "all_parts_pass": all(p["pass"] for p in parts.values()),
                            "switchable": all(p["pass"] for p in parts.values()) and (a.no_speed or bool(sp and sp >= 2.0))}
    out["switchable"] = all(x["switchable"] for x in out["groups"].values())
    print(json.dumps({g: {k: v for k, v in x.items() if k != "parts"} | {"parts": {p: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for p, r in x["parts"].items()}} for g, x in out["groups"].items()}, indent=1))
    print("SWITCHABLE" if out["switchable"] else "NOT SWITCHABLE")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0 if out["switchable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
