#!/usr/bin/env python3
"""g1g2_summary.py MECH_DIR: summarize the Phase 5 mechanism readouts (validation, 500 items x 1 sample, Qwen3.5-4B).

G1 (MECH_DIR/ablate): per checkpoint condition full / strip / hybrid for wide1 and wide x seeds 0-2, plus warm-up and
ctrl x 3. G2 (MECH_DIR/dose): wide1 x seeds 0-2 with the gate scaled by 0, 0.5, 1.5, 2 (scale 1 = G1 full).
Per readout: norm_auc_log2, natural length (reasoning tokens at 8,192), accuracy at 8,192, closed share at 8,192,
beta > 1 share (readout --gate_stats, gated engines only). Contrasts (descriptive; the record's rule shape: per-seed
differences, all seeds the same sign, paired item bootstrap 95% interval of the seed-mean difference, 10,000 draws,
default_rng(20261009)): full - strip, full - hybrid, strip - ctrl, hybrid - warm-up, strip - warm-up, full - ctrl, on
area and natural length. G2: seed-mean natural length and area per scale, and the per-seed slope of natural length
on scale over 0..2. Writes MECH_DIR/g1g2_summary.json and prints the tables."""
import json
import math
import sys
from pathlib import Path

import numpy as np

CAPS = [256, 512, 1024, 2048, 4096, 8192]


def norm_auc(accs):
    x = [math.log2(c) for c in CAPS]
    return sum((x[j + 1] - x[j]) * (accs[j] + accs[j + 1]) / 2 for j in range(len(x) - 1)) / (x[-1] - x[0])


def load(p):
    d = json.loads(Path(p).read_text())
    rows = {r["id"]: r for r in d["rows"] if r["sample"] == 0}
    return d, rows


def per_item(rows, ids):
    auc = np.array([norm_auc([rows[i]["caps"][str(c)][0] for c in CAPS]) for i in ids])
    nat = np.array([min(rows[i]["think_len"], 8192) for i in ids], dtype=float)
    return auc, nat


def summary(d, rows, ids):
    auc, nat = per_item(rows, ids)
    gs = d.get("gate_stats") or {}
    return {"norm_auc_log2": float(auc.mean()), "natural_len": float(nat.mean()),
            "acc_8192": float(np.mean([rows[i]["caps"]["8192"][0] for i in ids])),
            "closed_share_8192": float(np.mean([1 - rows[i]["caps"]["8192"][2] for i in ids])),
            "beta_gt1_share": (gs["beta_gt1"] / gs["beta_n"]) if gs.get("beta_n") else None,
            "beta_max": gs.get("beta_max"), "parity_pass": (d.get("parity") or {}).get("pass"),
            "gpu": d["env"]["gpu"], "ckpt_sha256": d["env"]["sha256"].get("ckpt")}


def contrast(a_list, b_list, idx):
    diffs = [a - b for a, b in zip(a_list, b_list)]
    per_seed = [float(x.mean()) for x in diffs]
    m = np.mean(diffs, axis=0)
    boot = m[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    same = all(x > 0 for x in per_seed) or all(x < 0 for x in per_seed)
    return {"per_seed": per_seed, "seed_mean": float(m.mean()), "ci95": [float(lo), float(hi)],
            "rule_shape_effect": bool(same and (lo > 0 or hi < 0))}


def main():
    M = Path(sys.argv[1])
    A, D = M / "ablate", M / "dose"
    R = {}
    for f in sorted(A.glob("*.json")) + sorted(D.glob("*.json")):
        R[f.stem] = load(f)
    ids = sorted(next(iter(R.values()))[1])
    for k, (d, rows) in R.items():
        assert sorted(rows) == ids, k
    out = {"per_readout": {k: summary(d, rows, ids) for k, (d, rows) in R.items()}, "n_items": len(ids)}
    gpus = {v["gpu"] for v in out["per_readout"].values()}
    out["gpus"] = sorted(gpus)
    idx = np.random.default_rng(20261009).integers(0, len(ids), size=(10_000, len(ids)))
    item = {k: per_item(rows, ids) for k, (d, rows) in R.items()}
    con = {}

    def have(*ks):
        return all(k in item for k in ks)

    for arm in ("wide1", "wide"):
        pairs = {
            "full - strip": ("{a}_s{s}_full", "{a}_s{s}_strip"),
            "full - hybrid": ("{a}_s{s}_full", "{a}_s{s}_hybrid"),
            "strip - ctrl": ("{a}_s{s}_strip", "ctrl_s{s}"),
            "full - ctrl": ("{a}_s{s}_full", "ctrl_s{s}"),
            "hybrid - warmup": ("{a}_s{s}_hybrid", "warmup"),
            "strip - warmup": ("{a}_s{s}_strip", "warmup"),
        }
        for name, (x, y) in pairs.items():
            ka = [x.format(a=arm, s=s) for s in range(3)]
            kb = [y.format(a=arm, s=s) for s in range(3)]
            if not have(*ka, *kb):
                continue
            con[f"{arm}: {name}"] = {m: contrast([item[k][j] for k in ka], [item[k][j] for k in kb], idx)
                                     for j, m in enumerate(("norm_auc_log2", "natural_len"))}
    out["g1_contrasts_descriptive"] = con
    cond = {}
    for c in ("full", "strip", "hybrid"):
        for arm in ("wide1", "wide"):
            ks = [f"{arm}_s{s}_{c}" for s in range(3)]
            if have(*ks):
                cond[f"{arm}_{c}"] = {m: float(np.mean([out["per_readout"][k][m] for k in ks]))
                                      for m in ("norm_auc_log2", "natural_len", "acc_8192", "closed_share_8192")}
    for ref in (["warmup"], [f"ctrl_s{s}" for s in range(3)]):
        if have(*ref):
            cond["warmup" if ref == ["warmup"] else "ctrl"] = {m: float(np.mean([out["per_readout"][k][m] for k in ref]))
                                                                for m in ("norm_auc_log2", "natural_len", "acc_8192", "closed_share_8192")}
    out["g1_condition_means"] = cond
    dose = {}
    for s in range(3):
        pts = []
        for f, key in ((0.0, f"wide1_s{s}_x0"), (0.5, f"wide1_s{s}_x0.5"), (1.0, f"wide1_s{s}_full"),
                       (1.5, f"wide1_s{s}_x1.5"), (2.0, f"wide1_s{s}_x2")):
            if key in out["per_readout"]:
                pts.append((f, out["per_readout"][key]["natural_len"], out["per_readout"][key]["norm_auc_log2"],
                            out["per_readout"][key]["beta_gt1_share"]))
        if pts:
            xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
            slope = float(np.polyfit(xs, ys, 1)[0]) if len(pts) >= 2 else None
            dose[f"wide1_s{s}"] = {"points": [{"scale": p[0], "natural_len": p[1], "norm_auc_log2": p[2]} for p in pts],
                                   "natural_len_slope_per_unit_scale": slope}
    out["g2_dose"] = dose
    (M / "g1g2_summary.json").write_text(json.dumps(out, indent=1))
    print(f"n_items {len(ids)}, gpus {sorted(gpus)}")
    print(f"{'readout':<22}{'area':>8}{'natlen':>8}{'acc8k':>7}{'closed':>7}{'beta>1':>9}")
    for k, v in out["per_readout"].items():
        b = "-" if v["beta_gt1_share"] is None else f"{v['beta_gt1_share']:.4f}"
        print(f"{k:<22}{v['norm_auc_log2']:8.4f}{v['natural_len']:8.0f}{v['acc_8192']:7.3f}{v['closed_share_8192']:7.3f}{b:>9}")
    print("\ncondition means:", json.dumps({k: {m: round(x, 4) for m, x in v.items()} for k, v in cond.items()}, indent=0))
    for k, v in con.items():
        print(f"{k:<26} area {v['norm_auc_log2']['seed_mean']:+.4f} {['%.4f' % x for x in v['norm_auc_log2']['ci95']]} "
              f"len {v['natural_len']['seed_mean']:+.0f} [{v['natural_len']['ci95'][0]:.0f}, {v['natural_len']['ci95'][1]:.0f}] "
              f"seeds {[round(x) for x in v['natural_len']['per_seed']]}")
    for k, v in dose.items():
        print(k, [(p["scale"], round(p["natural_len"]), round(p["norm_auc_log2"], 4)) for p in v["points"]],
              "slope", v["natural_len_slope_per_unit_scale"])


if __name__ == "__main__":
    main()
