"""Gate parameters read directly from checkpoint tensors (CPU, no forward pass). W (one 32 x 2560 matrix per GDN
layer, zero at init) maps the layer input x to t = tanh(W x) per value head. Questions answered from the weights alone:
  P1 growth: ||W|| over steps (diffusive ~ sqrt(step) vs directed ~ step)
  P2 reproducibility: cosine of a head's W row across seeds (same arm) and across arms (wide vs wide1, same seed =
     same prompts), against the random-direction null (sd 1/sqrt(2560) = 0.020), and direction stability over steps
  P3 rank: singular value energy of W per layer vs a Gaussian matrix of the same shape
  P4 alignment: cosine of W row h with the pretrained in_proj_b row h (the head's write-strength input b = w_b . x)
     and in_proj_a row h (decay input), and with the LoRA change of in_proj_b in the same run and in the
     same-seed ctrl / ctrlpm run (does the gate push the direction the LoRA alone pushes?)"""
import json, re, sys
import os
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(os.environ.get("WG_RUNS", "runs")) / "ml7_4b/grpo"
BASE = Path(sys.argv[2] if len(sys.argv) > 2 else str(Path.home() / ".cache/gdn_gate/mech/Qwen3.5-4B"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mech")
STEPS = [25, 50, 75, 100, 125, 150]
WARM = RUNS / "ctrl_s0_cap512_fmt_box/ckpt_step40.pt"


def gates(p):
    sd = torch.load(p, map_location="cpu")
    g = {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.linear_attn\.gate_w\.weight", k)
        if m:
            g[int(m.group(1))] = v.double().numpy()
    return g, sd


def lora_b(sd):
    A, B = {}, {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.linear_attn\.in_proj_b\.lora_([AB])\.default\.weight", k)
        if m:
            (A if m.group(2) == "A" else B)[int(m.group(1))] = v.double().numpy()
    return {l: 2.0 * B[l] @ A[l] for l in A}


def rowcos(X, Y):
    nx = np.linalg.norm(X, axis=1); ny = np.linalg.norm(Y, axis=1)
    return (X * Y).sum(1) / np.maximum(nx * ny, 1e-30)


def energy(W, ks=(1, 4, 16)):
    s = np.linalg.svd(W, compute_uv=False) ** 2
    return {f"top{k}": float(s[:k].sum() / s.sum()) for k in ks} | {"eff_rank": float(s.sum() ** 2 / (s ** 2).sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G, SD = {}, {}
    for arm in ("wide1", "wide"):
        for s in range(3):
            for st in STEPS:
                G[(arm, s, st)], sd = gates(RUNS / f"{arm}_s{s}_cap512_r_box/ckpt_step{st}.pt")
                if st == 150:
                    SD[(arm, s)] = lora_b(sd)
    for arm in ("ctrl", "ctrlpm"):
        for s in range(3):
            SD[(arm, s)] = lora_b(torch.load(RUNS / f"{arm}_s{s}_cap512_r_box/ckpt_step150.pt", map_location="cpu"))
    warm_b = lora_b(torch.load(WARM, map_location="cpu"))
    layers = sorted(G[("wide1", 0, 150)])
    base = {}
    idx = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
    for l in layers:
        for nm in ("in_proj_b", "in_proj_a"):
            k = f"model.language_model.layers.{l}.linear_attn.{nm}.weight"
            with safe_open(str(BASE / idx[k]), "pt") as f:
                base[(l, nm)] = f.get_tensor(k).double().numpy()
    res = {"layers": layers, "steps": STEPS, "null_cos_sd": 1 / np.sqrt(2560)}

    # P1 growth
    gn = {f"{a}_s{s}": [float(np.sqrt(sum((G[(a, s, st)][l] ** 2).sum() for l in layers))) for st in STEPS] for a in ("wide1", "wide") for s in range(3)}
    expo = {k: float(np.polyfit(np.log(STEPS), np.log(v), 1)[0]) for k, v in gn.items()}
    res["P1_norm_by_step"] = gn; res["P1_loglog_exponent"] = expo
    # P2 reproducibility at step 150
    cs = {}
    for a in ("wide1", "wide"):
        v = [np.concatenate([rowcos(G[(a, i, 150)][l], G[(a, j, 150)][l]) for l in layers]) for i, j in ((0, 1), (0, 2), (1, 2))]
        cs[f"{a}_across_seeds"] = {"mean": float(np.mean(np.concatenate(v))), "abs_mean": float(np.mean(np.abs(np.concatenate(v)))),
                                    "share_gt_0p1": float(np.mean(np.concatenate(v) > 0.1))}
    v = np.concatenate([rowcos(G[("wide", s, 150)][l], G[("wide1", s, 150)][l]) for s in range(3) for l in layers])
    cs["wide_vs_wide1_same_seed"] = {"mean": float(v.mean()), "share_gt_0p5": float(np.mean(v > 0.5))}
    v = np.concatenate([rowcos(G[("wide", s, 150)][l], G[("wide1", t, 150)][l]) for s in range(3) for t in range(3) if s != t for l in layers])
    cs["wide_vs_wide1_diff_seed"] = {"mean": float(v.mean())}
    stab = {}
    for a in ("wide1", "wide"):
        stab[a] = [float(np.mean([np.mean(rowcos(G[(a, s, st)][l], G[(a, s, 150)][l])) for s in range(3) for l in layers])) for st in STEPS]
    cs["direction_vs_step150"] = stab
    # the drift part: W(150) - W(25) across seeds
    v = [np.concatenate([rowcos(G[(a, i, 150)][l] - G[(a, i, 25)][l], G[(a, j, 150)][l] - G[(a, j, 25)][l]) for l in layers]) for a in ("wide1", "wide") for i, j in ((0, 1), (0, 2), (1, 2))]
    cs["update_25_150_across_seeds_mean"] = float(np.mean(np.concatenate(v)))
    res["P2_cosines"] = cs
    # P3 rank
    rng = np.random.default_rng(0)
    gauss = energy(rng.standard_normal((32, 2560)))
    en = {f"{a}_s{s}": {k: float(np.mean([energy(G[(a, s, 150)][l])[k] for l in layers])) for k in ("top1", "top4", "top16", "eff_rank")} for a in ("wide1", "wide") for s in range(3)}
    res["P3_energy"] = {"gauss_32x2560": gauss, "gate": en}
    # P4 alignment
    al = {}
    for a in ("wide1", "wide"):
        for s in range(3):
            W = G[(a, s, 150)]
            cb = np.concatenate([rowcos(W[l], base[(l, "in_proj_b")]) for l in layers])
            ca = np.concatenate([rowcos(W[l], base[(l, "in_proj_a")]) for l in layers])
            own = np.concatenate([rowcos(W[l], SD[(a, s)][l] - warm_b[l]) for l in layers])
            ctl = np.concatenate([rowcos(W[l], SD[("ctrl", s)][l] - warm_b[l]) for l in layers])
            cpm = np.concatenate([rowcos(W[l], SD[("ctrlpm", s)][l] - warm_b[l]) for l in layers])
            al[f"{a}_s{s}"] = {"cos_with_base_in_proj_b": {"mean": float(cb.mean()), "abs_mean": float(np.abs(cb).mean())},
                               "cos_with_base_in_proj_a": {"mean": float(ca.mean()), "abs_mean": float(np.abs(ca).mean())},
                               "cos_with_own_lora_b_change": {"mean": float(own.mean()), "abs_mean": float(np.abs(own).mean())},
                               "cos_with_ctrl_same_seed_lora_b_change": {"mean": float(ctl.mean()), "abs_mean": float(np.abs(ctl).mean())},
                               "cos_with_ctrlpm_same_seed_lora_b_change": {"mean": float(cpm.mean()), "abs_mean": float(np.abs(cpm).mean())}}
    # LoRA in_proj_b changes vs each other (ctrl vs ctrlpm vs wide1 same seed): do they agree among themselves?
    lc = {}
    for x, y in (("ctrl", "ctrlpm"), ("ctrl", "wide1"), ("wide1", "wide")):
        v = np.concatenate([rowcos(SD[(x, s)][l] - warm_b[l], SD[(y, s)][l] - warm_b[l]) for s in range(3) for l in layers])
        lc[f"{x}_vs_{y}_same_seed"] = float(v.mean())
    res["P4_alignment"] = al; res["P4_lora_b_change_agreement"] = lc
    # magnitude: gate row norm vs LoRA-b change row norm (same input space, per head)
    mg = {}
    for a in ("wide1", "wide", "ctrl", "ctrlpm"):
        mg[f"lora_b_change_row_norm_{a}"] = float(np.mean([np.linalg.norm(SD[(a, s)][l] - warm_b[l], axis=1).mean() for s in range(3) for l in layers]))
    for a in ("wide1", "wide"):
        mg[f"gate_row_norm_{a}"] = float(np.mean([np.linalg.norm(G[(a, s, 150)][l], axis=1).mean() for s in range(3) for l in layers]))
    mg["base_in_proj_b_row_norm"] = float(np.mean([np.linalg.norm(base[(l, "in_proj_b")], axis=1).mean() for l in layers]))
    res["magnitudes"] = mg
    hn = np.concatenate([np.linalg.norm(G[("wide1", s, 150)][l], axis=1) for s in range(3) for l in layers])
    res["head_norm_dist_wide1"] = {"p50": float(np.median(hn)), "p95": float(np.quantile(hn, 0.95)), "max": float(hn.max()), "cv": float(hn.std() / hn.mean())}
    json.dump(res, open(OUT / "w_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("P1_norm_by_step",)}, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, v in gn.items():
        ax[0].plot(STEPS, v, marker="o", color="#d4572a" if k.startswith("wide1") else "#8c3fb0", lw=1, label=k)
    x = np.array(STEPS, float); ax[0].plot(x, gn["wide1_s0"][0] * np.sqrt(x / 25), "k:", lw=1, label="sqrt(step)")
    ax[0].plot(x, gn["wide1_s0"][0] * x / 25, "k--", lw=1, label="linear")
    ax[0].set_xlabel("GRPO step"); ax[0].set_title("||W|| over all 24 GDN layers"); ax[0].legend(fontsize=7, frameon=False)
    for a, c in (("wide1", "#d4572a"), ("wide", "#8c3fb0")):
        ax[1].plot(STEPS, stab[a], marker="o", color=c, label=a)
    ax[1].set_title("mean cos(W row at step, W row at 150)"); ax[1].set_xlabel("GRPO step"); ax[1].legend(frameon=False)
    labels = ["same arm,\nother seed", "wide vs wide1,\nsame seed", "wide vs wide1,\nother seed", "W vs base\nin_proj_b row", "W vs ctrl\nLoRA-b change"]
    vals = [np.mean([cs["wide1_across_seeds"]["mean"], cs["wide_across_seeds"]["mean"]]), cs["wide_vs_wide1_same_seed"]["mean"], cs["wide_vs_wide1_diff_seed"]["mean"],
            np.mean([al[k]["cos_with_base_in_proj_b"]["mean"] for k in al]), np.mean([al[k]["cos_with_ctrl_same_seed_lora_b_change"]["mean"] for k in al])]
    ax[2].axhspan(-2 / np.sqrt(2560), 2 / np.sqrt(2560), color="#dddddd", label="random +-2 sd", zorder=0); ax[2].bar(labels, vals, color="#888888", zorder=2); ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title("mean cosine of per-head W rows (step 150)"); ax[2].tick_params(axis="x", labelsize=7); ax[2].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "w_stats.png", dpi=130)


if __name__ == "__main__":
    main()
