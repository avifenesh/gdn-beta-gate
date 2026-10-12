"""Parameter-level drift of the 4B GRPO arms from the shared warm-up (CPU, checkpoints only): per GDN layer, the
Frobenius norm of the gate W (wide, wide1) and of the change in each LoRA product scale*B@A relative to the warm-up
(in_proj_qkv, in_proj_a = decay input, in_proj_b = write-strength input, out_proj), at steps 25..150."""
import json, re, sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(os.environ.get("WG_RUNS", "runs")) / "ml7_4b/grpo"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mech")
WARM = RUNS / "ctrl_s0_cap512_fmt_box" / "ckpt_step40.pt"
ARMS = ["ctrl", "ctrlpm", "wide1", "wide"]
STEPS = [25, 50, 75, 100, 125, 150]
PROJ = ["in_proj_qkv", "in_proj_a", "in_proj_b", "out_proj"]
COL = {"ctrl": "#7f7f7f", "ctrlpm": "#2a7bbd", "wide1": "#d4572a", "wide": "#8c3fb0"}
PAT = re.compile(r"layers\.(\d+)\.linear_attn\.(\w+)\.lora_([AB])\.default\.weight")


def products(state):
    ab = {}
    for k, v in state.items():
        m = PAT.search(k)
        if m:
            ab.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v.float()
    return {k: 2.0 * d["B"] @ d["A"] for k, d in ab.items()}


def gates(state):
    g = {}
    for k, v in state.items():
        m = re.search(r"layers\.(\d+)\.linear_attn\.gate_w\.weight", k)
        if m:
            g[int(m.group(1))] = v.float()
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == "--plot":
        return plot(json.load(open(OUT / "param_drift.json")))
    w0 = products(torch.load(WARM, map_location="cpu"))
    layers = sorted({l for l, _ in w0})
    res = {"warmup": str(WARM), "layers": layers, "steps": STEPS, "runs": {}}
    res["warmup_norm"] = {p: [float(w0[(l, p)].norm()) for l in layers] for p in PROJ}
    for arm in ARMS:
        for s in range(3):
            run = f"{arm}_s{s}"
            r = {"lora_delta": {p: {} for p in PROJ}, "gate_norm": {}, "gate_row_norm_mean": {}}
            for st in STEPS:
                sd = torch.load(RUNS / f"{arm}_s{s}_cap512_r_box" / f"ckpt_step{st}.pt", map_location="cpu")
                P = products(sd)
                for p in PROJ:
                    r["lora_delta"][p][st] = [float((P[(l, p)] - w0[(l, p)]).norm()) for l in layers]
                G = gates(sd)
                if G:
                    r["gate_norm"][st] = [float(G[l].norm()) for l in layers]
                    r["gate_row_norm_mean"][st] = [float(G[l].norm(dim=1).mean()) for l in layers]
            res["runs"][run] = r
            tot = {p: sum(r["lora_delta"][p][150]) for p in PROJ}
            print(run, {p: round(v, 3) for p, v in tot.items()},
                  "gate", round(float(np.sqrt(sum(x * x for x in r["gate_norm"][150]))), 4) if r["gate_norm"] else "-")
    json.dump(res, open(OUT / "param_drift.json", "w"))
    plot(res)


def plot(res):
    layers = res["layers"]
    res["runs"] = {k: {kk: ({p: {int(a): b for a, b in d.items()} for p, d in vv.items()} if kk == "lora_delta" else {int(a): b for a, b in vv.items()}) for kk, vv in v.items()} for k, v in res["runs"].items()}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for run, r in res["runs"].items():
        arm = run.split("_s")[0]
        lab = arm if run.endswith("s0") else None
        ax[0].plot(layers, r["lora_delta"]["in_proj_b"][150], color=COL[arm], lw=1.2, label=lab, marker=".")
        ax[1].plot(STEPS, [sum(sum(r["lora_delta"][p][st]) for p in PROJ) for st in STEPS], color=COL[arm], lw=1.2, label=lab)
        if r["gate_norm"]:
            ax[2].plot(layers, r["gate_norm"][150], color=COL[arm], lw=1.2, label=lab, marker=".")
    ax[0].set_title("change in LoRA(in_proj_b) at step 150, per GDN layer"); ax[0].set_xlabel("layer index")
    ax[1].set_title("sum over layers and projections of LoRA change"); ax[1].set_xlabel("GRPO step")
    ax[2].set_title("gate W Frobenius norm at step 150"); ax[2].set_xlabel("layer index")
    for a in ax: a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "param_drift.png", dpi=130)


if __name__ == "__main__":
    main()
