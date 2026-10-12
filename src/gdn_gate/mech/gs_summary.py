"""Summarize gate_stats.py JSONs: what the learned gate does to the delta-rule write strength (beta) per layer and
over training, against what the LoRA alone does to it (ctrl, ctrlpm) relative to the warm-up."""
import glob, json, re, sys
import os
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SET = sys.argv[3] if len(sys.argv) > 3 else "fixed"
TOK = Tokenizer.from_file(glob.glob(os.path.join(os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub")), "models--Qwen--Qwen3.5-4B/snapshots/*/tokenizer.json"))[0])
COL = {"ctrl": "#7f7f7f", "ctrlpm": "#2a7bbd", "wide1": "#d4572a", "wide": "#8c3fb0"}
LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]


def load():
    J = {}
    for f in glob.glob(str(IN / f"*__{SET}.json")):
        d = json.load(open(f))
        J[d["name"]] = d
    return J


def A(d, k):
    return np.array(d["mean"][k])  # [L, H]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    J = load()
    w = J["warmup"]
    s0 = A(w, "s")
    summ = {"set": SET, "n_tokens": w["n_tokens"], "n_seq": w["n_seq"], "warmup": {
        "s_mean": float(s0.mean()), "s_lt_0p1_share": float(A(w, "s_lo").mean()),
        "s_mean_by_layer": s0.mean(1).tolist(), "decay_mean": float(A(w, "dec").mean())}, "runs": {}}
    for name, d in sorted(J.items()):
        if name == "warmup":
            continue
        m = re.match(r"(ctrlpm|ctrl|wide1|wide)_s(\d)_(\d+)", name)
        arm, seed, step = m.group(1), int(m.group(2)), int(m.group(3))
        dd, s, bt, tabs = A(d, "d"), A(d, "s"), A(d, "beta"), A(d, "tabs")
        dlora = s - s0                      # write-strength change carried by the LoRA (through in_proj_b)
        dtot = bt - s0                      # total write-strength change vs the warm-up
        r = {"arm": arm, "seed": seed, "step": step,
             "gate_d_mean": float(dd.mean()), "gate_d_abs_mean": float(np.abs(dd).mean()),
             "gate_t_abs_mean": float(tabs.mean()), "gate_t_pos_share": float(A(d, "tpos").mean()),
             "beta_gt1_share": float(A(d, "bgt1").mean()),
             "lora_ds_mean": float(dlora.mean()), "lora_ds_abs_mean": float(np.abs(dlora).mean()),
             "total_dbeta_mean": float(dtot.mean()), "total_dbeta_abs_mean": float(np.abs(dtot).mean()),
             "decay_change_mean": float((A(d, "dec") - A(w, "dec")).mean()),
             "gate_d_by_layer": dd.mean(1).tolist(), "gate_d_abs_by_layer": np.abs(dd).mean(1).tolist(),
             "lora_ds_abs_by_layer": np.abs(dlora).mean(1).tolist(),
             "beta_gt1_by_layer": A(d, "bgt1").mean(1).tolist(),
             "heads_gate_abs_d_gt_0p05": int((np.abs(dd) > 0.05).sum()), "heads_lora_abs_ds_gt_0p05": int((np.abs(dlora) > 0.05).sum()),
             "gate_d_max_head": float(dd.max()), "gate_d_min_head": float(dd.min()),
             "d_by_sbin": np.average(np.array(d["d_by_sbin"]), axis=0, weights=np.array(d["n_by_sbin"]).sum(0) * 0 + np.array(d["n_by_sbin"])).tolist()
                          if False else (np.array(d["d_by_sbin"]) * np.array(d["n_by_sbin"])).sum(0).__truediv__(np.maximum(1, np.array(d["n_by_sbin"]).sum(0))).tolist(),
             "n_by_sbin": np.array(d["n_by_sbin"]).sum(0).tolist(),
             "d_by_relpos": np.array(d["d_by_relpos"]).mean(0).tolist()}
        if arm in ("wide1", "wide") and step == 150:
            li = int(np.argmax(np.abs(dd).mean(1)))
            tp = d["token_delta_top"][str(li)]
            r["top_tokens_layer"] = LAYERS[li]
            r["top_tokens_up"] = [(TOK.decode([i]), round(v, 4), n) for i, v, n in tp["up"][:10]]
            r["top_tokens_down"] = [(TOK.decode([i]), round(v, 4), n) for i, v, n in tp["down"][:10]]
        summ["runs"][name] = r
    json.dump(summ, open(OUT / f"gate_stats_summary_{SET}.json", "w"), indent=1)
    for name, r in sorted(summ["runs"].items()):
        print(f"{name:14s} gate: d {r['gate_d_mean']:+.4f} |d| {r['gate_d_abs_mean']:.4f} |t| {r['gate_t_abs_mean']:.4f} t>0 {r['gate_t_pos_share']:.2f} "
              f"beta>1 {r['beta_gt1_share']:.4f} heads|d|>.05 {r['heads_gate_abs_d_gt_0p05']:3d} | LoRA: ds {r['lora_ds_mean']:+.4f} |ds| {r['lora_ds_abs_mean']:.4f} "
              f"heads {r['heads_lora_abs_ds_gt_0p05']:3d} | total |dbeta| {r['total_dbeta_abs_mean']:.4f} decay chg {r['decay_change_mean']:+.4f}")
        if "top_tokens_up" in r:
            print("   layer", r["top_tokens_layer"], "up", r["top_tokens_up"][:6], "down", r["top_tokens_down"][:6])

    R = summ["runs"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for name, r in R.items():
        if r["step"] != 150:
            continue
        c = COL[r["arm"]]; lab = r["arm"] if r["seed"] == 0 else None
        if r["arm"] in ("wide1", "wide"):
            ax[0].plot(LAYERS, r["gate_d_by_layer"], color=c, marker=".", lw=1, label=lab)
            ax[1].plot(LAYERS, r["gate_d_abs_by_layer"], color=c, marker=".", lw=1, label=lab + " gate |beta - s|" if lab else None)
        ax[1].plot(LAYERS, r["lora_ds_abs_by_layer"], color=c, ls="--", lw=1, label=(lab + " LoRA |s - s_warm|") if lab else None)
        if r["arm"] == "wide":
            ax[2].plot(LAYERS, r["beta_gt1_by_layer"], color=c, marker=".", lw=1, label=lab)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_title("gate effect beta - s, mean over heads (step 150)"); ax[0].set_xlabel("layer")
    ax[1].set_title("mean |change in beta| per head: gate vs LoRA"); ax[1].set_xlabel("layer")
    ax[2].set_title("wide: share of tokens x heads with beta > 1"); ax[2].set_xlabel("layer")
    for a in ax: a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"gate_stats_layers_{SET}.png", dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for arm in ("wide1", "wide"):
        for sd in range(3):
            pts = sorted((r["step"], r) for r in R.values() if r["arm"] == arm and r["seed"] == sd)
            if not pts:
                continue
            st = [p[0] for p in pts]
            ax[0].plot(st, [p[1]["gate_t_abs_mean"] for p in pts], color=COL[arm], marker="o", lw=1, label=arm if sd == 0 else None)
            ax[1].plot(st, [p[1]["gate_d_mean"] for p in pts], color=COL[arm], marker="o", lw=1, label=arm if sd == 0 else None)
    ax[0].set_title("mean |t| = |tanh(Wx)| over reasoning tokens"); ax[1].set_title("mean beta - s"); ax[0].set_xlabel("GRPO step"); ax[1].set_xlabel("GRPO step")
    ax[1].axhline(0, color="k", lw=0.5)
    for name, r in R.items():
        if r["step"] == 150 and r["arm"] in ("wide1", "wide"):
            x = (np.arange(20) + 0.5) / 20
            ax[2].plot(x, r["d_by_sbin"], color=COL[r["arm"]], lw=1, label=r["arm"] if r["seed"] == 0 else None)
    ax[2].axhline(0, color="k", lw=0.5); ax[2].set_title("gate effect beta - s by pretrained-path s"); ax[2].set_xlabel("s = sigmoid(b)")
    for a in ax: a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"gate_stats_steps_{SET}.png", dpi=130)


if __name__ == "__main__":
    main()
