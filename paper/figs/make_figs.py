"""Draw every paper figure from the JSON receipts (CPU only, matplotlib, PDF output).

Run: python3 paper/figs/derive.py && python3 paper/figs/make_figs.py
Inputs: results/** analysis and receipt JSON, paper/data/derived.json.
Outputs: paper/figs/*.pdf
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "paper/figs"
DER = json.load(open(ROOT / "paper/data/derived.json"))

# Two families, two lightness levels each: blue = LoRA arms, orange = gated arms; dark/light within a family.
# Grayscale-safe by marker shape AND fill: LoRA filled circle (dark), LoRA-r20 open diamond (light),
# Gate(0,1) filled triangle (light), Gate(0,2) open square (dark).
COL = {"ctrl": "#1b4f9c", "ctrlpm": "#6fa8dc", "wide1": "#f28e2b", "wide": "#9c3d10"}
MRK = {"ctrl": "o", "ctrlpm": "D", "wide1": "^", "wide": "s"}
MFC = {"ctrl": COL["ctrl"], "ctrlpm": "white", "wide1": COL["wide1"], "wide": "white"}
ORDER = ["ctrl", "ctrlpm", "wide1", "wide"]
# Paper names: ctrl = LoRA, ctrlpm = LoRA-r20, wide1 = Gate(0,1), wide = Gate(0,2) (receipts keep the run names).
LBL = {"ctrl": "LoRA", "wide": "Gate$_{(0,2)}$", "wide1": "Gate$_{(0,1)}$",
       "ctrlpm": "LoRA-r20", "baseline": "base model", "warmup": "warm-up"}
PN = {"ctrl": "LoRA", "wide": "Gate$_{(0,2)}$", "wide1": "Gate$_{(0,1)}$", "ctrlpm": "LoRA-r20"}


def pname(contrast):
    a, b = [x.strip() for x in contrast.split("-")]
    return f"{PN[a]} $-$ {PN[b]}"


def scat(ax, x, y, arm, s=14, z=3):
    ax.scatter(x, y, s=s, marker=MRK[arm], facecolor=MFC[arm], edgecolor=COL[arm], lw=0.9, zorder=z)
GRAY = {"baseline": "#8a8985", "warmup": "#52514e"}
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def load(rel):
    return json.load(open(RES / rel))


A2 = {"gsm8k": load("ml7/step14/analysis_gsm8k.json"), "math500": load("ml7/step14/analysis_math500.json")}
A2X = {"gsm8k": load("ml7/step14_ext/analysis_gsm8k.json"), "math500": load("ml7/step14_ext/analysis_math500.json")}
A4 = {"gsm8k": load("ml7_4b/step14_ctrlpm/analysis_gsm8k.json"), "math500": load("ml7_4b/step14_ctrlpm/analysis_math500.json")}
SETNAME = {"math500": "MATH-500 (transfer)", "gsm8k": "GSM8K test (in domain)"}


def budget_panel(ax, d, arms, title):
    caps = d["caps"]
    x = np.log2(caps)
    for name, ls in (("baseline", "--"), ("warmup", ":")):
        ax.plot(x, [d[name]["acc"][str(c)] for c in caps], ls, color=GRAY[name], lw=1.2, label=LBL[name])
    for arm in arms:
        a = d["arms"][arm]
        per = np.array([a["acc"][str(c)]["per_seed"] for c in caps])  # caps x seeds
        for s in range(per.shape[1]):
            ax.plot(x, per[:, s], color=COL[arm], lw=0.5, alpha=0.45)
        ax.plot(x, per.mean(1), color=COL[arm], marker=MRK[arm], ms=4.5, mfc=MFC[arm], mec=COL[arm], mew=1.0,
                label=LBL[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c // 1024}k" if c >= 1024 else str(c) for c in caps])
    ax.set_xlabel("thinking budget (tokens, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)


def fig_budget():
    for name, A, arms, tag in (("fig_budget.pdf", A4, ["ctrl", "ctrlpm", "wide1", "wide"], "Qwen3.5-4B"),
                               ("fig_budget2b.pdf", A2, ["ctrl", "wide1", "wide"], "Qwen3.5-2B (seeds 0\u20132)")):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
        budget_panel(axes[0], A["math500"], arms, f"{tag}, " + SETNAME["math500"])
        budget_panel(axes[1], A["gsm8k"], arms, f"{tag}, " + SETNAME["gsm8k"])
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="center right", ncol=1, frameon=False, bbox_to_anchor=(1.0, 0.55))
        fig.tight_layout(rect=(0, 0, 0.83, 1))
        fig.savefig(FIG / name)
        plt.close(fig)


def fig_length():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, ds in zip(axes, ("gsm8k", "math500")):
        groups = [("2B s0-2", A2[ds], ["ctrl", "wide1", "wide"]), ("2B s3-5", A2X[ds], ["ctrl", "wide1", "wide"]),
                  ("4B s0-2", A4[ds], ["ctrl", "ctrlpm", "wide1", "wide"])]
        xpos, ticks, tlabels = 0, [], []
        for gname, d, arms in groups:
            start = xpos
            for arm in arms:
                v = d["arms"][arm]["natural_len"]["per_seed"]
                ax.bar(xpos, np.mean(v), width=0.7, color=COL[arm], alpha=0.35, edgecolor="none")
                scat(ax, [xpos] * len(v), v, arm)
                xpos += 1
            ax.axhline(d["baseline"]["natural_len"], xmin=0, xmax=0, color="none")
            ticks.append((start + xpos - 1) / 2)
            tlabels.append(gname)
            ax.plot([start - 0.4, xpos - 0.6], [d["baseline"]["natural_len"]] * 2, "--", color=GRAY["baseline"], lw=1)
            xpos += 0.8
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("reasoning length\n(mean tokens at the 8,192 budget)")
        ax.set_title(SETNAME[ds])
    handles = [plt.Line2D([], [], color=COL[a], marker=MRK[a], mfc=MFC[a], mec=COL[a], ls="", label=LBL[a]) for a in ORDER]
    handles.append(plt.Line2D([], [], color=GRAY["baseline"], ls="--", label="base model"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig_length.pdf")
    plt.close(fig)


TAGS = {("2B s3\u20135", "effects", "wide - ctrl"): "H1", ("2B s3\u20135", "length_effects", "wide - ctrl"): "H2",
        ("4B s0\u20132", "effects", "wide - ctrl"): "H1", ("4B s0\u20132", "length_effects", "wide - ctrl"): "H2",
        ("4B s0\u20132", "effects", "wide1 - ctrlpm"): "H3", ("4B s0\u20132", "length_effects", "wide1 - ctrlpm"): "H4"}


def contrast_rows(block):
    """(label, per_seed, mean, ci or None, effect, first arm, tag) for the panel."""
    rows = []
    specs = [("2B s0\u20132", A2, ["wide1 - ctrl", "wide - ctrl", "wide - wide1"]),
             ("2B s3\u20135", A2X, ["wide1 - ctrl", "wide - ctrl", "wide - wide1"]),
             ("4B s0\u20132", A4, ["wide1 - ctrl", "wide - ctrl", "wide - wide1", "wide1 - ctrlpm", "wide - ctrlpm",
                                "ctrlpm - ctrl"])]
    ds = "math500" if block == "effects" else "gsm8k"
    for g, A, cons in specs:
        d = A[ds]
        for c in cons:
            first = c.split("-")[0].strip()
            tag = TAGS.get((g, block, c), "")
            if block in d and c in d[block]:
                e = d[block][c]
                rows.append((f"{g}: {pname(c)}", e["per_seed_diff"], e["seed_mean_diff"], e["ci95"], e["effect"], first, tag))
            else:  # 2B seeds 0-2 analysis has no length block: per-seed differences from length, no interval
                a, b = [x.strip() for x in c.split("-")]
                pa, pb = d["arms"][a]["natural_len"]["per_seed"], d["arms"][b]["natural_len"]["per_seed"]
                diff = [x - y for x, y in zip(pa, pb)]
                rows.append((f"{g}: {pname(c)}", diff, float(np.mean(diff)), None, None, first, tag))
    return rows


def fig_contrasts():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), sharey=True)
    for ax, block, xl, title in (
            (axes[0], "effects", "difference in area (accuracy vs log$_2$ budget)", "MATH-500 area"),
            (axes[1], "length_effects", "difference in length (tokens at the 8,192 budget)", "GSM8K length")):
        rows = contrast_rows(block)
        y = np.arange(len(rows))[::-1]
        for yi, (lab, per, m, ci, eff, first, tag) in zip(y, rows):
            col = COL[first]
            if ci is not None:
                ax.plot(ci, [yi, yi], color=col, lw=2.4, alpha=0.5, solid_capstyle="round")
            scat(ax, per, [yi] * len(per), first, s=13)
            ax.scatter([m], [yi], s=60, color=INK, marker="|", zorder=4, lw=1.4)
            if eff:
                ax.annotate("*", (max(ci[1], max(per)), yi), xytext=(3, -4), textcoords="offset points", fontsize=9, color=INK)
            if tag:
                ax.annotate(f"[{tag}]", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(2, 0),
                            textcoords="offset points", fontsize=6.5, color=INK, va="center")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel(xl)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_contrasts.pdf")
    plt.close(fig)


def fig_olmo():
    cp = load("olmo/vllm2/counting_paired.json")
    loc = load("olmo/vllm2/counting_localization.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for task, col, mk in (("ruler_cwe", "#1b4f9c", "o"), ("ruler_fwe", "#9c3d10", "s")):
        rows = sorted([r for r in cp["rows"] if r["task"] == task], key=lambda r: r["L"])
        L = [r["L"] for r in rows]
        x = np.log2(L)
        name = "CWE" if task == "ruler_cwe" else "FWE"
        ax.plot(x, [r["stock"] for r in rows], color=col, marker=mk, ms=4, label=f"{name}, stock")
        ax.plot(x, [r["arm_mean"] for r in rows], color=col, marker=mk, ms=4, ls="--", mfc="white", label=f"{name}, $\\beta$ clamped at 1")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l // 1024}k" for l in L])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("accuracy (500 items per cell)")
    ax.set_title("Olmo-Hybrid-7B: counting tasks")
    ax.legend(frameon=False, ncol=1, loc="center right")
    ax = axes[1]
    labels = ["Q1\n(GDN 0-5)", "Q2\n(6-11)", "Q3\n(12-17)", "Q4\n(18-23)"]
    rows = sorted(loc["macro"], key=lambda r: r["arm"])
    for i, r in enumerate(rows):
        ax.plot([i, i], [100 * v for v in r["ci95"]], color="#1b4f9c", lw=2.2, alpha=0.55)
        ax.scatter([i], [100 * r["diff"]], color="#1b4f9c", s=18, zorder=3)
    full = [m for m in cp["macro"] if m["L"] == 8192][0]
    cwe8 = [r for r in cp["rows"] if r["L"] == 8192 and r["task"] == "ruler_cwe"][0]
    ax.axhline(100 * cwe8["diff"], color=INK2, ls=":", lw=1)
    ax.text(3.4, 100 * cwe8["diff"], "all layers", va="bottom", ha="right", fontsize=6.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("CWE 8k accuracy change (points)")
    ax.set_title("Clamp one depth quarter")
    ax.grid(axis="x", visible=False)
    _ = full
    fig.tight_layout()
    fig.savefig(FIG / "fig_olmo.pdf")
    plt.close(fig)


def fig_sft_forget():
    s6 = DER["step6_sft"]["rows"]
    fg = DER["forgetting_humaneval"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    arms = ["ctrl", "wide1", "wide"]
    for s in range(6):
        v = [s6[f"{a}_s{s}"]["val_loss"] for a in arms]
        ax.plot(range(3), v, color="#c3c2b7", lw=0.8, zorder=1)
        for i, a in enumerate(arms):
            scat(ax, [i], [v[i]], a, s=16)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LBL[a] for a in arms])
    ax.set_ylabel("held-out loss (nats/token, lower is better)")
    ax.text(0.98, 0.97, "base model (step 0): 1.322, off scale", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=INK2)
    ax.set_title("Qwen3.5-4B-Base, chat SFT, 6 seeds")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    groups = [("4B", [("ctrl", "4B_main_ctrl"), ("wide1", "4B_main_wide1"), ("wide", "4B_main_wide")]),
              ("9B", [("ctrl", "9B_main_ctrl"), ("wide", "9B_main_wide")])]
    xpos, ticks, tl = 0, [], []
    for g, items in groups:
        start = xpos
        for arm, key in items:
            v = [fg[key][f"s{s}"]["pass1"] for s in range(3)]
            ax.bar(xpos, np.mean(v), width=0.7, color=COL[arm], alpha=0.35)
            scat(ax, [xpos] * 3, v, arm, s=16)
            ticks.append(xpos)
            tl.append(LBL[arm])
            xpos += 1
        if g == "4B":
            b = fg["4B_base_untouched"]["pass1"]
            ax.plot([start - 0.4, xpos - 0.6], [b, b], "--", color=GRAY["baseline"], lw=1)
            ax.text(xpos - 0.6, b, "4B base", fontsize=6, ha="right", va="bottom", color=INK2)
        ax.text((start + xpos - 1) / 2, -0.2, f"Qwen3.5-{g}-Base", ha="center", fontsize=7, transform=ax.get_xaxis_transform())
        xpos += 0.8
    ax.set_xticks(ticks)
    ax.set_xticklabels(tl)
    ax.set_ylabel("HumanEval pass@1 (higher is better)")
    ax.set_title("After a damaging LoRA fine-tune")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sft_forget.pdf")
    plt.close(fig)


F4 = load("fresh/analysis_fresh_4b.json")
F2 = load("fresh/analysis_fresh_2b.json")
SRCNAME = {"aime2024": "AIME 2024", "aime2025": "AIME 2025", "aime2026": "AIME 2026", "amc2023": "AMC 2023"}


def budget_panel_fresh(ax, d, arms, title):
    caps = d["caps"]
    x = np.log2(caps)
    for name, ls in (("baseline", "--"), ("warmup", ":")):
        if d.get(name):
            ax.plot(x, [d[name]["acc"][str(c)] for c in caps], ls, color=GRAY[name], lw=1.2, label=LBL[name])
    for arm in arms:
        a = d["arms"][arm]
        per = np.array([a["acc"][str(c)]["per_seed"] for c in caps])
        for s in range(per.shape[1]):
            ax.plot(x, per[:, s], color=COL[arm], lw=0.5, alpha=0.45)
        ax.plot(x, per.mean(1), color=COL[arm], marker=MRK[arm], ms=4.5, mfc=MFC[arm], mec=COL[arm], mew=1.0,
                label=LBL[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c // 1024}k" if c >= 1024 else str(c) for c in caps])
    ax.set_xlabel("thinking budget (tokens, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)


def fig_fresh():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    budget_panel_fresh(axes[0], F4, ["ctrl", "ctrlpm", "wide1", "wide"], "Qwen3.5-4B (3 seeds)")
    ax = axes[1]
    src = F4["per_source_descriptive"]
    arms = ["ctrl", "ctrlpm", "wide1", "wide"]
    w = 0.2
    for j, k in enumerate(src):
        base = src[k]["baseline_norm_auc_log2"]
        ax.plot([j - 0.45, j + 0.45], [base, base], "--", color=GRAY["baseline"], lw=1)
        for i, arm in enumerate(arms):
            v = src[k]["arms"][arm]["norm_auc_log2"]
            xpos = j + (i - 1.5) * w
            ax.bar(xpos, v["seed_mean"], width=w * 0.9, color=COL[arm], alpha=0.35, edgecolor="none")
            scat(ax, [xpos] * len(v["per_seed"]), v["per_seed"], arm, s=7)
    ax.set_xticks(range(len(src)))
    ax.set_xticklabels([SRCNAME[k].replace(" ", "\n") + f"\n({src[k]['n_items']})" for k in src])
    ax.set_ylabel("area (accuracy vs log$_2$ budget)")
    ax.set_title("Qwen3.5-4B by source (descriptive)")
    ax.grid(axis="x", visible=False)
    budget_panel_fresh(axes[2], F2, ["ctrl", "wide1", "wide"], "Qwen3.5-2B (6 seeds)")
    for a in (axes[0], axes[2]):
        a.set_ylim(0, 0.68)
    handles = [plt.Line2D([], [], color=COL[a], marker=MRK[a], mfc=MFC[a], mec=COL[a], label=LBL[a]) for a in arms]
    handles += [plt.Line2D([], [], color=GRAY["baseline"], ls="--", label="base model"),
                plt.Line2D([], [], color=GRAY["warmup"], ls=":", label="warm-up (4B only)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_fresh.pdf")
    plt.close(fig)


def fig_mech():
    g = load("mech/g1g2_summary.json")
    pr = g["per_readout"]
    G3 = DER["mech"]["g3"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    xpos, ticks, tl = 0, [], []
    ax.bar(xpos, pr["warmup"]["natural_len"], width=0.7, color=GRAY["warmup"], alpha=0.35)
    ax.scatter([xpos], [pr["warmup"]["natural_len"]], color=GRAY["warmup"], s=12, zorder=3)
    ticks.append(xpos); tl.append("warm-up")
    xpos += 1
    v = [pr[f"ctrl_s{s}"]["natural_len"] for s in range(3)]
    ax.bar(xpos, np.mean(v), width=0.7, color=COL["ctrl"], alpha=0.35)
    scat(ax, [xpos] * 3, v, "ctrl", s=12)
    ticks.append(xpos); tl.append("LoRA")
    xpos += 1.6
    for arm in ("wide1", "wide"):
        start = xpos
        for cond, lab in (("hybrid", "hybrid"), ("strip", "strip"), ("full", "full")):
            v = [pr[f"{arm}_s{s}_{cond}"]["natural_len"] for s in range(3)]
            ax.bar(xpos, np.mean(v), width=0.7, color=COL[arm], alpha={"hybrid": 0.2, "strip": 0.3, "full": 0.55}[cond])
            scat(ax, [xpos] * 3, v, arm, s=12)
            ticks.append(xpos); tl.append(lab)
            xpos += 1
        ax.text((start + xpos - 1) / 2, 2650, LBL[arm], ha="center", fontsize=7, color=COL[arm])
        xpos += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(tl, fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("reasoning length (tokens)")
    ax.set_title("(a) Remove one trained part")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    dose = g["g2_dose"]
    allL = []
    for s in range(3):
        pts = sorted(dose[f"wide1_s{s}"]["points"], key=lambda r: r["scale"])
        sc = [r["scale"] for r in pts]
        L = [r["natural_len"] for r in pts]
        allL.append(L)
        ax.plot(sc, L, color=COL["wide1"], lw=0.6, alpha=0.6, marker=MRK["wide1"], ms=3)
    ax.plot(sc, np.mean(allL, 0), color=COL["wide1"], lw=1.6, marker=MRK["wide1"], ms=4.5, mec=COL["wide1"])
    ax.axhline(np.mean([pr[f"ctrl_s{s}"]["natural_len"] for s in range(3)]), color=COL["ctrl"], ls="--", lw=1)
    ax.text(2.0, np.mean([pr[f"ctrl_s{s}"]["natural_len"] for s in range(3)]), "LoRA", ha="right", va="bottom",
            fontsize=6.5, color=COL["ctrl"])
    ax.set_xticks(sc)
    ax.set_xlabel("gate scale (1 = trained $W$)")
    ax.set_ylabel("reasoning length (tokens)")
    ax.set_title("(b) Scale the trained gate")
    ax = axes[2]
    layers = np.arange(1, 25)
    ax.plot(layers, G3["wide1"]["gate_abs_dbeta_by_layer_seedmean"], color=COL["wide1"], marker=MRK["wide1"], ms=2.5,
            label="gate, " + LBL["wide1"])
    ax.plot(layers, G3["wide"]["gate_abs_dbeta_by_layer_seedmean"], color=COL["wide"], marker=MRK["wide"], ms=2.5,
            mfc="white", label="gate, " + LBL["wide"])
    ax.plot(layers, G3["ctrl_lora_abs_dbeta_by_layer_seedmean"], color=COL["ctrl"], marker=MRK["ctrl"], ms=2.5,
            label="LoRA, LoRA arm")
    ax.set_yscale("log")
    ax.set_xlabel("GDN layer (1 = first)")
    ax.set_ylabel("mean $|\\Delta\\beta|$ at step 150")
    ax.set_title("(c) Change in $\\beta$ by depth")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_mech.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_budget()
    fig_length()
    fig_contrasts()
    fig_olmo()
    fig_sft_forget()
    fig_fresh()
    fig_mech()
    print("figures written to", FIG)
