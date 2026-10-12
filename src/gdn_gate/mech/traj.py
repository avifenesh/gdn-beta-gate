"""Training trajectories of the 4B cap-512 GRPO arms (ctrl, ctrlpm, wide1, wide x seeds 0-2) from train_log.jsonl and
rollouts.jsonl: forced share, reasoning length, accuracy, reward, and the length-signal supply (groups where the length
term of the reward can differ between correct samples). CPU only, reads saved logs."""
import json, statistics as st, sys
import os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(os.environ.get("WG_RUNS", "runs")) / "ml7_4b/grpo"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mech")
ARMS = ["ctrl", "ctrlpm", "wide1", "wide"]
COL = {"ctrl": "#7f7f7f", "ctrlpm": "#2a7bbd", "wide1": "#d4572a", "wide": "#8c3fb0"}


def roll(x, k=10):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - k + 1): i + 1].mean() for i in range(len(x))])


def per_step_rollouts(path):
    by = {}
    for ln in open(path):
        r = json.loads(ln)
        by.setdefault(r["step"], []).append(r)
    out = []
    for s in sorted(by):
        rs = by[s]
        groups = {}
        for r in rs:
            groups.setdefault(r["row"], []).append(r)
        nat = [not r["forced"] for r in rs]
        cn = [r["correct"] and not r["forced"] for r in rs]
        # a group carries length signal when its correct samples differ in reasoning length
        lg = 0
        for g in groups.values():
            cl = [r["reasoning"] for r in g if r["correct"]]
            if len(cl) >= 2 and max(cl) > min(cl):
                lg += 1
        nat_len = [r["reasoning"] for r in rs if not r["forced"]]
        out.append({"step": s, "n": len(rs), "nat_frac": float(np.mean(nat)), "correct_nat_frac": float(np.mean(cn)),
                    "acc_nat": float(np.mean([r["correct"] for r in rs if not r["forced"]])) if nat_len else None,
                    "acc_forced": float(np.mean([r["correct"] for r in rs if r["forced"]])) if len(nat_len) < len(rs) else None,
                    "len_groups": lg, "groups": len(groups),
                    "nat_len_mean": float(np.mean(nat_len)) if nat_len else None})
    return out


def onset(x, thr):
    r = roll(x)
    idx = np.where(r < thr)[0]
    return int(idx[0]) if len(idx) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for arm in ARMS:
        for s in range(3):
            run = RUNS / f"{arm}_s{s}_cap512_r_box"
            L = [json.loads(l) for l in open(run / "train_log.jsonl")]
            R = per_step_rollouts(run / "rollouts.jsonl")
            ff = [x["forced_frac"] for x in L]
            data[f"{arm}_s{s}"] = {
                "arm": arm, "seed": s,
                "forced_frac": ff, "reasoning_mean": [x["reasoning_mean"] for x in L],
                "acc": [x["acc"] for x in L], "reward_mean": [x["reward_mean"] for x in L],
                "w_norm": [x.get("w_norm") for x in L], "grad_norm": [x["grad_norm"] for x in L],
                "nonzero_adv": [x["nonzero_adv"] for x in L],
                "len_groups": [x["len_groups"] for x in R], "correct_nat_frac": [x["correct_nat_frac"] for x in R],
                "acc_nat": [x["acc_nat"] for x in R], "acc_forced": [x["acc_forced"] for x in R],
                "nat_len_mean": [x["nat_len_mean"] for x in R],
                "onset_forced_lt_0p9": onset(ff, 0.9), "onset_forced_lt_0p5": onset(ff, 0.5),
                "len_groups_first25": int(sum(x["len_groups"] for x in R[:25])),
                "len_groups_first50": int(sum(x["len_groups"] for x in R[:50])),
                "forced_last20": float(np.mean(ff[-20:])), "reasoning_last20": float(np.mean([x["reasoning_mean"] for x in L][-20:])),
                "acc_first20": float(np.mean([x["acc"] for x in L][:20])), "acc_last20": float(np.mean([x["acc"] for x in L][-20:])),
            }
    # paired check: same seed -> same prompt order across arms?
    same_prompts = {}
    for s in range(3):
        firsts = {}
        for arm in ARMS:
            with open(RUNS / f"{arm}_s{s}_cap512_r_box" / "rollouts.jsonl") as f:
                rows = []
                for ln in f:
                    r = json.loads(ln)
                    if r["step"] > 2:
                        break
                    rows.append(r["row"])
            firsts[arm] = rows
        same_prompts[s] = all(firsts[a] == firsts["ctrl"] for a in ARMS)
    summ = {k: {kk: v[kk] for kk in v if not isinstance(v[kk], list)} for k, v in data.items()}
    json.dump({"source": str(RUNS), "same_prompt_order_across_arms_per_seed": same_prompts, "summary": summ,
               "series": data}, open(OUT / "traj.json", "w"))
    for k, v in summ.items():
        print(k, v)
    print("same prompts per seed:", same_prompts)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for k, v in data.items():
        c = COL[v["arm"]]
        lab = v["arm"] if v["seed"] == 0 else None
        ax[0, 0].plot(roll(v["forced_frac"]), color=c, lw=1.4, label=lab)
        ax[0, 1].plot(roll(v["reasoning_mean"]), color=c, lw=1.4, label=lab)
        ax[1, 0].plot(roll(v["acc"]), color=c, lw=1.4, label=lab)
        ax[1, 1].plot(roll(v["len_groups"]), color=c, lw=1.4, label=lab)
    ax[0, 0].set_ylabel("share of rollouts cut at the 512 cap"); ax[0, 1].set_ylabel("mean reasoning tokens")
    ax[1, 0].set_ylabel("rollout accuracy"); ax[1, 1].set_ylabel("groups (of 16) with length signal")
    for a in ax[1]: a.set_xlabel("GRPO step")
    ax[0, 0].legend(frameon=False)
    fig.suptitle("Qwen3.5-4B length-penalty GRPO, cap 512: 10-step rolling means, 3 seeds per arm")
    fig.tight_layout(); fig.savefig(OUT / "traj_training.png", dpi=130)


if __name__ == "__main__":
    main()
