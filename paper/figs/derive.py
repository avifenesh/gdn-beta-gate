"""Collect the numbers the paper needs that are not a single field of one analysis receipt.

CPU only. Reads run logs and receipts, writes paper/data/derived.json. Every value carries its source path.
Run: python3 paper/figs/derive.py
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
RUNS = RES / "run_logs"  # the small training and validation logs the paper reads, copied from the run dirs
OUT = ROOT / "paper/data/derived.json"


def rel(p):
    """Repository-relative path, so derived.json is the same on every machine."""
    return str(Path(p).relative_to(ROOT))


def last_steps(path, n=20):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "forced_frac" in r]
    tail = rows[-n:]
    return {
        "steps": [tail[0]["step"], tail[-1]["step"]],
        "forced_frac": st.mean(r["forced_frac"] for r in tail),
        "reasoning_mean": st.mean(r["reasoning_mean"] for r in tail),
        "acc": st.mean(r["acc"] for r in tail),
        "source": rel(path),
    }


def train_signature():
    out = {"2B": {}, "4B": {}}
    for arm in ("ctrl", "wide", "wide1"):
        for s in range(6):
            tag = "gb10" if s < 3 else "box"
            p = RUNS / f"ml7/grpo/{arm}_s{s}_cap512_r_{tag}/train_log.jsonl"
            if p.exists():
                out["2B"][f"{arm}_s{s}"] = last_steps(p)
        for s in range(3):
            p = RUNS / f"ml7_4b_runs/{arm}_s{s}_cap512_r_box/train_log.jsonl"
            if p.exists():
                out["4B"][f"{arm}_s{s}"] = last_steps(p)
    for s in range(3):
        p = RUNS / f"ml7_4b/grpo/ctrlpm_s{s}_cap512_r_box/train_log.jsonl"
        if p.exists():
            out["4B"][f"ctrlpm_s{s}"] = last_steps(p)
    return out


def cond_lengths():
    files = {
        "2B_touch1_gsm8k": "ml7/step14/analysis_gsm8k.json",
        "2B_touch1_math500": "ml7/step14/analysis_math500.json",
        "4B_gsm8k": "ml7_4b/step14_ctrlpm/analysis_gsm8k.json",
        "4B_math500": "ml7_4b/step14_ctrlpm/analysis_math500.json",
    }
    out = {}
    for k, f in files.items():
        d = json.load(open(RES / f))
        row = {}
        for arm, a in d["arms"].items():
            nat = a["cond_length"]["natural"]
            row[arm] = {
                "correct_mean_seed_mean": nat["correct"]["mean_seed_mean"],
                "incorrect_mean_seed_mean": nat["incorrect"]["mean_seed_mean"],
                "cut_off_share_at_max_seed_mean": a["cut_off_share_at_max"]["seed_mean"],
                "natural_len_seed_mean": a["natural_len"]["seed_mean"],
            }
        for name in ("baseline", "warmup"):
            nat = d[name]["cond_length"]["natural"]
            row[name] = {"correct_mean": nat["correct"]["mean"], "natural_len": d[name]["natural_len"],
                         "cut_off_share_at_max": d[name]["cut_off_share_at_max"]}
        out[k] = {"source": rel(RES / f), "field": "arms.<arm>.cond_length.natural", "rows": row}
    return out


def step6():
    rows = {}
    for arm in ("ctrl", "wide", "wide1"):
        for s in range(6):
            p = RUNS / f"broad/step6/{arm}_s{s}/val_log.jsonl"
            last = [json.loads(l) for l in open(p) if l.strip()][-1]
            assert last["step"] == 1200, p
            rows[f"{arm}_s{s}"] = {"val_loss": last["val_loss"], "gate_stat_mean": last.get("gate_stat_mean"),
                                   "source": rel(p)}
    paired = {}
    for a, b in (("wide1", "ctrl"), ("wide", "ctrl"), ("wide", "wide1")):
        diffs = [rows[f"{a}_s{s}"]["val_loss"] - rows[f"{b}_s{s}"]["val_loss"] for s in range(6)]
        paired[f"{a} - {b}"] = {"per_seed": diffs, "mean": st.mean(diffs), "sd": st.stdev(diffs),
                                "n_negative": sum(x < 0 for x in diffs)}
    return {"rows": rows, "paired": paired}


def he(path):
    d = json.load(open(path))
    return {"pass1": d["he"]["pass1"], "gpu": d.get("gpu"), "source": rel(path)}


def forgetting():
    r4 = RES / "rescore/rescore4b_rtx5090"
    s4 = RES / "rescore/rescore4b_gb10"
    w1 = RES / "w1main/rescore4b"
    r9 = RES / "rescore/rescore9b"
    out = {
        "4B_base_untouched": he(RES / "vllm/step6_s0/base.json"),
        "4B_main_ctrl": {f"s{s}": he(r4 / f"main_ctrl_s{s}.json") for s in range(3)},
        "4B_main_ctrl_extra": {"s4": he(s4 / "main_ctrl_s4.json"), "s5": he(s4 / "main_ctrl_s5.json")},
        "4B_main_wide": {f"s{s}": he(r4 / f"main_wide_s{s}.json") for s in range(3)},
        "4B_main_wide_strip": {f"s{s}": he(r4 / f"main_wide_s{s}_strip.json") for s in range(3)},
        "4B_main_wide1": {f"s{s}": he(w1 / f"main_wide1_s{s}_box2.json") for s in range(3)},
        "4B_main_wide1_strip": {f"s{s}": he(w1 / f"main_wide1_s{s}_box2_strip.json") for s in range(3)},
        "4B_same_hw_s0": {"ctrl": he(w1 / "main_ctrl_s0_box2.json"), "wide": he(w1 / "main_wide_s0_box2.json"),
                          "wide1": he(w1 / "main_wide1_s0_box2.json")},
        "4B_scorer_check": {"ctrl_s0_rtx5090": he(r4 / "main_ctrl_s0.json"), "ctrl_s0_box": he(w1 / "main_ctrl_s0_boxcheck.json"),
                            "wide_s0_rtx5090": he(r4 / "main_wide_s0.json"), "wide_s0_box": he(w1 / "main_wide_s0_boxcheck.json")},
        "9B_main_ctrl": {f"s{s}": he(r9 / f"main9_ctrl_s{s}.json") for s in range(3)},
        "9B_main_wide": {f"s{s}": he(r9 / f"main9_wide_s{s}.json") for s in range(3)},
        "9B_ref_100step": he(r9 / "base9_ctrl_s100.json"),
        "9B_scorer_check": {"ctrl_s0_gb10": he(r9 / "main9_ctrl_s0.json"), "ctrl_s0_box": he(r9 / "main9_ctrl_s0_boxcheck.json")},
        "4B_step6_s0_vllm": {k: he(RES / f"vllm/step6_s0/{k}.json") for k in ("base", "ctrl_s0", "wide_s0", "wide1_s0")},
    }
    for k in ("4B_main_ctrl", "4B_main_wide", "4B_main_wide1", "9B_main_ctrl", "9B_main_wide"):
        out[k]["mean_s0_s2"] = st.mean(v["pass1"] for kk, v in out[k].items() if kk.startswith("s"))
    return out


def olmo():
    lm = {}
    for arm in ("stock", "erase", "half"):
        d = json.load(open(RES / f"olmo/vllm2/battery/{arm}/lmeval_lm.json"))["results"]
        lm[arm] = {
            "arc_challenge_acc_norm": d["arc_challenge"]["acc_norm,none"],
            "gsm8k_flexible": d["gsm8k"]["exact_match,flexible-extract"],
            "gsm8k_strict": d["gsm8k"]["exact_match,strict-match"],
            "humaneval": d["humaneval"]["pass@1,create_test"],
            "mbpp_plus": d["mbpp_plus"]["pass_at_1,none"],
            "swde": d["swde"]["contains,none"],
            "fda": d["fda"]["contains,none"],
            "squad_completion": d["squad_completion"]["contains,none"],
            "source": rel(RES / f"olmo/vllm2/battery/{arm}/lmeval_lm.json"),
        }
    mmlu = {}
    for arm in ("stock", "erase", "half"):
        p = RUNS / f"olmo/battery/{arm}/mmlu_cloze.json"
        d = json.load(open(p))
        mmlu[arm] = {"cloze_acc": d["cloze_acc"], "n": d["n"], "source": rel(p)}
    use = json.load(open(RES / "olmo/vllm2/usage_by_task_8k.json"))
    usage = {}
    for t, v in use["tasks"].items():
        pl = v["per_layer"]
        q = [st.mean(pl[i * 6:(i + 1) * 6]) for i in range(4)]
        usage[t] = {"gt1_frac": v["gt1_frac"], "quarter_means": q}
    rp = json.load(open(RES / "olmo/vllm2/ruler_paired.json"))
    nc = [r for r in rp["rows"] if r["arm"] == "erase" and r["task"] != "ruler_cwe"]
    noncwe = {"n_rows": len(nc), "max_abs_diff": max(abs(r["diff"]) for r in nc), "min_p": min(r["p"] for r in nc),
              "source": rel(RES / "olmo/vllm2/ruler_paired.json")}
    return {"lm": lm, "mmlu_cloze": mmlu, "ruler_nonCWE_erase": noncwe,
            "usage_8k": {"source": rel(RES / "olmo/vllm2/usage_by_task_8k.json"), "n_items": use["limit"], "tasks": usage}}


MECH = RES / "mech"


def _rng(vals, nd=None):
    return {"min": min(vals), "max": max(vals), "mean": st.mean(vals)}


def mech():
    """Mechanism numbers (validation slice): G1 ablation, G2 dose, G3 gate stats, CPU text and training analyses."""
    g = json.load(open(MECH / "g1g2_summary.json"))
    cm = g["g1_condition_means"]
    pr = g["per_readout"]
    g1 = {"source": rel(MECH / "g1g2_summary.json")}
    for arm in ("wide1", "wide"):
        g1[arm] = {
            "hybrid_cut_of_warmup": 1 - cm[f"{arm}_hybrid"]["natural_len"] / cm["warmup"]["natural_len"],
            "strip_cut_vs_ctrl": 1 - cm[f"{arm}_strip"]["natural_len"] / cm["ctrl"]["natural_len"],
            "full_cut_vs_ctrl": 1 - cm[f"{arm}_full"]["natural_len"] / cm["ctrl"]["natural_len"],
            "full_cut_vs_strip": 1 - cm[f"{arm}_full"]["natural_len"] / cm[f"{arm}_strip"]["natural_len"],
        }
    ctrl_cut_of_warmup = 1 - cm["ctrl"]["natural_len"] / cm["warmup"]["natural_len"]
    g1["ctrl_cut_of_warmup"] = ctrl_cut_of_warmup
    g1_keys = [k for k in pr if not k.split("_")[-1].startswith("x")]
    g1["area_range_excl_warmup"] = _rng([pr[k]["norm_auc_log2"] for k in g1_keys if k != "warmup"])
    g1["closed_range_excl_warmup"] = _rng([pr[k]["closed_share_8192"] for k in g1_keys if k != "warmup"])
    g1["n_readouts"] = len(g1_keys)
    con = g["g1_contrasts_descriptive"]
    g1["area_contrasts_with_rule_shape"] = sorted(k for k, v in con.items() if v["norm_auc_log2"]["rule_shape_effect"])
    g1["area_rule_shape_only_vs_warmup"] = all(k.endswith("- warmup") for k in g1["area_contrasts_with_rule_shape"])
    g1["length_contrasts_all_rule_shape"] = all(v["natural_len"]["rule_shape_effect"] for v in con.values())
    wide_full = [pr[f"wide_s{s}_full"]["beta_gt1_share"] for s in range(3)]
    g1["wide_full_beta_gt1"] = _rng(wide_full)
    g1["wide_full_beta_max"] = _rng([pr[f"wide_s{s}_full"]["beta_max"] for s in range(3)])
    dose = g["g2_dose"]
    g2 = {"source": rel(MECH / "g1g2_summary.json")}
    areas, mono, first_step_share = [], [], []
    for s in range(3):
        pts = sorted(dose[f"wide1_s{s}"]["points"], key=lambda r: r["scale"])
        L = [r["natural_len"] for r in pts]
        areas += [r["norm_auc_log2"] for r in pts]
        mono.append(all(b < a for a, b in zip(L, L[1:])))
        first_step_share.append((L[0] - L[1]) / (L[0] - L[-1]))
    g2["area_range_all_scales"] = _rng(areas)
    g2["monotone_all_seeds"] = all(mono)
    g2["share_of_0_to_2_drop_in_0_to_0p5"] = _rng(first_step_share)
    g2["slopes"] = _rng([dose[f"wide1_s{s}"]["natural_len_slope_per_unit_scale"] for s in range(3)])
    g2["x0_vs_strip"] = {f"s{s}": [pr[f"wide1_s{s}_x0"]["natural_len"], pr[f"wide1_s{s}_strip"]["natural_len"]] for s in range(3)}

    gs = json.load(open(MECH / "gate_stats_summary/gate_stats_summary_fixed.json"))
    R = gs["runs"]
    g3 = {"source": rel(MECH / "gate_stats_summary/gate_stats_summary_fixed.json"), "n_tokens": gs["n_tokens"],
          "n_seq": gs["n_seq"], "warmup_s_mean": gs["warmup"]["s_mean"], "warmup_s_lt_0p1": gs["warmup"]["s_lt_0p1_share"]}
    g3["n_runs"] = len(R)
    f150 = lambda arm: [R[f"{arm}_s{s}_150"] for s in range(3)]  # noqa: E731
    for arm in ("wide1", "wide"):
        rr = f150(arm)
        g3[arm] = {"gate_abs_dbeta_150": _rng([r["gate_d_abs_mean"] for r in rr]),
                   "gate_net_dbeta_150": _rng([r["gate_d_mean"] for r in rr]),
                   "lora_abs_dbeta_150": _rng([r["lora_ds_abs_mean"] for r in rr]),
                   "beta_gt1_150": _rng([r["beta_gt1_share"] for r in rr]),
                   "beta_gt1_25": _rng([R[f"{arm}_s{s}_25"]["beta_gt1_share"] for s in range(3)]),
                   "heads_gt_0p05_150": _rng([r["heads_gate_abs_d_gt_0p05"] for r in rr]),
                   "heads_gt_0p05_25": _rng([R[f"{arm}_s{s}_25"]["heads_gate_abs_d_gt_0p05"] for s in range(3)])}
        lay = [st.mean(r["gate_d_abs_by_layer"][i] for r in rr) for i in range(24)]
        g3[arm]["gate_abs_dbeta_by_layer_seedmean"] = lay
        g3[arm]["layers_1_9_range"] = _rng(lay[1:10])
        g3[arm]["last8_range"] = _rng(lay[16:24])
        g3[arm]["first_layer"] = lay[0]
        if arm == "wide":
            b = [st.mean(r["beta_gt1_by_layer"][i] for r in rr) for i in range(24)]
            g3[arm]["beta_gt1_by_layer_seedmean"] = b
            w1 = g3["wide1"]["gate_abs_dbeta_by_layer_seedmean"]
            g3["wide_over_wide1_by_layer"] = _rng([a / c for a, c in zip(lay, w1)])
            g3[arm]["beta_gt1_last6_range"] = _rng(b[18:24])
            g3[arm]["beta_gt1_first_layer"] = b[0]
            g3[arm]["beta_gt1_layers_1_17_max"] = max(b[1:18])
    ctl = [R[f"{a}_s{s}_150"] for a in ("ctrl", "ctrlpm") for s in range(3)]
    g3["ctrl_ctrlpm_lora_abs_dbeta_150"] = _rng([r["lora_ds_abs_mean"] for r in ctl])
    g3["gated_lora_abs_dbeta_150"] = _rng([R[f"{a}_s{s}_150"]["lora_ds_abs_mean"] for a in ("wide1", "wide") for s in range(3)])
    g3["ratio_gate_wide1_over_ctrl_lora_means"] = g3["wide1"]["gate_abs_dbeta_150"]["mean"] / g3["ctrl_ctrlpm_lora_abs_dbeta_150"]["mean"]
    g3["ratio_gate_wide_over_ctrl_lora_means"] = g3["wide"]["gate_abs_dbeta_150"]["mean"] / g3["ctrl_ctrlpm_lora_abs_dbeta_150"]["mean"]
    g3["ratio_gate_wide1_over_own_lora_means"] = g3["wide1"]["gate_abs_dbeta_150"]["mean"] / g3["wide1"]["lora_abs_dbeta_150"]["mean"]
    g3["ratio_gate_wide_over_own_lora_means"] = g3["wide"]["gate_abs_dbeta_150"]["mean"] / g3["wide"]["lora_abs_dbeta_150"]["mean"]
    g3["decay_abs_change_max_all_runs"] = max(abs(r["decay_change_mean"]) for r in R.values())
    g3["t_abs_by_step_mean_6runs"] = {str(k): st.mean(R[f"{a}_s{s}_{k}"]["gate_t_abs_mean"] for a in ("wide1", "wide") for s in range(3))
                                     for k in (25, 50, 75, 100, 125, 150)}
    g3["gate_abs_dbeta_by_step"] = {a: {str(k): st.mean(R[f"{a}_s{s}_{k}"]["gate_d_abs_mean"] for s in range(3))
                                        for k in (25, 50, 75, 100, 125, 150)} for a in ("wide1", "wide")}
    g3["ctrl_lora_abs_dbeta_by_layer_seedmean"] = [st.mean(R[f"ctrl_s{s}_150"]["lora_ds_abs_by_layer"][i] for s in range(3)) for i in range(24)]

    ts = json.load(open(MECH / "text_stats.json"))
    text = {"source": rel(MECH / "text_stats.json")}
    for ds in ("gsm8k", "math500"):
        m = {a: ts[ds][a]["mean"] for a in ("ctrl", "ctrlpm", "wide1", "wide")}
        tot = {a: m[a]["tokens_to_first_gold"] + m[a]["tokens_after_first_gold"] for a in m}
        text[ds] = {"post_answer_share_of_saving_wide1_vs_ctrl":
                    (m["ctrl"]["tokens_after_first_gold"] - m["wide1"]["tokens_after_first_gold"]) / (tot["ctrl"] - tot["wide1"])}
    tj = json.load(open(MECH / "traj_extra.json"))
    win = {}
    for w, row in tj["natural_close_rate_by_window"].items():
        win[w] = {a: st.mean(row[f"{a}_s{s}"] for s in range(3)) for a in ("ctrl", "ctrlpm", "wide1", "wide")}
    cum = tj["cum_length_signal_groups_to_reach_natural_close"]
    gated10 = [cum[f"{a}_s{s}"]["0.1"]["cum_groups"] for a in ("wide1", "wide") for s in range(3)]
    traj = {"source": rel(MECH / "traj_extra.json"), "natural_close_by_window_seedmean": win,
            "cum_groups_to_0p1_gated": _rng(gated10),
            "cum_groups_to_0p1_ctrl": [cum[f"ctrl_s{s}"]["0.1"]["cum_groups"] for s in range(3)],
            "cum_groups_to_0p1_ctrlpm": [cum[f"ctrlpm_s{s}"]["0.1"]["cum_groups"] for s in range(3)]}
    traj["s0_never_reach_0p1_ctrl_and_ctrlpm"] = (cum["ctrl_s0"]["0.1"]["cum_groups"] is None
                                                  and cum["ctrlpm_s0"]["0.1"]["cum_groups"] is None)
    t = json.load(open(MECH / "traj.json"))["summary"]
    traj["acc_last20_range_all"] = _rng([v["acc_last20"] for v in t.values()])
    return {"g1": g1, "g2": g2, "g3": g3, "text": text, "traj": traj}


def fresh():
    """Fresh held-out set: values that are not one field of the analysis receipts."""
    out = {}
    for rec in ("4b", "2b"):
        p = RES / f"fresh/analysis_fresh_{rec}.json"
        d = json.load(open(p))
        arms = d["arms"]
        row = {"source": rel(p),
               "wide1_vs_ctrl_len_cut": 1 - arms["wide1"]["natural_len"]["seed_mean"] / arms["ctrl"]["natural_len"]["seed_mean"]}
        src = d["per_source_descriptive"]
        row["per_source_all_gated_above_ctrl"] = all(
            src[k]["arms"][a]["norm_auc_log2"]["seed_mean"] > src[k]["arms"]["ctrl"]["norm_auc_log2"]["seed_mean"]
            for k in src for a in ("wide1", "wide"))
        out[rec] = row
    return out


def main():
    out = {"train_signature_last20": train_signature(), "cond_length": cond_lengths(), "step6_sft": step6(),
           "forgetting_humaneval": forgetting(), "olmo": olmo(),
           "mech": mech(), "fresh": fresh()}
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
