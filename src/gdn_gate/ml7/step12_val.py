#!/usr/bin/env python3
"""Helpers for step12_val_local.sh (ml-pipeline step 12, model evaluation on the validation slice, step 7).

plan --grpo DIR --res DIR --out PLAN.tsv [--seeds 0,1,2] [--warm_run R] [--warm_step N] [--arm_tag T]
     [--arm_step N] [--arm_cap N] [--answer_max N]
    (defaults: the step 7 record, Qwen3.5-2B; the overrides serve the pre-registered extensions: seeds 3-5 of the 2B
    record, and the Qwen3.5-4B record with its own warm-up run, tag, cap and answer cap)
    Finds the models to read: the untouched Qwen3.5-2B, the shared format warm-up (ctrl_s0_cap512_fmt_gb10,
    ckpt_step40.pt, DONE at step 40) and every arm run (ctrl / wide / wide1 x seeds 0, 1, 2, run tag r, cap 512) whose
    run dir has a DONE marker at step 150 and ckpt_step150.pt. A run with no dir, no DONE yet, or a DONE at another
    step (a stopped pilot) is pending and listed, not read. Refuses (exit 2, every reason printed) when the warm-up is
    missing or unfinished, or a finished arm's meta.json does not show its slot's arm, seed, cap 512 and answer_max
    1792, or was not initialized from the warm-up checkpoint (init_from.sha256 = sha256 of the warm-up checkpoint,
    applied true). Plan TSV columns: label, arm, ckpt ("none" for the untouched model), out, run_dir, step.

check --readout OUT.json --label L --arm A --ckpt CK --data VAL.jsonl --val_sha SHA --samples N --answer_max N
      --caps CAPS --seed N
    After a readout: schema wg-readout-v1, not a red arm, arm and checkpoint as planned, the readout arguments
    as planned, data sha256 = the frozen validation sha256, every item read N times. Exit 1 with the differences.

figures --analysis analysis_val.json --ml ML_DIR [--receipt FIG.json]
    The step 12 figures from the validation analysis (step14_analysis.py --split val), saved as
    ML_DIR/figures/12_<name>.png with the ml-pipeline guard's conventions (guard.fig: dpi 120, tight bbox, the
    `- Figure <file>: <explanation>` ledger line in ML_DIR/PIPELINE.md). The ledger line of a figure is replaced in
    place on a rerun (guard.fig would append a second line whenever the numbers change), so the figures can be redrawn
    as more arm runs finish. Every explanation is computed from the analysis and says "partial" while fewer than 9
    arm runs are present. Refuses an analysis of any split other than validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("ctrl", "wide", "wide1")
SEEDS = (0, 1, 2)
WARM_RUN, WARM_STEP = "ctrl_s0_cap512_fmt_gb10", 40
ARM_TAG, ARM_STEP = "cap512_r_gb10", 150
ARM_CAP, ARM_ANSWER_MAX = 512, 1792
SCHEMA = "wg-readout-v1"
COLORS = {"ctrl": "#1f77b4", "wide": "#d62728", "wide1": "#2ca02c"}
BASE_STYLE = {"untouched": dict(color="black", ls="--", marker="s"), "warm-up": dict(color="#7f7f7f", ls="-", marker="D")}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ plan
def _done_steps(rd: Path):
    try:
        return int(json.loads((rd / "DONE").read_text()).get("steps", -1))
    except (OSError, ValueError, AttributeError):
        return None


def make_plan(grpo: Path, res: Path, seeds=SEEDS, warm_run: str = WARM_RUN, warm_step: int = WARM_STEP,
              arm_tag: str = ARM_TAG, arm_step: int = ARM_STEP, arm_cap: int = ARM_CAP,
              answer_max: int = ARM_ANSWER_MAX) -> tuple[list[dict], list[str], list[str]]:
    WARM_RUN, WARM_STEP, ARM_TAG, ARM_STEP, ARM_CAP, ARM_ANSWER_MAX = (  # noqa: N806 (the record's constants)
        warm_run, warm_step, arm_tag, arm_step, arm_cap, answer_max)
    rows, pending, bad = [], [], []
    rows.append({"label": "untouched", "arm": "stock", "ckpt": "none", "out": str(res / "readout_val_untouched.json"),
                 "run_dir": "none", "step": "0"})
    wd = grpo / WARM_RUN
    wck = wd / f"ckpt_step{WARM_STEP}.pt"
    warm_sha = None
    if not wck.is_file():
        bad.append(f"warmup: checkpoint missing: {wck}")
    elif _done_steps(wd) != WARM_STEP:
        bad.append(f"warmup: no DONE at step {WARM_STEP} in {wd} (DONE steps {_done_steps(wd)})")
    else:
        warm_sha = sha(wck)
    rows.append({"label": "warmup", "arm": "ctrl", "ckpt": str(wck), "out": str(res / "readout_val_warmup.json"),
                 "run_dir": str(wd), "step": str(WARM_STEP)})
    for arm in ARMS:
        for seed in seeds:
            lab = f"{arm}_{seed}"
            rd = grpo / f"{arm}_s{seed}_{ARM_TAG}"
            ck = rd / f"ckpt_step{ARM_STEP}.pt"
            if not rd.is_dir():
                pending.append(f"{lab}: no run dir yet ({rd})")
                continue
            st = _done_steps(rd)
            if st is None:
                pending.append(f"{lab}: running or not finished (no readable DONE in {rd})")
                continue
            if st != ARM_STEP:
                pending.append(f"{lab}: DONE says steps {st}, waiting for {ARM_STEP} ({rd})")
                continue
            if not ck.is_file():
                bad.append(f"{lab}: DONE at step {ARM_STEP} but no checkpoint {ck}")
                continue
            try:
                meta = json.loads((rd / "meta.json").read_text())
            except (OSError, ValueError) as e:
                bad.append(f"{lab}: no readable meta.json in {rd} ({type(e).__name__})")
                continue
            args, init = meta.get("args", {}), meta.get("init_from") or {}
            for k, v in {"arm": arm, "seed": seed, "cap": ARM_CAP, "answer_max": ARM_ANSWER_MAX}.items():
                if args.get(k) != v:
                    bad.append(f"{lab}: meta.json args.{k} = {args.get(k)!r}, slot needs {v!r}")
            if warm_sha and init.get("sha256") != warm_sha:
                bad.append(f"{lab}: initialized from sha256 {str(init.get('sha256'))[:16]}, not the warm-up "
                           f"checkpoint {warm_sha[:16]}")
            if not init.get("applied", False):
                bad.append(f"{lab}: meta.json init_from.applied is not true")
            rows.append({"label": lab, "arm": arm, "ckpt": str(ck), "out": str(res / f"readout_val_{lab}.json"),
                         "run_dir": str(rd), "step": str(ARM_STEP)})
    return rows, pending, bad


def cmd_plan(a) -> int:
    seeds = tuple(int(x) for x in a.seeds.split(","))
    rows, pending, bad = make_plan(Path(a.grpo), Path(a.res), seeds, a.warm_run, a.warm_step, a.arm_tag, a.arm_step,
                                   a.arm_cap, a.answer_max)
    for p in pending:
        print(f"pending (not read this time): {p}")
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    keys = ("label", "arm", "ckpt", "out", "run_dir", "step")
    Path(a.out).write_text("".join("\t".join(r[k] for k in keys) + "\n" for r in rows))
    n = len(rows) - 2
    print(f"plan: untouched + warm-up + {n} of {len(ARMS) * len(seeds)} arm runs "
          f"({', '.join(r['label'] for r in rows[2:]) or 'none'}){'' if n == len(ARMS) * len(seeds) else ' (partial)'}")
    return 0


# ------------------------------------------------------------------------------------------------ check
def check_reasons(a) -> list[str]:
    p = Path(a.readout)
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return [f"{a.label}: readout unreadable ({type(e).__name__} {e})"]
    bad = []
    ar = d.get("args", {})
    want = {"schema": (d.get("schema"), SCHEMA), "arm": (d.get("arm"), a.arm), "ckpt": (ar.get("ckpt"), a.ckpt),
            "caps": (ar.get("caps"), a.caps), "answer_max": (ar.get("answer_max"), a.answer_max),
            "samples": (ar.get("samples"), a.samples), "seed": (ar.get("seed"), a.seed),
            "data sha256": ((d.get("env") or {}).get("sha256", {}).get("data"), a.val_sha)}
    for k, (got, need) in want.items():
        if got != need:
            bad.append(f"{a.label}: {k} {got!r}, planned {need!r}")
    if d.get("red"):
        bad.append(f"{a.label}: red-arm readout (red={d['red']!r})")
    ids = [json.loads(x)["id"] for x in open(a.data) if x.strip()]
    by: dict = {}
    for r in d.get("rows", []):
        by.setdefault(r["id"], set()).add(r["sample"])
    short = [i for i in ids if len(by.get(i, ())) != a.samples]
    if short or set(by) - set(ids):
        bad.append(f"{a.label}: {len(short)} items not read {a.samples} times (e.g. {short[:1]}), "
                   f"{len(set(by) - set(ids))} ids outside the data")
    return bad


def cmd_check(a) -> int:
    bad = check_reasons(a)
    if bad:
        print("CHECK FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print(f"check: {a.label} ok ({a.readout})")
    return 0


# ------------------------------------------------------------------------------------------------ figures
def _plt():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def ledger(md: Path, fname: str, explanation: str) -> None:
    """guard._ledger's line, replaced in place when the figure already has one; a new line goes after the last
    figure line (the Figures section), else at the end as guard.fig does."""
    line = f"- Figure {fname}: {' '.join(explanation.split())}"
    lines = md.read_text().splitlines() if md.is_file() else []
    mine = re.compile(r"^\s*-\s*Figure\s+" + re.escape(fname) + r":")
    hit = [i for i, x in enumerate(lines) if mine.match(x)]
    if hit:
        lines[hit[0]] = line
        for i in reversed(hit[1:]):
            del lines[i]
    else:
        anyfig = [i for i, x in enumerate(lines) if re.match(r"^\s*-\s*Figure\s+\S+\.png:", x)]
        if anyfig:
            lines.insert(anyfig[-1] + 1, line)
        else:
            lines.append(line)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n")


def save(plt, figure, ml: Path, name: str, explanation: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    out = ml / "figures" / f"12_{slug}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(figure)
    ledger(ml / "PIPELINE.md", out.name, explanation)
    return out


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _scope(r: dict) -> tuple[str, str]:
    present = {a: v for a, v in r["arms_present"].items() if v}
    runs = ", ".join(f"{a} seeds {'/'.join(map(str, v))}" for a, v in present.items()) or "no arm run yet"
    head = (f"Validation slice ({r['n_items']} held-out GSM8K train items x {r['samples_per_item']} samples, one GPU "
            f"for every model)")
    if r["partial"]:
        return head, (f"PARTIAL: {r['n_arm_runs']} of {r['n_arm_runs_planned']} arm runs present ({runs}); "
                      "redrawn as runs finish, no arm verdict until 3 seeds per arm")
    return head, f"All {r['n_arm_runs_planned']} arm runs present ({runs})"


def fig_budget(plt, r: dict):
    caps = r["caps"]
    x = [math.log2(c) for c in caps]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, blk in (("untouched", r["baseline"]), ("warm-up", r.get("warmup"))):
        if blk:
            ax.plot(x, [blk["acc"][str(c)] for c in caps], lw=2, ms=5, label=f"{name} (area {_f(blk['norm_auc_log2'])})",
                    **BASE_STYLE[name])
    for a, blk in r["arms"].items():
        for s, v in zip(blk["seeds"], zip(*[blk["acc"][str(c)]["per_seed"] for c in caps])):
            ax.plot(x, v, color=COLORS[a], lw=1, alpha=0.35)
        ax.plot(x, [blk["acc"][str(c)]["seed_mean"] for c in caps], color=COLORS[a], lw=2.4, marker="o", ms=5,
                label=f"{a}, mean of seeds {','.join(map(str, blk['seeds']))} (area {_f(blk['norm_auc_log2']['seed_mean'])})")
    ax.set_xticks(x, [str(c) for c in caps])
    ax.set_xlabel("reasoning budget (tokens, log2 scale)")
    ax.set_ylabel("accuracy")
    head, scope = _scope(r)
    ax.set_title(f"Step 12, {head}\naccuracy vs reasoning budget; thin lines = single seeds\n{scope}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    b, w = r["baseline"], r.get("warmup")
    parts = [f"untouched {_f(b['norm_auc_log2'])}"] + ([f"warm-up {_f(w['norm_auc_log2'])}"] if w else []) + \
            [f"{a} {_f(blk['norm_auc_log2']['seed_mean'])} (seeds {'/'.join(_f(v) for v in blk['norm_auc_log2']['per_seed'])})"
             for a, blk in r["arms"].items()]
    expl = (f"{head}: accuracy against reasoning budget 256 to 8,192 tokens for the untouched model, the shared format "
            f"warm-up and each arm (thick line = mean of its seeds, thin = single seeds). Area under the curve: "
            f"{'; '.join(parts)}. The arms start from the warm-up, so the warm-up curve is the bar: an arm above it "
            f"at small budgets got more accuracy for fewer reasoning tokens. {scope}.")
    return fig, expl


def fig_contrasts(plt, r: dict):
    eff = r["effects"]
    names = list(eff)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.8))
    for j, k in enumerate(names):
        e = eff[k]
        y = len(names) - 1 - j
        if e["rule"] == "descriptive":
            col, mk = "#7f7f7f", "D"
        elif not e["complete"]:
            col, mk = "#bbbbbb", "o"
        elif e["effect"]:
            col, mk = ("#2ca02c" if e["seed_mean_diff"] > 0 else "#d62728"), "o"
        else:
            col, mk = "#444444", "o"
        lo, hi = e["ci95"]
        ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="butt")
        ax.plot([e["seed_mean_diff"]], [y], marker=mk, color=col, ms=8)
        if e["seeds"] is not None:
            ax.scatter(e["per_seed_diff"], [y + 0.18] * len(e["per_seed_diff"]), s=14, color=col, alpha=0.7, zorder=3)
        ax.annotate(e["verdict"], (1.01, y), xycoords=("axes fraction", "data"), va="center", fontsize=7.5, color=col)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(names)), list(reversed(names)), fontsize=8)
    ax.set_xlabel("difference in normalized area under accuracy vs log2 budget (paired by item)")
    head, scope = _scope(r)
    ax.set_title(f"Step 12, {head}\ncontrasts: seed-mean difference, bootstrap 95% interval (bar), single seeds (dots)\n"
                 f"{scope}", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    txt = "; ".join(f"{k} {e['seed_mean_diff']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] "
                    f"{'descriptive' if e['rule'] == 'descriptive' else e['verdict']}" for k, e in eff.items())
    expl = (f"{head}: every pre-registered contrast on the primary metric, as the seed-mean difference in area with "
            f"its paired item bootstrap 95% interval and the single-seed differences. Rule: an arm effect needs all 3 "
            f"seeds with the same sign and an interval that excludes 0; warm-up minus untouched has one readout per "
            f"side, so it is descriptive. {txt}. {scope}.")
    return fig, expl


def fig_tokens(plt, r: dict):
    caps = r["caps"]
    big = str(max(caps))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    singles = [("untouched", r["baseline"]), ("warm-up", r.get("warmup"))]
    for name, blk in singles:
        if blk:
            ax.plot([blk["reasoning_mean"][str(c)] for c in caps], [blk["acc"][str(c)] for c in caps], lw=2, ms=5,
                    label=f"{name} (natural length {_f(blk['natural_len'], 0)})", **BASE_STYLE[name])
    for a, blk in r["arms"].items():
        ax.plot([blk["reasoning_mean"][str(c)]["seed_mean"] for c in caps],
                [blk["acc"][str(c)]["seed_mean"] for c in caps], color=COLORS[a], lw=2.4, marker="o", ms=5,
                label=f"{a}, seeds {','.join(map(str, blk['seeds']))} (natural length {_f(blk['natural_len']['seed_mean'], 0)})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("mean reasoning tokens used (log2 scale); one point per budget 256..8192")
    ax.set_ylabel("accuracy")
    head, scope = _scope(r)
    ax.set_title(f"Step 12, {head}\naccuracy vs reasoning tokens actually used\n{scope}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    tgt = r["baseline"]["acc"][big]

    def tk(v):
        return "never" if v is None else f"{v:.0f}"
    parts = [f"untouched {_f(r['baseline']['natural_len'], 0)} tokens, cut off {_f(r['baseline']['cut_off_share_at_max'])}"]
    if r.get("warmup"):
        w = r["warmup"]
        parts.append(f"warm-up {_f(w['natural_len'], 0)}, cut off {_f(w['cut_off_share_at_max'])}, reaches the untouched "
                     f"8k accuracy at {tk(w['tokens_to_base_max_acc'])}")
    for a, blk in r["arms"].items():
        parts.append(f"{a} {_f(blk['natural_len']['seed_mean'], 0)}, cut off {_f(blk['cut_off_share_at_max']['seed_mean'])}, "
                     f"reaches it at {tk(blk['tokens_to_base_max_acc']['from_seed_mean_curve'])}")
    expl = (f"{head}: accuracy against the mean reasoning tokens each model actually spent at each budget, so a curve "
            f"further left at the same height reasons less for the same accuracy. Natural length (mean reasoning "
            f"tokens at 8,192) and budget to reach the untouched model's 8k accuracy ({_f(tgt)}): {'; '.join(parts)}. "
            f"{scope}.")
    return fig, expl


def cmd_figures(a) -> int:
    r = json.loads(Path(a.analysis).read_text())
    if r.get("split") != "val":
        print(f"REFUSED: {a.analysis} is a {r.get('split')!r} analysis; step 12 figures come from the validation "
              "split only (step14_analysis.py --split val)", file=sys.stderr)
        return 2
    plt = _plt()
    ml = Path(a.ml)
    out = []
    for name, fn in (("val_accuracy_vs_budget", fig_budget), ("val_area_contrasts", fig_contrasts),
                     ("val_accuracy_vs_tokens", fig_tokens)):
        figure, expl = fn(plt, r)
        p = save(plt, figure, ml, name, expl)
        out.append({"file": str(p), "sha256": sha(p), "bytes": p.stat().st_size, "explanation": expl})
        print(f"figure {p} ({p.stat().st_size} bytes)")
    if a.receipt:
        Path(a.receipt).write_text(json.dumps({"analysis": a.analysis, "analysis_sha256": sha(a.analysis),
                                               "partial": r["partial"], "n_arm_runs": r["n_arm_runs"], "figures": out},
                                              indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--grpo", required=True)
    p.add_argument("--res", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--warm_run", default=WARM_RUN)
    p.add_argument("--warm_step", type=int, default=WARM_STEP)
    p.add_argument("--arm_tag", default=ARM_TAG)
    p.add_argument("--arm_step", type=int, default=ARM_STEP)
    p.add_argument("--arm_cap", type=int, default=ARM_CAP)
    p.add_argument("--answer_max", type=int, default=ARM_ANSWER_MAX)
    p = sub.add_parser("check")
    for k in ("readout", "label", "arm", "ckpt", "data", "val_sha", "caps"):
        p.add_argument(f"--{k}", required=True)
    for k in ("samples", "answer_max", "seed"):
        p.add_argument(f"--{k}", type=int, required=True)
    p = sub.add_parser("figures")
    p.add_argument("--analysis", required=True)
    p.add_argument("--ml", required=True)
    p.add_argument("--receipt", default=None)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "check": cmd_check, "figures": cmd_figures}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
