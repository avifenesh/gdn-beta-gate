#!/usr/bin/env python3
"""CPU checks for step14_analysis.py on real readouts (the step 7 parity files: 50 val items x 8 samples, stock and
red arms) used as fake arms and seeds. Copies are written under ~/.cache/gdn_gate/ml7/step14_test with the arm
label, red field and readout seed set so the input checks pass; the raw files are the red arm of the input checks.
The parity files read val.jsonl, so cases 1-6 and 8-12 run with --split val; case 7 relabels the data as a test file.

1. identical: every slot = the baseline file -> every difference is exactly 0, its interval contains 0, no effect.
2. red: wide seeds = red readouts (noclose, half, noclose) vs ctrl = stock readouts -> strongly negative, the interval
   excludes 0 and the rule reports a negative effect.
3. sign: wide = (noclose, noclose, stock) vs ctrl = (stock, stock, stock_s2): the interval excludes 0 but seed 2 has
   the opposite sign -> no effect.
4. refusals: raw files (readout seed, arm label, red), caps differ, items differ, samples per item differ, a missing
   slot, a duplicated checkpoint -> exit 2 naming the difference.
5. checkpoint noise: the spread equals the hand-computed area difference.
6. answer tokens: unavailable without them; rows carrying a 6th element are reported; with the Qwen3.5-2B tokenizer
   (if transformers imports) the traces sidecar route reports counts.
7. MATH-500 unseen-450 check: ids renamed to 25 pilot ids + 25 others -> the subset has 25 items.
8. determinism: two runs give the same intervals.
9. warm-up role (second baseline, arm ctrl with a checkpoint): identical inputs give arm - warm-up = 0; red wide arms
   vs a stock warm-up give a negative effect; a red warm-up vs the untouched model is reported descriptively (no
   verdict) although its interval excludes 0; refusals: warm-up with arm stock, warm-up without a checkpoint, the
   warm-up checkpoint in an arm slot, the warm-up given as --baseline.
10. split labels: --split val is labeled step 12 / validation; --split test on validation data, --split val on data
    with a test sha256, and --split val with test ids are refused.
11. partial (validation while runs train): missing slots refused without --allow_partial, --allow_partial refused on
    the test split; with 5 of 9 runs the receipt says partial, contrasts use the seeds present and a contrast with
    fewer than 3 seeds has no verdict although its interval excludes 0; a diagnostic readout without its final slot is
    refused; with all 9 present the receipt is complete.
12. one device: a file read on another GPU is refused.
13. seed set (--seeds): the same files labeled seeds 3, 4, 5 and read with --seeds 3,4,5 give the numbers and verdicts
    of seeds 0, 1, 2 under the default; seed-0 labels with --seeds 3,4,5 are refused (outside the set, set missing);
    the pooled set 0-5 uses the 6/6 rule (three negative and three zero seeds: no effect although the interval excludes
    0); a later default call is the 3/3 rule again.
14. natural length under the rule (length_effects) and the pre-registered hypotheses: identical inputs give 0 and no
    effect; never-closing wide readouts give a positive length effect equal to the arms' natural_len difference, with
    the interval recomputed by hand from the shared index matrix; on data relabeled as GSM8K test, H2 is supported when
    wide is shorter on 3/3 seeds, and red arms (wide longer; signs disagreeing) are not supported (the length red
    arm is built: the stock readout with every think_len halved); on MATH-500 test H1
    is supported when wide's area is higher on 3/3 seeds and not supported with red wide readouts; each hypothesis is
    not applicable on the other set.
Usage: test_step14_cpu.py [--out FILE]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step14_analysis as S  # noqa: E402

PAR = Path(__file__).resolve().parents[3] / "results/ml7/readout_parity"  # released fixtures (2B validation smoke readouts)
SRC = {"stock": PAR / "v_stock_k8.json", "stock_s2": PAR / "v_stock_k8_s2.json",
       "noclose": PAR / "v_red_noclose_k8.json", "half": PAR / "v_red_half_k8.json"}
SCR = Path.home() / ".cache/gdn_gate/ml7/step14_test"
TOK = os.path.join(os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub")), "models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc")
SEED = 20261006
RAW = {k: json.loads(v.read_text()) for k, v in SRC.items()}
RESULTS: list[dict] = []


def put(name: str, src: str, arm: str, mutate=None) -> str:
    d = copy.deepcopy(RAW[src])
    d["arm"], d["red"], d["args"]["seed"] = arm, "", SEED
    if mutate:
        mutate(d)
    p = SCR / f"{name}.json"
    p.write_text(json.dumps(d))
    side = SCR / f"{name}.json.traces.jsonl.gz"
    if side.is_symlink() or side.exists():
        side.unlink()
    os.symlink(str(SRC[src]) + ".traces.jsonl.gz", side)
    return str(p)


def run(base: str, slots: dict, extra=(), tok=None, manifest=None, split="val", warm=None, partial=False):
    argv = ["--split", split, "--baseline", base] + (["--warmup", warm] if warm else []) + \
           [f"{k}={v}" for k, v in slots.items()] + list(extra) + ["--out", str(SCR / "receipt.json")] + \
           (["--allow_partial"] if partial else [])
    if tok:
        argv += ["--tokenizer", tok]
    if manifest:
        argv += ["--manifest", manifest]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            res = S.main(argv)
            code = 0
        except SystemExit as e:
            res, code = None, e.code
    return res, code, out.getvalue(), err.getvalue()


def check(name: str, ok: bool, detail) -> None:
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def warmck(d):
    d["ckpt"] = "/runs/ml7/grpo/ctrl_s0_cap512_fmt_gb10/ckpt_step40.pt"
    d["env"]["sha256"]["ckpt"] = "cd" * 32


def slots_of(ctrl, wide, wide1):
    s = {}
    for arm, files in (("ctrl", ctrl), ("wide", wide), ("wide1", wide1)):
        for i, f in enumerate(files):
            s[f"{arm}_{i}"] = put(f"{arm}_{i}", f, arm)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCR / "test_step14_cpu.json"))
    a = ap.parse_args()
    SCR.mkdir(parents=True, exist_ok=True)
    base = put("base", "stock", "stock")

    # 1. identical inputs
    res, code, out, err = run(base, slots_of(["stock"] * 3, ["stock"] * 3, ["stock"] * 3))
    e = res["effects"]
    zero = all(v["seed_mean_diff"] == 0 and all(x == 0 for x in v["per_seed_diff"]) for v in e.values())
    cont = all(v["ci95"][0] <= 0 <= v["ci95"][1] for v in e.values())
    noeff = all(not v["effect"] for v in e.values())
    check("identical: diff 0, interval contains 0, no effect", code == 0 and zero and cont and noeff,
          {k: [v["seed_mean_diff"], v["ci95"], v["verdict"]] for k, v in e.items()})
    tb = res["baseline"]["tokens_to_base_max_acc"]
    check("identical: tokens to the untouched 8k accuracy match the baseline's own",
          all(res["arms"][x]["tokens_to_base_max_acc"]["from_seed_mean_curve"] == tb for x in S.ARMS), tb)
    print(out)

    # 2. red arms strongly negative, effect rule passes on the interval (and the signs)
    res, code, out, err = run(base, slots_of(["stock", "stock_s2", "stock"], ["noclose", "half", "noclose"],
                                             ["stock_s2", "stock", "stock_s2"]))
    w = res["effects"]["wide - ctrl"]
    check("red: wide (red files) - ctrl strongly negative, interval excludes 0, negative effect",
          code == 0 and w["seed_mean_diff"] < -0.2 and w["ci95"][1] < 0 and w["ci_excludes_0"] and w["effect"]
          and w["verdict"] == "effect, negative", {k: w[k] for k in ("per_seed_diff", "seed_mean_diff", "ci95", "verdict")})
    wu = res["effects"]["wide - untouched"]
    check("red: wide - untouched also a negative effect", wu["effect"] and wu["seed_mean_diff"] < -0.2,
          {k: wu[k] for k in ("per_seed_diff", "seed_mean_diff", "ci95")})
    hw = res["effects"]["wide - wide1"]["per_seed_diff"][1]
    check("red: the half red arm alone is negative but small (seed 1 of wide - wide1)", -0.1 < hw < 0, hw)
    print(out)

    # 3. sign disagreement fails the rule even with an interval excluding 0
    res, code, out, err = run(base, slots_of(["stock", "stock", "stock_s2"], ["noclose", "noclose", "stock"],
                                             ["stock"] * 3))
    w = res["effects"]["wide - ctrl"]
    check("sign: seeds disagree -> no effect although the interval excludes 0",
          code == 0 and w["ci_excludes_0"] and not w["signs_agree"] and not w["effect"] and w["per_seed_diff"][2] > 0,
          {k: w[k] for k in ("per_seed_diff", "seed_mean_diff", "ci95", "verdict")})

    # 4. refusals
    good = slots_of(["stock", "stock_s2", "stock"], ["noclose", "half", "noclose"], ["stock_s2", "stock", "stock_s2"])
    raw = {k: str(SRC[f]) for k, f in zip(good, ["stock", "stock_s2", "stock", "noclose", "half", "noclose",
                                                  "stock_s2", "stock", "stock_s2"])}
    _, code, _, err = run(str(SRC["stock"]), raw)
    check("refuse: raw files (readout seed, arm label, red)",
          code == 2 and "readout seed" in err and "does not match the label" in err and "red-arm" in err,
          err.strip().splitlines()[:4])

    def drop_cap(d):
        d["caps"] = d["caps"][:-1]
        for r in d["rows"]:
            r["caps"].pop("8192")
    _, code, _, err = run(base, {**good, "wide_1": put("bad_caps", "half", "wide", drop_cap)})
    check("refuse: caps differ", code == 2 and "caps" in err, err.strip().splitlines()[:2])

    def drop_item(d):
        gone = d["rows"][0]["id"]
        d["rows"] = [r for r in d["rows"] if r["id"] != gone]
    _, code, _, err = run(base, {**good, "ctrl_2": put("bad_items", "stock", "ctrl", drop_item)})
    check("refuse: items differ", code == 2 and "items differ" in err, err.strip().splitlines()[:2])

    def drop_sample(d):
        d["rows"] = [r for r in d["rows"] if r["sample"] != 7]
        d["args"]["samples"] = 7
    _, code, _, err = run(put("bad_base", "stock", "stock", drop_sample), good)
    check("refuse: baseline sample count differs", code == 2 and "samples per item" in err, err.strip().splitlines()[:2])

    def other_seed(d):
        d["args"]["seed"] = SEED + 1
    _, code, _, err = run(put("bad_seed", "stock", "stock", other_seed), good)
    check("refuse: baseline readout seed differs", code == 2 and "readout seed" in err, err.strip().splitlines()[:2])

    miss = {k: v for k, v in good.items() if k != "wide1_2"}
    _, code, _, err = run(base, miss)
    check("refuse: missing slot", code == 2 and "wide1_2: missing" in err, err.strip().splitlines()[:2])

    def ck(d):
        d["env"]["sha256"]["ckpt"] = "ab" * 32
    dup = {**good, "ctrl_0": put("dup0", "stock", "ctrl", ck), "wide_0": put("dup1", "noclose", "wide", ck)}
    _, code, _, err = run(base, dup)
    check("refuse: same checkpoint in two slots", code == 2 and "same checkpoint" in err, err.strip().splitlines()[:2])

    # 5. checkpoint noise diagnostic
    st = put("wide_0_s140", "stock_s2", "wide")
    res, code, out, err = run(base, {**good, "wide_0": put("wide_0n", "stock", "wide")}, extra=[f"wide_0_step140={st}"])
    sp = res["checkpoint_noise_diagnostic"]["runs"]["wide_0"]["spread_max_minus_min"]
    want = abs(RAW["stock"]["summary"]["norm_auc_log2"] - RAW["stock_s2"]["summary"]["norm_auc_log2"])
    check("noise: spread equals the hand-computed area difference", code == 0 and abs(sp - want) < 1e-12, [sp, want])

    # 6. answer tokens
    check("answer tokens: unavailable when rows carry none and no tokenizer",
          not res["baseline"]["answer_tokens"]["available"] and not res["arms"]["wide"]["answer_tokens"]["available"],
          res["baseline"]["answer_tokens"])

    def add6(d):
        for r in d["rows"]:
            for k, v in r["caps"].items():
                v.append(int(k) // 16)
    b6 = put("base6", "stock", "stock", add6)
    res, code, out, err = run(b6, good)
    at = res["baseline"]["answer_tokens"]
    check("answer tokens: read from rows when they carry them",
          code == 0 and at["available"] and at["source"] == "rows" and at["per_cap"]["8192"]["mean"] == 512, at["per_cap"]["8192"])
    try:
        import transformers  # noqa: F401
        res, code, out, err = run(base, good, tok=TOK)
        at = res["arms"]["ctrl"]["answer_tokens"]
        check("answer tokens: re-tokenized from the traces sidecar",
              code == 0 and at["available"] and at["per_cap"]["8192"]["mean"]["seed_mean"] > 0,
              {"source": at.get("source"), "8192 mean": at["per_cap"]["8192"]["mean"]["seed_mean"]})
    except ImportError:
        check("answer tokens: sidecar route skipped (transformers not importable here)", True, "skipped")

    # 7. MATH-500 unseen check
    man = json.loads(S.DEFAULT_MANIFEST.read_text())["pilot50_math500_ids"]
    ids = sorted({r["id"] for r in RAW["stock"]["rows"]})
    ren = {old: (man[j] if j < 25 else f"math500:fake/{j}.json") for j, old in enumerate(ids)}

    msha = json.loads(S.DEFAULT_MANIFEST.read_text())["files"]["test_math500.jsonl"]

    def to_math(d):
        for r in d["rows"]:
            r["id"] = ren[r["id"]]
        d["env"]["sha256"]["data"] = msha
    mslots = {k: put(f"m_{k}", f, k.split("_")[0], to_math)
              for k, f in zip(good, ["stock", "stock_s2", "stock", "noclose", "half", "noclose", "stock_s2", "stock", "stock_s2"])}
    mwarm = put("m_warm", "stock_s2", "ctrl", lambda d: (to_math(d), warmck(d)))
    res, code, out, err = run(put("m_base", "stock", "stock", to_math), mslots, split="test", warm=mwarm)
    sub = res["math500_unseen450"] if res else {}
    check("math500 (test split): unseen subset excludes the pilot items, warm-up contrasts included",
          code == 0 and sub["n_items"] == 25 and sub["n_pilot_excluded"] == 25
          and sub["effects"]["wide - ctrl"]["seed_mean_diff"] < 0 and "wide - warmup" in sub["effects"]
          and "warmup" in sub["norm_auc_log2"] and res["split"] == "test" and res["test_set"] == ["math500"],
          {"n": sub.get("n_items"), "wide - ctrl": sub["effects"]["wide - ctrl"]["seed_mean_diff"]} if sub else err[-300:])
    res, code, out, err = run(put("m_base", "stock", "stock", to_math), mslots, split="val", warm=mwarm)
    check("refuse: --split val on data with a test sha256 (and test ids)",
          code == 2 and "not the validation slice" in err and "test ids in a validation analysis" in err,
          err.strip().splitlines()[:3])

    # 8. determinism
    r1, _, _, _ = run(base, good)
    r2, _, _, _ = run(base, good)
    check("determinism: identical intervals across runs",
          all(r1["effects"][k]["ci95"] == r2["effects"][k]["ci95"] for k in r1["effects"]), r1["effects"]["wide - ctrl"]["ci95"])

    # 9. warm-up role
    warm = put("warm", "stock", "ctrl", warmck)
    res, code, out, err = run(base, slots_of(["stock"] * 3, ["stock"] * 3, ["stock"] * 3), warm=warm)
    e = res["effects"] if res else {}
    wk = [f"{x} - warmup" for x in S.ARMS]
    check("warm-up: identical inputs give arm - warm-up = 0 with no effect; warm-up - untouched descriptive",
          code == 0 and all(e[k]["seed_mean_diff"] == 0 and not e[k]["effect"] for k in wk)
          and e["warmup - untouched"]["rule"] == "descriptive" and e["warmup - untouched"]["effect"] is None
          and res["warmup"]["arm"] == "ctrl" and "warm-up" in out,
          {k: [e[k]["seed_mean_diff"], e[k]["verdict"]] for k in wk + ["warmup - untouched"]} if e else err[-300:])
    rslots = slots_of(["stock", "stock_s2", "stock"], ["noclose", "half", "noclose"], ["stock_s2", "stock", "stock_s2"])
    res, code, out, err = run(base, rslots, warm=warm)
    w = res["effects"]["wide - warmup"]
    check("warm-up: red wide arms - stock warm-up is a negative effect",
          code == 0 and w["effect"] and w["verdict"] == "effect, negative" and w["seed_mean_diff"] < -0.2,
          {k: w[k] for k in ("per_seed_diff", "seed_mean_diff", "ci95", "verdict")})
    rwarm = put("warm_red", "noclose", "ctrl", warmck)
    res, code, out, err = run(base, rslots, warm=rwarm)
    w = res["effects"]["warmup - untouched"]
    check("warm-up: a red warm-up vs untouched is reported descriptively (interval excludes 0, no verdict)",
          code == 0 and w["ci_excludes_0"] and w["seed_mean_diff"] < -0.2 and w["effect"] is None
          and w["verdict"] == S.DESCRIPTIVE and res["effects"]["wide - warmup"]["seed_mean_diff"] > -0.1,
          {k: w[k] for k in ("seed_mean_diff", "ci95", "verdict")})
    _, code, _, err = run(base, rslots, warm=put("warm_stockarm", "stock", "stock", warmck))
    check("refuse: warm-up with arm stock", code == 2 and "warm-up role needs arm 'ctrl'" in err, err.strip().splitlines()[:2])
    _, code, _, err = run(base, rslots, warm=put("warm_nock", "stock", "ctrl"))
    check("refuse: warm-up without a checkpoint", code == 2 and "warm-up role needs arm 'ctrl'" in err,
          err.strip().splitlines()[:2])
    _, code, _, err = run(base, {**rslots, "ctrl_1": put("ctrl_1w", "stock_s2", "ctrl", warmck)}, warm=warm)
    check("refuse: the warm-up checkpoint in an arm slot", code == 2 and "ctrl_1 and warmup: same checkpoint" in err,
          err.strip().splitlines()[:2])
    _, code, _, err = run(warm, rslots)
    check("refuse: the warm-up given as --baseline", code == 2 and "baseline: arm 'ctrl', need 'stock'" in err,
          err.strip().splitlines()[:2])

    # 10. split labels
    res, code, out, err = run(base, rslots, warm=warm)
    check("split: --split val is labeled step 12 / validation",
          code == 0 and res["split"] == "val" and res["step"] == 12 and res["split_label"] == "validation"
          and "test_set" not in res and "step 12, VALIDATION split" in out and "validation receipt:" in out
          and res["receipt"]["split"] == "val" and res["receipt"]["inputs"]["warmup"]["role"] == "warmup",
          out.strip().splitlines()[0])
    _, code, _, err = run(base, rslots, warm=warm, split="test")
    check("refuse: --split test on validation data", code == 2 and "it is the validation slice: run with --split val" in err,
          err.strip().splitlines()[:2])

    # 11. partial
    part = {k: rslots[k] for k in ("ctrl_0", "ctrl_1", "wide_0", "wide_1", "wide1_0")}
    _, code, _, err = run(base, part, warm=warm)
    check("refuse: missing slots on validation without --allow_partial", code == 2 and "wide1_2: missing" in err,
          err.strip().splitlines()[:2])
    _, code, _, err = run(base, rslots, warm=warm, split="test", partial=True)
    check("refuse: --allow_partial on the test split", code == 2 and "validation split only" in err,
          err.strip().splitlines()[:2])
    res, code, out, err = run(base, part, warm=warm, partial=True)
    e = res["effects"] if res else {}
    wc = e.get("wide - ctrl", {})
    check("partial: 5 of 9 runs, contrasts on the seeds present, under 3 seeds no verdict although the interval "
          "excludes 0", code == 0 and res["partial"] and res["n_arm_runs"] == 5 and wc["seeds"] == [0, 1]
          and wc["ci_excludes_0"] and not wc["effect"] and wc["verdict"].startswith("incomplete (2 of 3 seeds)")
          and e["wide - wide1"]["seeds"] == [0] and e["wide1 - warmup"]["seeds"] == [0]
          and set(res["arms"]) == {"ctrl", "wide", "wide1"} and "PARTIAL: 5 of 9 arm runs present" in out,
          {k: [v["seeds"], v["verdict"]] for k, v in e.items()} if e else err[-300:])
    part2 = {k: rslots[k] for k in ("ctrl_0", "wide_0")}
    res, code, out, err = run(base, part2, warm=warm, partial=True)
    check("partial: an arm with no run is absent and its contrasts are listed as skipped",
          code == 0 and "wide1" not in res["arms"] and "wide1 - ctrl: no seed present in both arms" in res["contrasts_skipped"]
          and "wide1 absent" in " ".join(out.split()), res["contrasts_skipped"] if res else err[-300:])
    _, code, _, err = run(base, part, warm=warm, partial=True, extra=[f"wide1_2_step140={rslots['wide1_2']}"])
    check("refuse: diagnostic readout without its final slot", code == 2 and "without the final readout wide1_2" in err,
          err.strip().splitlines()[:2])
    res, code, out, err = run(base, rslots, warm=warm, partial=True)
    check("partial flag with all 9 runs present: complete", code == 0 and res["complete"] and not res["partial"]
          and res["effects"]["wide - ctrl"]["verdict"] == "effect, negative", res["n_arm_runs"] if res else err[-300:])

    # 12. one device
    def other_gpu(d):
        d["env"]["gpu"] = "NVIDIA GB10"
    _, code, _, err = run(base, {**rslots, "wide_2": put("gpu_other", "noclose", "wide", other_gpu)}, warm=warm)
    check("refuse: a readout from another GPU", code == 2 and "wide_2: GPU 'NVIDIA GB10' differs" in err,
          err.strip().splitlines()[:2])

    # 13. seed set (--seeds): an extension set and the pooled set under the same rule
    def slots_seeded(ctrl, wide, wide1, seeds):
        sl = {}
        for arm, files in (("ctrl", ctrl), ("wide", wide), ("wide1", wide1)):
            for sd, f in zip(seeds, files):
                sl[f"{arm}_{sd}"] = put(f"{arm}_{sd}", f, arm)
        return sl
    pat = (["stock", "stock_s2", "stock"], ["noclose", "half", "noclose"], ["stock_s2", "stock", "stock_s2"])
    keys = ("per_seed_diff", "seed_mean_diff", "ci95", "verdict", "rule")
    r0, c0, _, e0 = run(base, slots_seeded(*pat, (0, 1, 2)), warm=warm)
    r3, c3, _, e3 = run(base, slots_seeded(*pat, (3, 4, 5)), warm=warm, extra=["--seeds", "3,4,5"])
    same = c0 == 0 and c3 == 0 and set(r0["effects"]) == set(r3["effects"]) and all(
        r0["effects"][k][x] == r3["effects"][k][x] for k in r0["effects"] for x in keys)
    check("seeds 3,4,5: the same files as an extension set give the numbers and verdicts of seeds 0,1,2",
          same and r3["effects"]["wide - ctrl"]["seeds"] == [3, 4, 5] and r3["arms_present"]["wide"] == [3, 4, 5]
          and r3["effects"]["wide - ctrl"]["rule"] == "3/3 seeds" and r3["complete"],
          {k: r3["effects"][k]["verdict"] for k in ("wide - ctrl", "wide1 - ctrl", "wide - wide1")} if r3 else e3[-300:])
    _, c, _, err = run(base, slots_seeded(*pat, (0, 1, 2)), warm=warm, extra=["--seeds", "3,4,5"])
    check("refuse: seed-0 labels read with --seeds 3,4,5", c == 2 and "ctrl_0: seed 0 outside (3, 4, 5)" in err
          and "ctrl_3: missing" in err, err.strip().splitlines()[:3])
    pooled = {**slots_seeded(*pat, (0, 1, 2)), **slots_seeded(["stock"] * 3, ["stock"] * 3, ["stock"] * 3, (3, 4, 5))}
    rp, cp, _, ep = run(base, pooled, warm=warm, extra=["--seeds", "0,1,2,3,4,5"])
    wc = rp["effects"]["wide - ctrl"] if rp else {}
    check("seeds 0-5 pooled: 6/6 rule; three negative and three zero seeds are no effect although the interval "
          "excludes 0", cp == 0 and wc["rule"] == "6/6 seeds" and wc["n_seeds"] == 6 and wc["ci_excludes_0"]
          and not wc["effect"] and wc["verdict"] == "no effect" and rp["n_arm_runs_planned"] == 18 and rp["complete"]
          and "mean of 6 seeds" in rp["primary_metric"], {k: wc.get(k) for k in keys} if wc else ep[-300:])
    rd, cd, _, ed = run(base, slots_seeded(*pat, (0, 1, 2)), warm=warm)
    check("default after --seeds calls: the 3/3 rule again", cd == 0 and rd["effects"]["wide - ctrl"]["rule"] == "3/3 seeds"
          and rd["n_arm_runs_planned"] == 9 and "the 3-seed rule" in S.DESCRIPTIVE
          and rd["effects"]["warmup - untouched"]["verdict"] == S.DESCRIPTIVE, rd["n_arm_runs_planned"] if rd else ed[-300:])

    # 14. natural length under the rule (length_effects) and the pre-registered hypotheses H1 / H2. The parity files
    # differ little in length, so the length red arm is built: "short" = the stock readout with every think_len halved.
    def shorten(d):
        for r in d["rows"]:
            r["think_len"] = r["think_len"] // 2

    def build(tag, ctrl, wide, wide1, also=None):
        out = {}
        for arm, files in (("ctrl", ctrl), ("wide", wide), ("wide1", wide1)):
            for i, f in enumerate(files):
                src, short = (f[:-6], True) if f.endswith("_short") else (f, False)
                out[f"{arm}_{i}"] = put(f"{tag}_{arm}_{i}", src, arm,
                                        lambda d, s=short: ((shorten(d) if s else None), (also(d) if also else None)))
        return out
    res, code, _, err = run(base, build("li", ["stock"] * 3, ["stock"] * 3, ["stock"] * 3), warm=warm)
    le = res["length_effects"] if res else {}
    check("length: identical inputs give 0 on every length contrast, no effect",
          code == 0 and le and all(v["seed_mean_diff"] == 0 and not v["effect"] for v in le.values()),
          {k: v["seed_mean_diff"] for k, v in le.items()} if le else err[-300:])
    sl = build("ls", ["stock", "stock_s2", "stock"], ["stock_short", "stock_s2_short", "stock_short"], ["stock_s2", "stock", "stock_s2"])
    res, code, _, err = run(base, sl)
    w = res["length_effects"]["wide - ctrl"] if res else {}
    nat = (res["arms"]["wide"]["natural_len"]["seed_mean"] - res["arms"]["ctrl"]["natural_len"]["seed_mean"]) if res else None
    check("length red: wide = halved reasoning reads shorter than ctrl: negative effect, and the item-mean difference "
          "equals the arms' natural_len difference",
          code == 0 and w["effect"] and w["verdict"] == "effect, negative" and w["seed_mean_diff"] < -1000
          and abs(w["seed_mean_diff"] - nat) < 1e-6, {k: w.get(k) for k in ("per_seed_diff", "seed_mean_diff", "ci95", "verdict")})
    ids = sorted({r["id"] for r in RAW["stock"]["rows"]})
    da = [S.item_natural(json.loads(Path(sl[f"wide_{s}"]).read_text()), ids) -
          S.item_natural(json.loads(Path(sl[f"ctrl_{s}"]).read_text()), ids) for s in range(3)]
    ci = S.boot_ci(sum(da) / 3, S.boot_index(len(ids)))
    check("length: the interval comes from the shared index matrix (recomputed by hand)", ci == w.get("ci95"), [ci, w.get("ci95")])
    ea = res["effects"]["wide - ctrl"] if res else {}
    check("length: the area rule is unchanged by the length block (halved think_len leaves accuracy as it was)",
          ea.get("seed_mean_diff") is not None and abs(ea["seed_mean_diff"]) < 0.02, ea.get("seed_mean_diff"))
    gsha = json.loads(S.DEFAULT_MANIFEST.read_text())["files"]["test_gsm8k.jsonl"]

    def to_gsm(d):
        for r in d["rows"]:
            r["id"] = "gsm8k_test:" + r["id"].split(":", 1)[1]
        d["env"]["sha256"]["data"] = gsha
    gbase = put("g_base", "stock", "stock", to_gsm)
    gwarm = put("g_warm", "stock_s2", "ctrl", lambda d: (to_gsm(d), warmck(d)))
    res, code, _, err = run(gbase, build("ga", ["stock", "stock_s2", "stock"], ["stock_short", "stock_s2_short", "stock_short"],
                                         ["stock"] * 3, to_gsm), split="test", warm=gwarm)
    h = res["preregistered_hypotheses"] if res else {}
    check("H2 (GSM8K test): wide shorter than ctrl on 3/3 seeds with the interval below 0 -> supported; H1 not "
          "applicable on GSM8K", code == 0 and h["H2"]["applies"] and h["H2"]["supported"] and h["H2"]["verdict"] == "supported"
          and h["H2"]["seed_mean_diff"] < 0 and not h["H1"]["applies"] and "not applicable" in h["H1"]["verdict"],
          {k: (v.get("verdict"), v.get("seed_mean_diff")) for k, v in h.items()} if h else err[-300:])
    res, code, _, err = run(gbase, build("gb", ["stock_short", "stock_s2_short", "stock_short"], ["stock", "stock_s2", "stock"],
                                         ["stock"] * 3, to_gsm), split="test", warm=gwarm)
    h2 = res["preregistered_hypotheses"]["H2"] if res else {}
    check("H2 red arm: wide longer than ctrl (an effect in the other direction) -> not supported",
          code == 0 and h2["applies"] and not h2["supported"] and h2["verdict"] == "rule effect in the other direction",
          {k: h2.get(k) for k in ("verdict", "seed_mean_diff", "ci95")})
    res, code, _, err = run(gbase, build("gc", ["stock", "stock_s2", "stock_short"], ["stock_short", "stock_s2_short", "stock"],
                                         ["stock"] * 3, to_gsm), split="test", warm=gwarm)
    h2 = res["preregistered_hypotheses"]["H2"] if res else {}
    check("H2 red arm: seed 2 reversed (signs disagree) -> not supported (no effect by the rule)",
          code == 0 and not h2["supported"] and h2["verdict"].startswith("not supported") and h2["seed_mean_diff"] < 0
          and h2["per_seed_diff"][2] > 0, {k: h2.get(k) for k in ("verdict", "per_seed_diff", "seed_mean_diff")})
    res, code, _, err = run(put("m_base", "stock", "stock", to_math),
                            {k: put(f"mh_{k}", f, k.split("_")[0], to_math) for k, f in
                             zip(good, ["noclose", "noclose", "noclose", "stock", "stock_s2", "stock", "stock", "stock", "stock"])},
                            split="test", warm=mwarm)
    h = res["preregistered_hypotheses"] if res else {}
    check("H1 (MATH-500 test): wide area above ctrl on 3/3 seeds -> supported; H2 not applicable on MATH-500",
          code == 0 and h["H1"]["supported"] and h["H1"]["seed_mean_diff"] > 0 and not h["H2"]["applies"],
          {k: (v.get("verdict"), v.get("seed_mean_diff")) for k, v in h.items()} if h else err[-300:])
    res, code, _, err = run(put("m_base", "stock", "stock", to_math), mslots, split="test", warm=mwarm)
    h1 = res["preregistered_hypotheses"]["H1"] if res else {}
    check("H1 red arm: wide = red readouts (area below ctrl) -> not supported",
          code == 0 and h1["applies"] and not h1["supported"] and h1["seed_mean_diff"] < 0,
          {k: h1.get(k) for k in ("verdict", "seed_mean_diff")})

    ok = all(r["pass"] for r in RESULTS)
    Path(a.out).write_text(json.dumps({"pass": ok, "checks": RESULTS}, indent=1, default=str))
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}: {sum(r['pass'] for r in RESULTS)}/{len(RESULTS)}; receipt {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
