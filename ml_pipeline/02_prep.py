import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    # Steps 4-8 of the ml-pipeline for step 7 (length-cost GRPO), run after the owner's first gate.
    # Reads the raw sets and the step 1 manifest read-only; writes the frozen split files, their hashes, the guard
    # state (test fingerprint) and the step 4/6 figures.
    import hashlib
    import json
    import sys
    import types
    from difflib import SequenceMatcher
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ML = Path(__file__).resolve().parent
    ROOT = ML.parent
    sys.path.insert(0, str(ML))
    import guard  # noqa: E402

    BROAD = ROOT / "src/gdn_gate/broad"
    OUT = Path(__import__("os").environ.get("WG_RUNS", str(ROOT / "runs"))) / "ml7/data"
    NEAR = 0.85          # sibling line, same as step 1
    TOPK = 20            # closest train items (by n-gram cosine) checked per held-out item
    VAL_N = 500          # contract
    VAL_SALT = bytes.fromhex("6e65676569672d6d6c372d76616c3a").decode()  # the frozen split's salt, kept byte-exact (data/splits/ lists the ids)
    return (BROAD, ML, NEAR, OUT, ROOT, SequenceMatcher, TOPK, VAL_N, VAL_SALT, guard, hashlib, json, np, pd, sys,
            types)


@app.cell
def _(BROAD, sys, types):
    # rl_common imports torch at module level for its generation helpers; the grader (boxed, correct) is pure Python.
    # A stub torch (no_grad as an identity decorator) lets the exact grader run in this CPU venv without torch.
    if "torch" not in sys.modules:
        _t = types.ModuleType("torch")
        _t.no_grad = lambda: (lambda f: f)
        sys.modules["torch"] = _t
    sys.path.insert(0, str(BROAD))
    import rl_common as R  # noqa: E402

    return (R,)


@app.cell
def _(ML, pd):
    manifest = pd.read_parquet(ML / "manifest.parquet")
    pool_all = manifest[manifest["set"] == "gsm8k_train"].reset_index(drop=True)
    heldout = manifest[manifest["set"] != "gsm8k_train"].reset_index(drop=True)
    return heldout, manifest, pool_all


@app.cell
def _(np):
    import re as _re
    import zlib as _zlib

    def norm(s, mask_numbers=False):
        s = s.lower()
        s = _re.sub(r"(?<=\d),(?=\d{3})", "", s)
        if mask_numbers:
            s = _re.sub(r"\d+(\.\d+)?", "#", s)
        s = _re.sub(r"[^a-z0-9#\\]+", " ", s)
        return _re.sub(r"\s+", " ", s).strip()

    def hashed_ngrams(texts, dim=1 << 20):
        """Unit-norm counts of hashed word 1-3 grams (no statistics learned from any set)."""
        from scipy.sparse import csr_matrix

        indptr, idx, val = [0], [], []
        for t in texts:
            w = t.split()
            grams = w + [" ".join(w[i:i + 2]) for i in range(len(w) - 1)] + [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
            counts = {}
            for g in grams:
                h = _zlib.crc32(g.encode()) % dim
                counts[h] = counts.get(h, 0) + 1
            ks = np.fromiter(counts.keys(), dtype=np.int64)
            vs = np.fromiter(counts.values(), dtype=np.float64)
            vs = vs / (np.linalg.norm(vs) or 1.0)
            idx.extend(ks.tolist())
            val.extend(vs.tolist())
            indptr.append(len(idx))
        return csr_matrix((val, idx, indptr), shape=(len(texts), dim))

    return hashed_ngrams, norm


@app.cell
def _(NEAR, SequenceMatcher, TOPK, hashed_ngrams, norm, np, pd):
    def sibling_scan(left, right):
        """Every (left, right) pair among each left item's TOPK closest right items (cosine on numbers kept and on
        numbers masked) whose character match ratio is at least NEAR with numbers kept or with numbers masked."""
        ln = left["question"].map(norm).tolist()
        lm = left["question"].map(lambda s: norm(s, True)).tolist()
        rn = right["question"].map(norm).tolist()
        rm = right["question"].map(lambda s: norm(s, True)).tolist()
        cos = (hashed_ngrams(ln) @ hashed_ngrams(rn).T).toarray()
        cosm = (hashed_ngrams(lm) @ hashed_ngrams(rm).T).toarray()
        top = np.argsort(-cos, axis=1)[:, :TOPK]
        topm = np.argsort(-cosm, axis=1)[:, :TOPK]
        out = []
        for i in range(len(left)):
            for j in set(top[i].tolist()) | set(topm[i].tolist()):
                r = SequenceMatcher(None, ln[i], rn[j], autojunk=False).ratio()
                rmk = SequenceMatcher(None, lm[i], rm[j], autojunk=False).ratio()
                if max(r, rmk) >= NEAR:
                    out.append({"left_id": left.loc[i, "id"], "right_id": right.loc[j, "id"], "ratio": round(r, 4),
                                "ratio_masked": round(rmk, 4),
                                "same_answer": norm(left.loc[i, "answer"]) == norm(right.loc[j, "answer"])})
        return pd.DataFrame(out, columns=["left_id", "right_id", "ratio", "ratio_masked", "same_answer"])

    return (sibling_scan,)


@app.cell
def _(heldout, pool_all, sibling_scan):
    # Rule 4.1: drop from the RL pool every GSM8K train item that is a near-duplicate or template sibling of any
    # held-out item (GSM8K test or MATH-500).
    test_sibs = sibling_scan(heldout, pool_all)
    flagged_test = set(test_sibs["right_id"])
    return flagged_test, test_sibs


@app.cell
def _(VAL_N, VAL_SALT, flagged_test, hashlib, pool_all):
    # Step 6 rule (computed here because rule 4.2 depends on it): the validation slice is the VAL_N GSM8K train items
    # with the smallest sha256(VAL_SALT + id), chosen among items not flagged by rule 4.1.
    _cand = pool_all[~pool_all["id"].isin(flagged_test)].copy()
    _cand["h"] = _cand["id"].map(lambda i: hashlib.sha256((VAL_SALT + i).encode()).hexdigest())
    val_ids = set(_cand.sort_values("h").head(VAL_N)["id"])
    return (val_ids,)


@app.cell
def _(flagged_test, pool_all, sibling_scan, val_ids):
    # Rule 4.2: drop from the RL pool every remaining train item that is a sibling of a validation item, so the
    # validation curve during training is not lifted by trained-on templates.
    _val = pool_all[pool_all["id"].isin(val_ids)].reset_index(drop=True)
    _rest = pool_all[~pool_all["id"].isin(val_ids | flagged_test)].reset_index(drop=True)
    val_sibs = sibling_scan(_val, _rest)
    flagged_val = set(val_sibs["right_id"])
    return flagged_val, val_sibs


@app.cell
def _(R, manifest, np, pd):
    # Rule 4.3: grader self-check. rl_common.correct is run on gold-derived completions for every item, twice:
    # with math_verify importable, and with it blocked (the string fallback rl_common uses when the import fails,
    # which is what runs today on every venv the step 7 code uses; see PIPELINE.md step 4).
    # Accept arms (A*) must pass, red arms (R*) must fail.
    import sys as _sys

    def _variants(gold, kind, other):
        v = {"A0_verbatim": f"So the answer is \\boxed{{{gold}}}.",
             "A1_spaced": f"\\boxed{{ {gold} }}",
             "R1_wrong_value": f"\\boxed{{{other}}}",
             "R2_no_box": f"So the answer is {gold}.",
             "R3_empty_box": "\\boxed{}",
             "R4_last_box_wrong": f"First \\boxed{{{gold}}}, but on reflection \\boxed{{{other}}}."}
        if kind == "integer" and gold.lstrip("-").isdigit():
            n = int(gold)
            v["A2_trailing_zeros"] = f"\\boxed{{{gold}.00}}"
            v["A3_dollar"] = f"\\boxed{{\\${gold}}}"
            v["A4_unit_text"] = f"\\boxed{{{gold} \\text{{ dollars}}}}"
            if abs(n) >= 1000:
                v["A5_thousands_comma"] = f"\\boxed{{{n:,}}}"
        if "\\frac" in gold:
            v["A6_dfrac"] = f"\\boxed{{{gold.replace(chr(92) + 'frac', chr(92) + 'dfrac')}}}"
        return v

    _rows = manifest[["id", "set", "answer", "answer_type"]].to_dict("records")
    _by_set = {}
    for _r in _rows:
        _by_set.setdefault(_r["set"], []).append(_r["answer"])
    _rng = np.random.default_rng(0)
    jobs = []
    for _r in _rows:
        g = _r["answer"]
        if _r["answer_type"] == "integer" and g.lstrip("-").isdigit():
            other = str(int(g) + 1)
        else:
            _pool = [a for a in _by_set[_r["set"]] if a != g]
            other = _pool[int(_rng.integers(len(_pool)))]
        for name, text in _variants(g, _r["answer_type"], other).items():
            jobs.append((_r["id"], _r["set"], _r["answer_type"], name, text, g))

    def grade_all(block_math_verify):
        saved = _sys.modules.get("math_verify", "absent")
        if block_math_verify:
            _sys.modules["math_verify"] = None  # any `from math_verify import ...` now raises ImportError
        try:
            return [bool(R.correct(j[4], j[5])) for j in jobs]
        finally:
            if saved == "absent":
                _sys.modules.pop("math_verify", None)
            else:
                _sys.modules["math_verify"] = saved

    _mv = grade_all(False)
    _fb = grade_all(True)
    grader = pd.DataFrame(jobs, columns=["id", "set", "answer_type", "variant", "text", "gold"])
    grader["math_verify"] = _mv
    grader["fallback"] = _fb
    return (grader,)


@app.cell
def _(ML, grader, json):
    _g = grader.assign(arm=grader["variant"].str[0].map({"A": "accept", "R": "red"}))
    grader_table = (_g.groupby(["set", "variant"])[["math_verify", "fallback"]].mean().round(4)
                    .join(_g.groupby(["set", "variant"]).size().rename("n")).reset_index())
    _cols = ["id", "variant", "text", "gold"]
    grader_check = {
        "verbatim_pass_math_verify": float(_g.loc[_g["variant"] == "A0_verbatim", "math_verify"].mean()),
        "verbatim_pass_fallback": float(_g.loc[_g["variant"] == "A0_verbatim", "fallback"].mean()),
        "red_accept_math_verify": float(_g.loc[_g["arm"] == "red", "math_verify"].mean()),
        "red_accept_fallback": float(_g.loc[_g["arm"] == "red", "fallback"].mean()),
        "accept_variants_math_verify": float(_g.loc[_g["arm"] == "accept", "math_verify"].mean()),
        "accept_variants_fallback": float(_g.loc[_g["arm"] == "accept", "fallback"].mean()),
        "n_checks": int(len(_g)),
        "red_failures_math_verify": _g[(_g["arm"] == "red") & _g["math_verify"]][_cols].head(40).to_dict("records"),
        "red_failures_fallback": _g[(_g["arm"] == "red") & _g["fallback"]][_cols].head(40).to_dict("records"),
        "accept_misses_math_verify": _g[(_g["arm"] == "accept") & ~_g["math_verify"]][_cols].head(60).to_dict("records"),
        "table": grader_table.to_dict("records"),
    }
    grader_check["math_verify_passes_contract"] = bool(grader_check["verbatim_pass_math_verify"] == 1.0
                                                       and grader_check["red_accept_math_verify"] == 0.0)
    (ML / "grader_check.json").write_text(json.dumps(grader_check, indent=1, ensure_ascii=False))
    return grader_check, grader_table


@app.cell
def _(BROAD, OUT, R, ROOT, flagged_test, flagged_val, hashlib, heldout, json, pd, pool_all, val_ids):
    # Step 5 (data engineering): one JSONL per split in the field layout the step 7 code reads (problem, answer;
    # rl_common.gold_of reads `answer`, Thinker.prompt reads `problem`).
    def _rec(row):
        r = {"id": row["id"], "problem": row["question"], "answer": row["answer"], "set": row["set"]}
        if row["set"] == "math500":
            r["level"] = row["level"]
            r["subject"] = row["subject"]
        else:
            r["level"] = "gsm8k"
            # compat: train_grpo.pool_rows filters on `level` and requires a boxed `solution`; gold_of reads `answer`
            r["solution"] = "\\boxed{" + row["answer"] + "}"
        return r

    dropped = flagged_test | flagged_val
    _pool = pool_all[~pool_all["id"].isin(val_ids | dropped)].reset_index(drop=True)
    _val = pool_all[pool_all["id"].isin(val_ids)].reset_index(drop=True)
    _hg = heldout[heldout["set"] == "gsm8k_test"].reset_index(drop=True)
    _hm = heldout[heldout["set"] == "math500"].reset_index(drop=True)
    files = {"rl_pool.jsonl": _pool, "val.jsonl": _val, "test_gsm8k.jsonl": _hg, "test_math500.jsonl": _hm}
    OUT.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "created": "2026-10-08", "out_dir": str(OUT), "files": {}, "counts": {},
        "rules": {"val": "500 smallest sha256(VAL_SALT + id) (VAL_SALT in ml_pipeline/02_prep.py) among GSM8K train items not dropped by rule 4.1",
                  "drop_4_1": "GSM8K train items that are near-duplicates or template siblings of a held-out item",
                  "drop_4_2": "GSM8K train items that are siblings of a validation item"},
        "dropped_4_1": sorted(flagged_test), "dropped_4_2": sorted(flagged_val - flagged_test),
        "pilot50_math500_ids": sorted(_hm.loc[_hm["in_pilot50"], "id"]),
        "code_sha256": {"rl_common.py": hashlib.sha256((BROAD / "rl_common.py").read_bytes()).hexdigest(),
                        "ml_pipeline/02_prep.py": hashlib.sha256((ROOT / "ml_pipeline/02_prep.py").read_bytes()).hexdigest()}}
    for _name, _df in files.items():
        _p = OUT / _name
        with open(_p, "w") as _f:
            for _, _row in _df.iterrows():
                _f.write(json.dumps(_rec(_row), ensure_ascii=False) + "\n")
        split_manifest["files"][_name] = hashlib.sha256(_p.read_bytes()).hexdigest()
        split_manifest["counts"][_name] = int(len(_df))
    # data-quality checks: unique ids, every gold present, prompts non-empty, pool accounting adds up
    _all = pd.concat(files.values())
    assert _all["id"].is_unique
    assert (_all["answer"].str.len() > 0).all()
    assert len(_pool) + len(_val) + len(dropped) == len(pool_all), "pool accounting"
    assert len(_hg) == 1319 and len(_hm) == 500
    for _name in files:
        for _line in open(OUT / _name):
            _r = json.loads(_line)
            assert R.gold_of(_r) == _r["answer"] and _r["problem"].strip()
    return files, split_manifest


@app.cell
def _(ML, files, guard, json, pd, split_manifest):
    # Step 6: the guard's overlap check on the text that matters (problem, answer), then freeze the test set.
    # guard.split() draws its own random or stratified split, which does not fit a fixed external test set, so the
    # guard's own fingerprint and state format are used directly; guard.final_test() reads this state at step 14.
    from datetime import date as _date

    def _frame(df):
        return df[["id", "question", "answer"]].rename(columns={"question": "problem"}).reset_index(drop=True)

    _tr = _frame(files["rl_pool.jsonl"])
    _va = _frame(files["val.jsonl"])
    _te = _frame(pd.concat([files["test_gsm8k.jsonl"], files["test_math500.jsonl"]]))
    guard.check_no_overlap(_tr[["problem", "answer"]], _va[["problem", "answer"]], _te[["problem", "answer"]])
    guard.check_no_overlap(_tr[["problem"]], _va[["problem"]], _te[["problem"]])
    _state = {"test_fingerprint": guard._fingerprint(_te), "n_test": int(len(_te)), "test_touches": 0,
              "duplicates_dropped": 0, "split": "fixed_external_test+hashed_validation",
              "created": _date.today().isoformat(),
              "test_frame": "columns id, problem, answer; rows of test_gsm8k.jsonl then test_math500.jsonl, file order"}
    (ML / guard.STATE_FILE).write_text(json.dumps(_state, indent=2))
    split_manifest["test_fingerprint"] = _state["test_fingerprint"]
    (ML / "split_manifest.json").write_text(json.dumps(split_manifest, indent=1))
    return


@app.cell
def _(ML, grader_check, grader_table, guard, np):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _w = grader_table.assign(mv=grader_table["math_verify"] * grader_table["n"], fb=grader_table["fallback"] * grader_table["n"])
    _agg = _w.groupby("variant")[["mv", "fb", "n"]].sum()
    _agg["math_verify"] = _agg["mv"] / _agg["n"]
    _agg["fallback"] = _agg["fb"] / _agg["n"]
    _agg = _agg.sort_index()
    _f, _ax = plt.subplots(figsize=(11, 4.2))
    _x = np.arange(len(_agg))
    _ax.bar(_x - 0.2, _agg["math_verify"], 0.4, label="math_verify (intended grader)", color="#4C72B0")
    _ax.bar(_x + 0.2, _agg["fallback"], 0.4, label="string fallback (what runs today)", color="#C44E52")
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"{v}\n(n={int(n)})" for v, n in zip(_agg.index, _agg["n"])], fontsize=7, rotation=25, ha="right")
    _ax.set_ylabel("share graded correct")
    _ax.set_ylim(0, 1.05)
    _ax.axvline(len([v for v in _agg.index if v.startswith("A")]) - 0.5, color="grey", ls=":")
    _ax.set_title("Grader self-check: A* = gold written differently (should pass), R* = wrong or unboxed (must fail)")
    _ax.legend(fontsize=8)
    _a = _agg["fallback"]
    _mv_min = _agg.loc[[v for v in _agg.index if v.startswith("A")], "math_verify"].min()
    guard.fig(4, "grader_selfcheck", _f,
              f"The grader run on answers built from the gold answer of every item ({grader_check['n_checks']:,} "
              f"checks). Gold written verbatim passes {grader_check['verbatim_pass_math_verify']:.1%} with math_verify "
              f"and {grader_check['verbatim_pass_fallback']:.1%} with the string fallback; the red arms (wrong value, no "
              f"box, empty box, a wrong last box) are accepted {grader_check['red_accept_math_verify']:.2%} and "
              f"{grader_check['red_accept_fallback']:.2%} of the time. The graders differ on harmless rewrites the "
              f"model often makes: the fallback accepts {_a.get('A2_trailing_zeros', float('nan')):.0%} of '18.00', "
              f"{_a.get('A3_dollar', float('nan')):.0%} of '$18', {_a.get('A4_unit_text', float('nan')):.0%} of "
              f"'18 dollars' and {_a.get('A5_thousands_comma', float('nan')):.0%} of '1,000', where math_verify accepts "
              f"at least {_mv_min:.0%} of every rewrite. A grader that marks right answers wrong lowers accuracy and "
              f"feeds false zero rewards into GRPO, so math_verify has to be installed wherever step 7 runs.", ML)
    return (plt,)


@app.cell
def _(ML, files, flagged_test, flagged_val, guard, plt, pool_all, test_sibs, val_sibs):
    _f, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _names = ["RL pool", "validation", "dropped: sibling of held-out", "dropped: sibling of validation",
              "GSM8K test", "MATH-500"]
    _n_val_only = len(flagged_val - flagged_test)
    _counts = [len(files["rl_pool.jsonl"]), len(files["val.jsonl"]), len(flagged_test), _n_val_only,
               len(files["test_gsm8k.jsonl"]), len(files["test_math500.jsonl"])]
    _cols = ["#4C72B0", "#8172B3", "#C44E52", "#DA8BC3", "#DD8452", "#55A868"]
    _ax[0].barh(_names[::-1], [max(c, 0.8) for c in _counts[::-1]], color=_cols[::-1])
    for _i, _c in enumerate(_counts[::-1]):
        _ax[0].text(max(_c, 0.8), _i, f" {_c:,}", va="center", fontsize=8)
    _ax[0].set_xscale("log")
    _ax[0].set_xlabel("items (log)")
    _ax[0].set_title("Frozen splits")
    for _lab, _key, _c in (("RL pool", "rl_pool.jsonl", "#4C72B0"), ("validation", "val.jsonl", "#8172B3"),
                           ("GSM8K test", "test_gsm8k.jsonl", "#DD8452")):
        _v = files[_key]["q_tokens"]
        _ax[1].hist(_v, bins=40, range=(0, 220), density=True, histtype="step", lw=2, color=_c,
                    label=f"{_lab} (median {int(_v.median())})")
    _ax[1].set_xlabel("question length, Qwen3.5 tokens")
    _ax[1].set_ylabel("density")
    _ax[1].legend(fontsize=8)
    _ax[1].set_title("Question length per split")
    _p = files["rl_pool.jsonl"]["q_tokens"].median()
    _v = files["val.jsonl"]["q_tokens"].median()
    guard.fig(6, "split_composition", _f,
              f"The frozen splits: {len(files['rl_pool.jsonl']):,} RL prompts and {len(files['val.jsonl'])} validation "
              f"items out of {len(pool_all):,} GSM8K train items; {len(flagged_test)} train items dropped as siblings "
              f"of a held-out question ({len(test_sibs)} pairs) and {_n_val_only} more as siblings of a validation "
              f"item ({len(val_sibs)} pairs). Question length matches across pool (median {int(_p)}), validation "
              f"(median {int(_v)}) and GSM8K test, so a cap chosen on validation carries over to the in-domain test.",
              ML)
    return


@app.cell
def _(ML, flagged_test, flagged_val, grader_check, json, split_manifest, test_sibs, val_sibs):
    prep_summary = {
        "sibling_pairs_heldout": int(len(test_sibs)), "dropped_4_1": int(len(flagged_test)),
        "heldout_items_with_sibling": int(test_sibs["left_id"].nunique()),
        "heldout_sibling_items_by_set": test_sibs.drop_duplicates("left_id")["left_id"].str.split(":").str[0].value_counts().to_dict(),
        "heldout_near_duplicates_same_answer": int(((test_sibs["ratio"] >= 0.85) & test_sibs["same_answer"]).sum()),
        "sibling_pairs_val": int(len(val_sibs)), "dropped_4_2": int(len(flagged_val - flagged_test)),
        "counts": split_manifest["counts"], "files_sha256": split_manifest["files"],
        "test_fingerprint": split_manifest["test_fingerprint"],
        "grader": {k: v for k, v in grader_check.items() if not isinstance(v, list)},
    }
    test_sibs.to_csv(ML / "siblings_heldout.csv", index=False)
    val_sibs.to_csv(ML / "siblings_val.csv", index=False)
    (ML / "step4_8_summary.json").write_text(json.dumps(prep_summary, indent=1, default=str))
    print(json.dumps(prep_summary, indent=1, default=str))
    return


if __name__ == "__main__":
    app.run()
