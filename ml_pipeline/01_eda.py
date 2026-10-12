import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    import math
    import re
    import sys
    import time
    from difflib import SequenceMatcher
    from pathlib import Path

    import numpy as np
    import pandas as pd

    ML = Path(__file__).resolve().parent
    ROOT = ML.parent
    sys.path.insert(0, str(ML))
    import guard  # noqa: E402

    RUNS = Path(__import__("os").environ.get("WG_RUNS", str(ROOT / "runs")))
    RL_PROMPTS = RUNS / "vllm/data/gsm8k_train.jsonl"
    EVAL_DIR = RUNS / "broad/eval_data"
    PILOT = RUNS / "broad/pilot7"
    TOKENIZER = Path(__import__("os").environ.get("WG_HF_CACHE", str(Path.home() / ".cache/huggingface/hub"))) / "models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
    SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."  # rl_common.SUFFIX
    return (EVAL_DIR, ML, PILOT, RL_PROMPTS, ROOT, SUFFIX, SequenceMatcher, TOKENIZER, guard, hashlib, json, math,
            np, pd, re, time)


@app.cell
def _(EVAL_DIR, RL_PROMPTS, hashlib, json):
    def _load(p):
        return [json.loads(line) for line in open(p)]

    def _sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    raw = {
        "gsm8k_train": _load(RL_PROMPTS),
        "gsm8k_test": _load(EVAL_DIR / "gsm8k.jsonl"),
        "math500": _load(EVAL_DIR / "math500.jsonl"),
    }
    frozen_manifest = json.loads(open(EVAL_DIR / "manifest.json").read())
    file_sha = {
        "gsm8k_train": _sha(RL_PROMPTS),
        "gsm8k_test": _sha(EVAL_DIR / "gsm8k.jsonl"),
        "math500": _sha(EVAL_DIR / "math500.jsonl"),
    }
    sha_matches_frozen = {
        "gsm8k_test": file_sha["gsm8k_test"] == frozen_manifest["files"]["gsm8k.jsonl"],
        "math500": file_sha["math500"] == frozen_manifest["files"]["math500.jsonl"],
    }
    return file_sha, raw, sha_matches_frozen


@app.cell
def _(TOKENIZER):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    return (tok,)


@app.cell
def _(re):
    _num_re = re.compile(r"^-?\d+(\.\d+)?$")
    _frac_re = re.compile(r"^(-?)\\[dt]?frac\{?(\d+)\}?\{?(\d+)\}?$")
    _slash_re = re.compile(r"^(-?\d+)/(\d+)$")

    def answer_kind(ans: str):
        """(answer type, numeric value or None) for a gold answer string."""
        a = re.sub(r"(?<=\d),(?=\d{3}\b)", "", ans.strip().replace("\\!", ""))
        a = a.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("$", "").strip()
        if _num_re.match(a):
            return ("integer" if "." not in a else "decimal"), float(a)
        m = _frac_re.match(a.replace(" ", ""))
        if m:
            v = float(m.group(2)) / float(m.group(3))
            return "fraction", -v if m.group(1) else v
        m = _slash_re.match(a.replace(" ", ""))
        if m:
            return "fraction", float(m.group(1)) / float(m.group(2))
        if "\\text" in a and not re.search(r"\d", a):
            return "text", None
        if (a.startswith(("(", "[", "\\left(", "\\left[")) and "," in a) or "\\cup" in a:
            return "tuple_or_interval", None
        return "expression", None

    return (answer_kind,)


@app.cell
def _(SUFFIX, answer_kind, math, pd, raw, tok):
    def _qtext(set_name, r):
        return r["question"] if set_name == "gsm8k_test" else r["problem"]

    def _ntok(s):
        return len(tok.encode(s, add_special_tokens=False))

    _template_cache = {}

    def _prompt_tokens(q):
        msgs = [{"role": "user", "content": q + SUFFIX}]
        return len(tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, tokenize=True,
                                           return_dict=False))

    pilot_ids = {r["id"] for r in raw["math500"][::10]}  # readout_len.py --stride 10 (pilot7)
    _src = {"gsm8k_train": "openai/gsm8k main train (frozen by the vllm stream)",
            "gsm8k_test": "openai/gsm8k main test (frozen by the broad stream)",
            "math500": "HuggingFaceH4/MATH-500 test (frozen by the broad stream)"}
    _role = {"gsm8k_train": "rl_prompts", "gsm8k_test": "heldout", "math500": "heldout"}
    _rows = []
    for _set, _items in raw.items():
        for _r in _items:
            _q = _qtext(_set, _r)
            _kind, _val = answer_kind(str(_r["answer"]))
            _rows.append({
                "id": f"{_set}:{_r['id']}",
                "set": _set,
                "source": _src[_set],
                "role": _role[_set],
                "level": str(_r.get("level", "gsm8k")) if _set == "math500" else "gsm8k",
                "subject": _r.get("subject", "gsm8k"),
                "question": _q,
                "q_chars": len(_q),
                "q_tokens": _ntok(_q),
                "prompt_tokens": _prompt_tokens(_q),
                "answer": str(_r["answer"]),
                "answer_type": _kind,
                "answer_tokens": _ntok(str(_r["answer"])),
                "answer_log10_abs": (math.log10(abs(_val)) if _val not in (None, 0.0) else (0.0 if _val == 0.0 else None)),
                "in_pilot50": _set == "math500" and _r["id"] in pilot_ids,
            })
    manifest = pd.DataFrame(_rows)
    manifest["answer_log10_abs"] = manifest["answer_log10_abs"].astype(float)
    return manifest, pilot_ids


@app.cell
def _(ML, manifest):
    manifest.to_parquet(ML / "manifest.parquet", index=False)
    manifest.drop(columns=["question"]).to_csv(ML / "manifest.csv", index=False)
    return


@app.cell
def _(ML, guard, manifest):
    # Profile the manifest with target = which set an item belongs to: the columns that tell the sets apart are the
    # distribution shift between the RL prompts and the held-out sets. Question text and answer text are left out of
    # the profile frame (unique free text); id is kept.
    profile_frame = manifest.drop(columns=["question", "answer", "source", "role"])
    data_profile = guard.profile(profile_frame, target="set", state_dir=ML)
    return data_profile, profile_frame


@app.cell
def _(manifest, np, re):
    def norm(s, mask_numbers=False):
        s = s.lower()
        s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
        if mask_numbers:
            s = re.sub(r"\d+(\.\d+)?", "#", s)
        s = re.sub(r"[^a-z0-9#\\]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def hashed_ngrams(texts, dim=1 << 20):
        """Unit-norm counts of hashed word 1-3 grams (no statistics learned from any set)."""
        from scipy.sparse import csr_matrix
        import zlib

        indptr, idx, val = [0], [], []
        for t in texts:
            w = t.split()
            grams = w + [" ".join(w[i:i + 2]) for i in range(len(w) - 1)] + [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
            counts = {}
            for g in grams:
                h = zlib.crc32(g.encode()) % dim
                counts[h] = counts.get(h, 0) + 1
            ks = np.fromiter(counts.keys(), dtype=np.int64)
            vs = np.fromiter(counts.values(), dtype=np.float64)
            vs = vs / (np.linalg.norm(vs) or 1.0)
            idx.extend(ks.tolist())
            val.extend(vs.tolist())
            indptr.append(len(idx))
        return csr_matrix((val, idx, indptr), shape=(len(texts), dim))

    sets = {s: manifest[manifest["set"] == s].reset_index(drop=True) for s in ("gsm8k_train", "gsm8k_test", "math500")}
    return hashed_ngrams, norm, sets


@app.cell
def _(SequenceMatcher, hashed_ngrams, norm, np, pd, sets):
    def _within_set_dupes(df):
        n = df["question"].map(norm)
        return {"exact_question_dupes": int(n.duplicated().sum()), "id_dupes": int(df["id"].duplicated().sum()),
                "exact_question_dupe_groups": int((n.value_counts() > 1).sum())}

    within = {s: _within_set_dupes(d) for s, d in sets.items()}

    _pool = sets["gsm8k_train"]
    _pool_norm = _pool["question"].map(norm).tolist()
    _pool_mask = _pool["question"].map(lambda s: norm(s, True)).tolist()
    _P = hashed_ngrams(_pool_norm)
    _PM = hashed_ngrams(_pool_mask)
    _pool_exact = set(_pool_norm)

    nearest = []
    for _s in ("gsm8k_test", "math500"):
        _held = sets[_s]
        _hn = _held["question"].map(norm).tolist()
        _hm = _held["question"].map(lambda s: norm(s, True)).tolist()
        _cos = (hashed_ngrams(_hn) @ _P.T).toarray()
        _cosm = (hashed_ngrams(_hm) @ _PM.T).toarray()
        _K = 10  # match ratio is computed against the 10 closest train items by n-gram cosine, the best one kept
        _top = np.argsort(-_cos, axis=1)[:, :_K]
        _topm = np.argsort(-_cosm, axis=1)[:, :_K]
        for _i in range(len(_held)):
            _rs = [(SequenceMatcher(None, _hn[_i], _pool_norm[_j], autojunk=False).ratio(), int(_j)) for _j in _top[_i]]
            _rm = [(SequenceMatcher(None, _hm[_i], _pool_mask[_j], autojunk=False).ratio(), int(_j)) for _j in _topm[_i]]
            _r, _j = max(_rs)
            _rmx, _jm = max(_rm)
            nearest.append({
                "set": _s, "id": _held.loc[_i, "id"],
                "exact_in_train": _hn[_i] in _pool_exact,
                "cos": float(_cos[_i].max()), "train_id": _pool.loc[_j, "id"], "ratio": _r,
                "same_answer": norm(_held.loc[_i, "answer"]) == norm(_pool.loc[_j, "answer"]),
                "train_id_masked": _pool.loc[_jm, "id"], "ratio_masked": _rmx,
            })
    nearest = pd.DataFrame(nearest)
    return nearest, within


@app.cell
def _(ML, json, manifest, nearest, within):
    # Definitions (match ratio = difflib character match of the normalized questions, 1.0 = identical):
    #   near-duplicate   ratio >= 0.85 AND the same gold answer: the same problem, reworded (direct leakage)
    #   template sibling ratio >= 0.85 once every number is masked, and not a near-duplicate: the same wording with
    #                    different numbers, so a different answer (soft leakage: training teaches the template)
    # Every pair at ratio 0.70 or more is written out for reading by eye.
    NEAR = 0.85
    REVIEW = 0.70
    _q = manifest.set_index("id")["question"]
    _a = manifest.set_index("id")["answer"]

    def _pairs(df, col, tid):
        out = []
        for _, r in df.sort_values(col, ascending=False).iterrows():
            out.append({"heldout_id": r["id"], "train_id": r[tid], "ratio": round(r[col], 4),
                        "heldout_q": _q[r["id"]], "train_q": _q[r[tid]],
                        "heldout_answer": _a[r["id"]], "train_answer": _a[r[tid]]})
        return out

    overlap = {"thresholds": {"near_duplicate_ratio": NEAR, "review_ratio": REVIEW}, "within_set": within, "cross": {}}
    for _s in ("gsm8k_test", "math500"):
        _d = nearest[nearest["set"] == _s]
        _near = (_d["ratio"] >= NEAR) & _d["same_answer"]
        _sib = (_d["ratio_masked"] >= NEAR) & ~_near
        overlap["cross"][_s] = {
            "n": int(len(_d)),
            "exact_in_gsm8k_train": int(_d["exact_in_train"].sum()),
            "near_duplicates": int(_near.sum()),
            "template_siblings": int(_sib.sum()),
            "n_review": int((_d["ratio"] >= REVIEW).sum()),
            "ratio_median": round(float(_d["ratio"].median()), 4),
            "ratio_p99": round(float(_d["ratio"].quantile(0.99)), 4),
            "review_pairs": _pairs(_d[_d["ratio"] >= REVIEW], "ratio", "train_id"),
            "template_sibling_pairs": _pairs(_d[_sib], "ratio_masked", "train_id_masked"),
        }
    (ML / "overlap.json").write_text(json.dumps(overlap, indent=1, ensure_ascii=False))
    nearest.to_csv(ML / "nearest_train.csv", index=False)
    return NEAR, REVIEW, overlap


@app.cell
def _(PILOT, json, math, pd):
    def wilson(k, n, z=1.96):
        if n == 0:
            return (float("nan"), float("nan"))
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (c - h, c + h)

    pilot_readout = json.loads(open(PILOT / "readout2b_base_stride10.json").read())
    _rows = []
    for _c, _s in pilot_readout["summary"].items():
        _k = round(_s["acc"] * _s["n"])
        _lo, _hi = wilson(_k, _s["n"])
        _rows.append({"cap": int(_c), "acc": _s["acc"], "k": _k, "n": _s["n"], "lo": _lo, "hi": _hi,
                      "reasoning_mean": _s["reasoning_mean"], "forced_frac": _s["forced_frac"]})
    pilot_curve = pd.DataFrame(_rows).sort_values("cap").reset_index(drop=True)
    _caps = sorted(int(c) for c in pilot_readout["items"])
    _ids = list(pilot_readout["items"][str(_caps[0])])
    pilot_items = pd.DataFrame({f"correct_{c}": [pilot_readout["items"][str(c)][i]["correct"] for i in _ids] for c in _caps},
                               index=_ids)
    pilot_items["think_len"] = [pilot_readout["think_len"][i] for i in _ids]
    _unclosed_len = max(_caps) + pilot_readout["args"]["answer_max"]
    pilot_items["closed"] = pilot_items["think_len"] < _unclosed_len

    def _load(p):
        return [json.loads(line) for line in open(p)]

    grpo_logs = {m: pd.DataFrame(_load(PILOT / f"grpo{m}_wide_s0/train_log.jsonl")) for m in ("2b", "4b")}
    grpo_rollouts = {m: pd.DataFrame(_load(PILOT / f"grpo{m}_wide_s0/rollouts.jsonl")) for m in ("2b", "4b")}
    return grpo_logs, grpo_rollouts, pilot_curve, pilot_items, pilot_readout, wilson


@app.cell
def _(ML, guard, profile_frame):
    eda_paths = guard.eda_figures(profile_frame, target="set", state_dir=ML)
    return (eda_paths,)


@app.cell
def _(ML, guard, manifest, np):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"gsm8k_train": "#4C72B0", "gsm8k_test": "#DD8452", "math500": "#55A868"}
    _f, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _bins = np.linspace(0, 700, 71)
    _med = {}
    for _s, _c in COLORS.items():
        _v = manifest.loc[manifest["set"] == _s, "q_tokens"]
        _med[_s] = (int(_v.median()), int(_v.quantile(0.95)), int(_v.max()))
        _ax[0].hist(_v, bins=_bins, density=True, histtype="step", lw=2, color=_c,
                    label=f"{_s} (median {_med[_s][0]})")
        _ax[1].plot(np.sort(_v), np.linspace(0, 1, len(_v)), color=_c, lw=2, label=_s)
    _ax[0].set_xlabel("question length, Qwen3.5 tokens")
    _ax[0].set_ylabel("density")
    _ax[0].legend()
    _ax[0].set_title("Question length by set")
    _ax[1].set_xscale("log")
    _ax[1].set_xlabel("question length, Qwen3.5 tokens (log)")
    _ax[1].set_ylabel("share of items at or below")
    _ax[1].set_title("Cumulative")
    _ax[1].legend()
    _ov = int(manifest["prompt_tokens"].iloc[0] - manifest["q_tokens"].iloc[0])
    guard.fig(2, "question_length", _f,
              f"Question length in Qwen3.5 tokens: GSM8K train median {_med['gsm8k_train'][0]} (95th pct "
              f"{_med['gsm8k_train'][1]}), GSM8K test median {_med['gsm8k_test'][0]}, MATH-500 median "
              f"{_med['math500'][0]} (95th pct {_med['math500'][1]}, max {_med['math500'][2]}). GSM8K train and test "
              f"have the same shape, so the RL prompts match the in-domain test set; MATH-500 is shorter at the median "
              f"but has a long tail. The chat template and boxed-answer instruction add {_ov} tokens to every prompt. "
              f"Prompts are tiny next to the 2k reasoning cap, so the cost of a rollout is the reasoning, not the "
              f"question.", ML)
    return COLORS, plt


@app.cell
def _(COLORS, ML, guard, manifest, np, plt):
    _f, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _bins = np.arange(-3, 9.5, 0.5)
    _num = {}
    for _s, _c in COLORS.items():
        _d = manifest[manifest["set"] == _s]
        _v = _d["answer_log10_abs"].dropna()
        _num[_s] = (len(_v) / len(_d), float(_v.median()) if len(_v) else float("nan"), float(_v.max()) if len(_v) else 0)
        _ax[0].hist(_v, bins=_bins, density=True, histtype="step", lw=2, color=_c, label=f"{_s} ({_num[_s][0]:.0%} numeric)")
    _ax[0].set_xlabel("log10 |answer| (numeric answers only)")
    _ax[0].set_ylabel("density")
    _ax[0].set_title("Answer magnitude")
    _ax[0].legend()
    _types = ["integer", "decimal", "fraction", "expression", "tuple_or_interval", "text"]
    _w = 0.27
    for _k, (_s, _c) in enumerate(COLORS.items()):
        _share = manifest[manifest["set"] == _s]["answer_type"].value_counts(normalize=True).reindex(_types).fillna(0)
        _ax[1].bar(np.arange(len(_types)) + (_k - 1) * _w, _share.to_numpy(), _w, color=_c, label=_s)
    _ax[1].set_xticks(np.arange(len(_types)))
    _ax[1].set_xticklabels(_types, rotation=20)
    _ax[1].set_ylabel("share of items")
    _ax[1].set_title("Answer type")
    _ax[1].legend()
    _gt = manifest[manifest["set"] == "gsm8k_train"]
    _m5 = manifest[manifest["set"] == "math500"]
    guard.fig(2, "answer_magnitude", _f,
              f"GSM8K answers are numbers ({_num['gsm8k_train'][0]:.1%} of train parse as a number; median size about "
              f"10^{_num['gsm8k_train'][1]:.1f}, largest about 10^{_num['gsm8k_train'][2]:.1f}); GSM8K test has the same "
              f"profile. MATH-500 answers are numeric for {_num['math500'][0]:.0%} of items and the rest are "
              f"expressions, tuples, intervals or text ({(_m5['answer_type'] == 'expression').mean():.0%} expressions). "
              f"So the RL reward on GSM8K only ever checks a number, while the MATH-500 readout also needs symbolic "
              f"answers the grader (math_verify) must match: MATH-500 measures transfer, GSM8K test measures the "
              f"trained skill.", ML)
    return


@app.cell
def _(ML, NEAR, REVIEW, guard, nearest, np, overlap, plt):
    _f, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _bins = np.linspace(0, 1, 51)
    for _k, _s in enumerate(("gsm8k_test", "math500")):
        _d = nearest[nearest["set"] == _s]
        _ax[_k].hist(_d["ratio"], bins=_bins, alpha=0.7, label="numbers kept")
        _ax[_k].hist(_d["ratio_masked"], bins=_bins, histtype="step", lw=2, label="numbers masked")
        _ax[_k].axvline(NEAR, color="red", ls="--", label=f"near-duplicate line {NEAR}")
        _ax[_k].axvline(REVIEW, color="grey", ls=":", label=f"read-by-eye line {REVIEW}")
        _ax[_k].set_yscale("log")
        _ax[_k].set_xlabel("match ratio with the closest GSM8K train question")
        _ax[_k].set_ylabel("items (log)")
        _ax[_k].set_title(f"{_s}: closest GSM8K train item")
        _ax[_k].legend(fontsize=8)
    _g = overlap["cross"]["gsm8k_test"]
    _m = overlap["cross"]["math500"]
    guard.fig(2, "train_test_overlap", _f,
              f"For every held-out question, how close the nearest GSM8K train question is (1.0 = identical text). "
              f"GSM8K test: {_g['exact_in_gsm8k_train']} exact copies, {_g['near_duplicates']} near-duplicates "
              f"(same problem reworded, same answer), {_g['template_siblings']} template siblings (same wording with "
              f"different numbers and answer, ratio {NEAR} or more once numbers are masked), median "
              f"closest ratio {_g['ratio_median']}. MATH-500: {_m['exact_in_gsm8k_train']} exact, "
              f"{_m['near_duplicates']} near-duplicates, {_m['template_siblings']} template siblings, median "
              f"{_m['ratio_median']}. The flagged train items (listed in overlap.json) are what step 4 proposes to drop "
              f"from the RL prompt pool, so a GSM8K test gain cannot come from a memorized training problem.", ML)
    return


@app.cell
def _(ML, guard, pilot_items, pilot_readout, plt):
    _caps = sorted(int(c) for c in pilot_readout["items"])
    _cl = pilot_items[pilot_items["closed"]]["think_len"].sort_values()
    _n_open = int((~pilot_items["closed"]).sum())
    _f, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _s = pilot_items.sort_values("think_len").reset_index(drop=True)
    _ax[0].bar(range(len(_s)), _s["think_len"],
               color=["#4C72B0" if c else "#C44E52" for c in _s["closed"]])
    for _c in _caps:
        _ax[0].axhline(_c, color="grey", ls=":", lw=1)
        _ax[0].text(0.5, _c, f" cap {_c}", fontsize=7, va="bottom")
    _ax[0].set_xlabel("MATH-500 item (sorted)")
    _ax[0].set_ylabel("reasoning tokens before </think>")
    _ax[0].set_title(f"Untouched 2B: blue closed its thinking, red still thinking at {max(_caps) + pilot_readout['args']['answer_max']}")
    _r2 = pilot_items.copy()
    _corr8 = _r2[f"correct_{max(_caps)}"]
    _ax[1].hist([_r2.loc[_r2["closed"] & (_corr8 == 1), "think_len"], _r2.loc[_r2["closed"] & (_corr8 == 0), "think_len"]],
                bins=12, stacked=True, label=["closed, correct at 8k", "closed, wrong at 8k"], color=["#55A868", "#DD8452"])
    _ax[1].set_xlabel("reasoning tokens (items that closed their thinking)")
    _ax[1].set_ylabel("items")
    _ax[1].legend()
    _ax[1].set_title("Length of the traces that finished")
    _acc_closed = float(_r2.loc[_r2["closed"], f"correct_{max(_caps)}"].mean()) if len(_cl) else float("nan")
    _acc_open = float(_r2.loc[~_r2["closed"], f"correct_{max(_caps)}"].mean())
    guard.fig(2, "pilot_reasoning_length", _f,
              f"Pilot7 readout, untouched Qwen3.5-2B thinking on 50 MATH-500 items (vendor sampling, one sample each): "
              f"only {len(_cl)} of 50 closed their thinking inside {max(_caps) + pilot_readout['args']['answer_max']} "
              f"tokens (shortest {int(_cl.min())}, median {int(_cl.median())}); {_n_open} were still thinking. Items "
              f"that finished were right {_acc_closed:.0%} of the time at the 8k cap, items cut off "
              f"{_acc_open:.0%}. None finished under 1,000 tokens, which is why a 2k training cap on MATH-level "
              f"problems gives almost no correct rollouts and no reward signal.", ML)
    return


@app.cell
def _(ML, guard, pilot_curve, pilot_items, plt):
    _f, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _ax[0].errorbar(pilot_curve["reasoning_mean"], pilot_curve["acc"],
                    yerr=[pilot_curve["acc"] - pilot_curve["lo"], pilot_curve["hi"] - pilot_curve["acc"]],
                    marker="o", capsize=4, lw=2)
    for _, _r in pilot_curve.iterrows():
        _ax[0].annotate(f"cap {int(_r['cap'])}", (_r["reasoning_mean"], _r["acc"]), textcoords="offset points",
                        xytext=(5, 6), fontsize=8)
    _ax[0].set_xscale("log", base=2)
    _ax[0].set_xlabel("mean reasoning tokens used per item (log2)")
    _ax[0].set_ylabel("accuracy (95% Wilson interval, n=50)")
    _ax[0].set_ylim(0, 0.6)
    _ax[0].set_title("Untouched 2B on MATH-500: accuracy vs reasoning tokens")
    _ax[1].bar(pilot_curve["cap"].astype(str), pilot_curve["forced_frac"], color="#C44E52")
    _ax[1].set_ylim(0, 1.05)
    _ax[1].set_xlabel("reasoning cap")
    _ax[1].set_ylabel("share of items cut off at the cap")
    _ax[1].set_title("Cut-off share by cap")
    _c = pilot_curve.set_index("cap")
    _both = int(((pilot_items["correct_512"] == 1) & (pilot_items["correct_8192"] == 1)).sum())
    guard.fig(2, "pilot_accuracy_vs_budget", _f,
              f"The accuracy-vs-tokens curve, the shape step 7 must move: 0.16 at 512, 0.16 at 1,024, 0.14 at 2,048, "
              f"0.20 at 4,096, 0.28 at 8,192 reasoning tokens, with {_c.loc[8192, 'forced_frac']:.0%} of items still cut "
              f"off at 8k. With 50 items every interval is about plus or minus 0.11, so 512 to 4k are statistically "
              f"flat and only 8k is plausibly higher; {_both} items are right at both 512 and 8k. This baseline curve "
              f"is from MATH-500, a held-out set, so it is used here only to describe the problem, not to pick the "
              f"training cap.", ML)
    return


@app.cell
def _(ML, grpo_logs, grpo_rollouts, guard, np, plt):
    _f, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _bins = np.linspace(0, 2100, 43)
    for _m, _c in (("2b", "#4C72B0"), ("4b", "#DD8452")):
        _r = grpo_rollouts[_m]
        _ax[0].hist(_r["reasoning"], bins=_bins, histtype="step", lw=2, color=_c,
                    label=f"{_m.upper()}: {len(_r)} rollouts, {_r['forced'].mean():.0%} cut off, {_r['correct'].mean():.1%} correct")
    _ax[0].axvline(2048, color="red", ls="--", label="2k cap")
    _ax[0].set_yscale("log")
    _ax[0].set_xlabel("reasoning tokens per rollout")
    _ax[0].set_ylabel("rollouts (log)")
    _ax[0].legend(fontsize=8)
    _ax[0].set_title("Pilot7 GRPO rollouts on MATH train (levels 3-5)")
    for _m, _mk in (("2b", "o"), ("4b", "s")):
        _l = grpo_logs[_m]
        _ax[1].plot(_l["step"], _l["groups_all_wrong"] / 16, marker=_mk, label=f"{_m.upper()} share of groups all wrong")
        _ax[1].plot(_l["step"], _l["acc"], marker=_mk, ls="--", label=f"{_m.upper()} rollout accuracy")
    _ax[1].set_ylim(0, 1.05)
    _ax[1].set_xlabel("GRPO step")
    _ax[1].legend(fontsize=8)
    _ax[1].set_title("Reward signal per step")
    _r2 = grpo_rollouts["2b"]
    _r4 = grpo_rollouts["4b"]
    guard.fig(2, "pilot_grpo_rollouts", _f,
              f"Pilot7 GRPO, wide arm, seed 0, 2k cap on MATH train: 2B rollouts were cut off {_r2['forced'].mean():.0%} "
              f"of the time and correct {_r2['correct'].mean():.1%}; 4B cut off {_r4['forced'].mean():.0%}, correct "
              f"{_r4['correct'].mean():.1%}. Most prompt groups were all wrong (2B: 13 to 16 of 16), and an all-wrong "
              f"group has zero advantage, so the policy gradient is near zero. The length penalty only acts on correct "
              f"rollouts, so it needs a prompt set where the untouched model is right 30 to 60% of the time at the "
              f"training cap: GSM8K train, with the cap read off a validation slice.", ML)
    return


@app.cell
def _(ML, data_profile, eda_paths, file_sha, json, manifest, overlap, pilot_curve, sha_matches_frozen, within):
    _by = manifest.groupby("set")
    summary = {
        "rows": {k: int(v) for k, v in _by.size().items()},
        "q_tokens": {k: {"median": float(v.median()), "p95": float(v.quantile(0.95)), "max": int(v.max()),
                         "min": int(v.min())} for k, v in _by["q_tokens"]},
        "prompt_overhead_tokens": int(manifest["prompt_tokens"].iloc[0] - manifest["q_tokens"].iloc[0]),
        "answer_type_share": {k: {t: round(float(x), 4) for t, x in v.value_counts(normalize=True).items()}
                              for k, v in _by["answer_type"]},
        "answer_tokens": {k: {"median": float(v.median()), "max": int(v.max())} for k, v in _by["answer_tokens"]},
        "answer_log10_abs": {k: {"median": float(v.median()), "max": float(v.max()), "numeric_share": float(v.notna().mean())}
                             for k, v in _by["answer_log10_abs"]},
        "file_sha256": file_sha,
        "sha_matches_frozen_manifest": sha_matches_frozen,
        "within_set": within,
        "cross": {s: {k: v for k, v in d.items() if not k.endswith("_pairs")} for s, d in overlap["cross"].items()},
        "profile_traits": data_profile["traits"],
        "profile_leakage_suspects": data_profile["leakage_suspects"],
        "profile_duplicates": data_profile["duplicates"],
        "pilot_curve": pilot_curve.to_dict(orient="records"),
        "eda_figures": [p.name for p in eda_paths],
    }
    (ML / "step1_3_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return (summary,)


if __name__ == "__main__":
    app.run()
