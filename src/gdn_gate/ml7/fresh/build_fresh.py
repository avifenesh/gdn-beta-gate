#!/usr/bin/env python3
"""build_fresh.py: the fresh held-out test set of lane wg Phase 5 (owner approval 2026-10-12): AIME 2024, AIME 2025,
AIME 2026 and AMC 2023, from Hugging Face datasets at pinned revisions, deduplicated against the RL pool, the
validation slice and each other, frozen with sha256, plus a fresh guard state (test touches 0).

Sources (repo, revision, file; every downloaded file's sha256 goes into the manifest):
  aime2024  HuggingFaceH4/aime_2024 2fe88a2f...   30 items, ids from the AoPS url (2024 AIME I / II, problem n)
  aime2025  MathArena/aime_2025     c94da77e...   30 items, problem_idx 1-15 = AIME I, 16-30 = AIME II
  aime2026  MathArena/aime_2026     d2de22f3...   30 items, same indexing (contests 2026-02-05 / 2026-02-11)
  amc2023   AI-MO/aimo-validation-amc 69d78a4a... rows whose AoPS url is 2023 AMC 12A / 12B (40 items); checked equal to
            math-ai/amc23 80815d37... (same urls and problem texts)
Answers are integers (AIME 0-999; the AMC rows carry the AIMO integer conversion), written as plain integer strings.

Dedup: the 2B record's sibling scan (ml_pipeline/02_prep.py: n-gram cosine top 20, then character match ratio >= 0.85
with numbers kept or masked) of every fresh item against rl_pool.jsonl and val.jsonl (any hit drops the item) and
against the other fresh items (a pair drops the later one); the scan against test_gsm8k / test_math500 is reported
only. Row: {"id": "fresh:<src>-<contest>-<n>", "problem", "answer", "set": "fresh", "source", "contest", "number"}.

Outputs: DATA/test_fresh.jsonl, DATA/fresh_manifest.json (sources, sha256, counts, drops, rules), and with --guard DIR
DIR/.guard_state.json (fingerprint of the frame id, problem, answer in file order, guard._fingerprint; touches 0) and
DIR/split_manifest.json ({"files": {"test_fresh.jsonl": sha256}}, the manifest the analysis reads). Refuses to
overwrite a frozen file whose content would change.

Usage: build_fresh.py --raw RAW_DIR --data DATA_DIR --guard ml_pipeline_4b/fresh_test [--download]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]
NEAR, TOPK = 0.85, 20
SOURCES = {
    "aime2024": ("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", "data/train-00000-of-00001.parquet"),
    "aime2025": ("MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302", "data/train-00000-of-00001.parquet"),
    "aime2026": ("MathArena/aime_2026", "d2de22f3c656b4f56cf8981212186377d1e23bc3", "data/train-00000-of-00001.parquet"),
    "amc2023": ("AI-MO/aimo-validation-amc", "69d78a4a2c840e82d69af6bc742bda09005f6316", "data/train-00000-of-00001.parquet"),
}
CROSS = ("math-ai/amc23", "80815d37005feb82cd7f8fbc6901d5d3eff43057", "test-00000-of-00001.parquet")
ORDER = ("aime2024", "aime2025", "aime2026", "amc2023")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def local(raw: Path, repo: str, f: str) -> Path:
    return raw / repo.replace("/", "__") / f


def download(raw: Path) -> None:
    from huggingface_hub import hf_hub_download

    for repo, rev, f in list(SOURCES.values()) + [CROSS]:
        hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=raw / repo.replace("/", "__"))


def int_answer(x) -> str:
    v = float(str(x).strip())
    if v != int(v):
        raise ValueError(f"non-integer answer {x!r}")
    return str(int(v))


def load_items(raw: Path) -> tuple[list[dict], dict]:
    items, info = [], {}
    for src in ORDER:
        repo, rev, f = SOURCES[src]
        p = local(raw, repo, f)
        df = pd.read_parquet(p)
        info[src] = {"repo": repo, "revision": rev, "file": f, "sha256": sha(p), "rows": int(len(df))}
        rows = []
        if src == "aime2024":
            for _, r in df.iterrows():
                m = re.search(r"2024_AIME_(I|II)_Problems/Problem_(\d+)", r["url"])
                rows.append((m.group(1), int(m.group(2)), r["problem"], r["answer"], r["url"]))
        elif src in ("aime2025", "aime2026"):
            for _, r in df.iterrows():
                k = int(r["problem_idx"])
                rows.append(("I" if k <= 15 else "II", k if k <= 15 else k - 15, r["problem"], r["answer"], None))
        else:
            for _, r in df.iterrows():
                m = re.search(r"2023_AMC_(12A|12B)_Problems/Problem_(\d+)", r["url"])
                if m:
                    rows.append((m.group(1), int(m.group(2)), r["problem"], r["answer"], r["url"]))
        rows.sort(key=lambda t: (t[0], t[1]))
        for contest, n, prob, ans, url in rows:
            items.append({"id": f"fresh:{src}-{contest}-{n}", "problem": str(prob).strip(), "answer": int_answer(ans),
                          "set": "fresh", "source": src, "contest": contest, "number": n,
                          **({"url": url} if url else {})})
        info[src]["kept_before_dedup"] = len(rows)
    # cross-check of the AMC 2023 rows against math-ai/amc23
    cp = local(raw, CROSS[0], CROSS[2])
    cdf = pd.read_parquet(cp)
    a23 = [x for x in items if x["source"] == "amc2023"]
    same_urls = set(cdf["url"]) == {x["url"] for x in a23}
    same_text = {str(q).strip() for q in cdf["question"]} == {x["problem"] for x in a23}
    same_ans = sorted(int_answer(v) for v in cdf["answer"]) == sorted(x["answer"] for x in a23)
    info["amc2023"]["cross_check"] = {"repo": CROSS[0], "revision": CROSS[1], "sha256": sha(cp), "same_urls": same_urls,
                                      "same_problem_texts": same_text, "same_answers": same_ans}
    if not (same_urls and same_text and same_ans):
        raise SystemExit(f"AMC 2023 cross-check failed: {info['amc2023']['cross_check']}")
    return items, info


# --- the 2B record's sibling scan (ml_pipeline/02_prep.py), verbatim logic
def norm(s, mask_numbers=False):
    s = s.lower()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    if mask_numbers:
        s = re.sub(r"\d+(\.\d+)?", "#", s)
    s = re.sub(r"[^a-z0-9#\\]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def hashed_ngrams(texts, dim=1 << 20):
    import zlib

    from scipy.sparse import csr_matrix

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


def sibling_scan(left: list[dict], right: list[dict], same_set: bool = False) -> list[dict]:
    ln = [norm(x["problem"]) for x in left]
    lm = [norm(x["problem"], True) for x in left]
    rn = [norm(x["problem"]) for x in right]
    rm = [norm(x["problem"], True) for x in right]
    cos = (hashed_ngrams(ln) @ hashed_ngrams(rn).T).toarray()
    cosm = (hashed_ngrams(lm) @ hashed_ngrams(rm).T).toarray()
    top = np.argsort(-cos, axis=1)[:, :TOPK + 1]
    topm = np.argsort(-cosm, axis=1)[:, :TOPK + 1]
    out = []
    for i in range(len(left)):
        for j in set(top[i].tolist()) | set(topm[i].tolist()):
            if same_set and j <= i:
                continue
            r = SequenceMatcher(None, ln[i], rn[j], autojunk=False).ratio()
            rmk = SequenceMatcher(None, lm[i], rm[j], autojunk=False).ratio()
            if max(r, rmk) >= NEAR:
                out.append({"left_id": left[i]["id"], "right_id": right[j]["id"], "ratio": round(r, 4),
                            "ratio_masked": round(rmk, 4), "same_answer": norm(left[i]["answer"]) == norm(right[j]["answer"])})
    return out


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p) if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--data", required=True, help="the frozen data dir (rl_pool.jsonl, val.jsonl, tests)")
    ap.add_argument("--guard", required=True, help="state dir of the fresh set's guard (e.g. ml_pipeline_4b/fresh_test)")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args(argv)
    raw, data, gd = Path(a.raw), Path(a.data), Path(a.guard)
    if a.download:
        download(raw)
    items, info = load_items(raw)
    against = {f: jl(data / f) for f in ("rl_pool.jsonl", "val.jsonl")}
    report_only = {f: jl(data / f) for f in ("test_gsm8k.jsonl", "test_math500.jsonl")}
    hits = {f: sibling_scan(items, v) for f, v in against.items()}
    info_hits = {f: sibling_scan(items, v) for f, v in report_only.items()}
    within = sibling_scan(items, items, same_set=True)
    drop = {h["left_id"] for v in hits.values() for h in v} | {h["right_id"] for h in within}
    kept = [x for x in items if x["id"] not in drop]
    lines = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept)
    out = data / "test_fresh.jsonl"
    if out.exists() and out.read_text() != lines:
        raise SystemExit(f"{out} exists with different content: frozen files are never rewritten")
    out.write_text(lines)
    digest = sha(out)
    man = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "what": "lane wg Phase 5 fresh held-out test set (owner approval 2026-10-12); pre-registration in "
                "ml_pipeline_4b/PIPELINE.md, 'Fresh test set'",
        "file": str(out), "sha256": digest, "n_items": len(kept),
        "counts": {s: sum(x["source"] == s for x in kept) for s in ORDER},
        "sources": info, "script_sha256": sha(__file__),
        "dedup": {"rule": f"02_prep.py sibling scan: n-gram cosine top {TOPK}, character ratio >= {NEAR} with numbers "
                          "kept or masked; a hit against rl_pool/val drops the fresh item; a pair inside the fresh set "
                          "drops the later item; test_gsm8k / test_math500 hits are reported only",
                  "hits_rl_pool": hits["rl_pool.jsonl"], "hits_val": hits["val.jsonl"], "hits_within": within,
                  "report_only_test_gsm8k": info_hits["test_gsm8k.jsonl"],
                  "report_only_test_math500": info_hits["test_math500.jsonl"], "dropped": sorted(drop)},
        "against_sha256": {f: sha(data / f) for f in list(against) + list(report_only)},
        "date_note": "Qwen/Qwen3.5-4B weights published on the hub 2026-02-27 (2B 2026-02-28); its card reports "
                     "HMMT Nov 2025 as its newest benchmark and states no data cutoff. AIME 2026 (2026-02-05 / 02-11; "
                     "MathArena/aime_2026 created 2026-02-13) is after every benchmark the card reports; AIME 2024, "
                     "AIME 2025 and AMC 2023 predate the models and may be in their training data.",
    }
    (data / "fresh_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    gd.mkdir(parents=True, exist_ok=True)
    (gd / "split_manifest.json").write_text(json.dumps({"files": {"test_fresh.jsonl": digest},
                                                        "counts": {"test_fresh.jsonl": len(kept)}}, indent=1) + "\n")
    sys.path.insert(0, str(WT / "ml_pipeline"))
    import guard  # noqa: PLC0415

    frame = pd.DataFrame([{k: x[k] for k in ("id", "problem", "answer")} for x in kept])
    st = gd / guard.STATE_FILE
    state = {"test_fingerprint": guard._fingerprint(frame), "n_test": len(kept), "test_touches": 0,
             "duplicates_dropped": len(drop), "split": "fixed_external_test (fresh, Phase 5)",
             "created": datetime.date.today().isoformat(),
             "test_frame": "columns id, problem, answer; rows of test_fresh.jsonl, file order",
             "record": "fresh held-out set read by the Qwen3.5-4B and Qwen3.5-2B arms of lane wg, touch count 1"}
    if st.exists():
        old = json.loads(st.read_text())
        if old["test_fingerprint"] != state["test_fingerprint"]:
            raise SystemExit(f"{st} exists with a different fingerprint: never rewritten")
        print(f"guard state kept ({st}, touches {old['test_touches']})")
    else:
        st.write_text(json.dumps(state, indent=2))
    print(json.dumps({"n_items": len(kept), "counts": man["counts"], "dropped": sorted(drop), "sha256": digest,
                      "hits_rl_pool": len(hits["rl_pool.jsonl"]), "hits_val": len(hits["val.jsonl"]),
                      "within": len(within), "math500_hits": len(info_hits["test_math500.jsonl"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
