#!/usr/bin/env python3
"""Rebuild the six frozen data files from their public sources and check them against the split manifests.

The repository ships only item ids (data/splits/*.ids, file order) and checksums, not problem text. This script
downloads GSM8K and MATH-500 at the pinned revisions in data/MANIFEST.json and writes rl_pool.jsonl, val.jsonl,
test_gsm8k.jsonl and test_math500.jsonl in the exact byte format the runs used, checked against
ml_pipeline/split_manifest.json. It then downloads AIME 2024, AIME 2025, AIME 2026 and AMC 2023 at their pinned
revisions and writes the fresh held-out set test_fresh.jsonl (the row rules of src/gdn_gate/ml7/fresh/build_fresh.py;
its dedup scan dropped no item), checked against ml_pipeline_4b/fresh_test/split_manifest.json, and the 9B record's
test_aime2025.jsonl (the 30 AIME 2025 lines of test_fresh.jsonl, bytes and order unchanged, as
src/gdn_gate/ml7/test9b/build_aime2025.py cut it), checked against ml_pipeline_9b/aime2025_test/split_manifest.json.
Exit 1 on any mismatch.

Usage: python scripts/rebuild_data.py [--out DIR] [--skip_fresh]   (default $WG_RUNS/ml7/data, WG_RUNS defaults to ./runs)
Needs: huggingface_hub, pyarrow (CPU only, about 3 MB of downloads).
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gsm8k_answer(solution: str) -> str:
    """Final answer after '####', thousands separators removed (the frozen files carry no commas)."""
    a = solution.split("####")[-1].strip()
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", a)


def int_answer(x) -> str:
    """build_fresh.py's answer rule: an integer value written as a plain integer string."""
    v = float(str(x).strip())
    if v != int(v):
        raise ValueError(f"non-integer answer {x!r}")
    return str(int(v))


def fresh_rows(man: dict) -> dict:
    """The fresh held-out set rows by id, as src/gdn_gate/ml7/fresh/build_fresh.py writes them."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    src = {s["name"]: s for s in man["sources"]}
    rows = {}
    for name in ("aime2024", "aime2025", "aime2026", "amc2023"):
        s = src[name]
        p = hf_hub_download(s["repo"], s["file"], repo_type="dataset", revision=s["revision"])
        got = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if got != s["sha256"]:
            sys.exit(f"FAIL {name}: downloaded file sha256 {got[:16]} differs from data/MANIFEST.json")
        items = []
        for r in pq.read_table(p).to_pylist():
            if name == "aime2024":
                m = re.search(r"2024_AIME_(I|II)_Problems/Problem_(\d+)", r["url"])
                items.append((m.group(1), int(m.group(2)), r["problem"], r["answer"], r["url"]))
            elif name in ("aime2025", "aime2026"):
                k = int(r["problem_idx"])
                items.append(("I" if k <= 15 else "II", k if k <= 15 else k - 15, r["problem"], r["answer"], None))
            else:
                m = re.search(r"2023_AMC_(12A|12B)_Problems/Problem_(\d+)", r["url"])
                if m:
                    items.append((m.group(1), int(m.group(2)), r["problem"], r["answer"], r["url"]))
        items.sort(key=lambda t: (t[0], t[1]))
        for contest, n, prob, ans, url in items:
            rid = f"fresh:{name}-{contest}-{n}"
            rows[rid] = {"id": rid, "problem": str(prob).strip(), "answer": int_answer(ans), "set": "fresh",
                         "source": name, "contest": contest, "number": n, **({"url": url} if url else {})}
    return rows


def write_check(out: Path, name: str, rows: dict, want: dict) -> bool:
    ids = (ROOT / f"data/splits/{name}.ids").read_text().split()
    text = "".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in ids)
    (out / f"{name}.jsonl").write_text(text)
    h = hashlib.sha256(text.encode()).hexdigest()
    ok = h == want[f"{name}.jsonl"]
    print(f"{'OK  ' if ok else 'FAIL'} {name}.jsonl rows {len(ids)} sha256 {h[:16]} (manifest {want[name + '.jsonl'][:16]})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("WG_RUNS", str(ROOT / "runs")), "ml7/data"))
    ap.add_argument("--skip_fresh", action="store_true", help="only the four RL files")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    man = json.loads((ROOT / "data/MANIFEST.json").read_text())
    src = {s["name"]: s for s in man["sources"]}
    g = src["gsm8k"]
    m = src["math500"]
    rows = {}
    for split in ("train", "test"):
        p = hf_hub_download(g["repo"], f"main/{split}-00000-of-00001.parquet", repo_type="dataset",
                            revision=g["revision"])
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            ans = gsm8k_answer(r["answer"])
            rid = f"gsm8k_{split}:gsm8k_train{i}" if split == "train" else f"gsm8k_test:{i}"
            row = {"id": rid, "problem": r["question"], "answer": ans, "set": f"gsm8k_{split}", "level": "gsm8k",
                   "solution": "\\boxed{" + ans + "}"}
            rows[rid] = row
    p = hf_hub_download(m["repo"], "test.jsonl", repo_type="dataset", revision=m["revision"])
    for line in open(p):
        r = json.loads(line)
        rid = "math500:" + r["unique_id"]
        rows[rid] = {"id": rid, "problem": r["problem"], "answer": r["answer"], "set": "math500",
                     "level": str(r["level"]), "subject": r["subject"]}
    want = json.loads((ROOT / "ml_pipeline/split_manifest.json").read_text())["files"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = [name for name in ("rl_pool", "val", "test_gsm8k", "test_math500") if not write_check(out, name, rows, want)]
    if not args.skip_fresh:
        fwant = json.loads((ROOT / "ml_pipeline_4b/fresh_test/split_manifest.json").read_text())["files"]
        if not write_check(out, "test_fresh", fresh_rows(man), fwant):
            bad.append("test_fresh")
        else:  # the 9B record's AIME 2025 file: the fresh file's aime2025 lines, bytes and order unchanged
            awant = json.loads((ROOT / "ml_pipeline_9b/aime2025_test/split_manifest.json").read_text())["files"]
            lines = [ln for ln in (out / "test_fresh.jsonl").read_text().splitlines(keepends=True)
                     if json.loads(ln)["id"].startswith("fresh:aime2025-")]
            (out / "test_aime2025.jsonl").write_text("".join(lines))
            h = hashlib.sha256("".join(lines).encode()).hexdigest()
            ok = h == awant["test_aime2025.jsonl"]
            print(f"{'OK  ' if ok else 'FAIL'} test_aime2025.jsonl rows {len(lines)} sha256 {h[:16]} "
                  f"(manifest {awant['test_aime2025.jsonl'][:16]})")
            if not ok:
                bad.append("test_aime2025")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
