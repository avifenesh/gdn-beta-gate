#!/usr/bin/env python3
"""Answer cap from an uncensored step 9 curve (rule in ml_pipeline/PIPELINE.md, 2026-10-08 about 22:32): answer_max =
the smallest multiple of 256 at or above the 99th percentile of answer tokens among answers that ended within the run's
answer cap, over all budgets. Also reports, per budget, the answer-token distribution, the unended share at the run's
cap and at the chosen cap, and boxed-missing shares. usage: answer_cap.py RECEIPT.json TOKENIZER_DIR OUT.json"""
import gzip
import json
import math
import sys

from transformers import AutoTokenizer


def main():
    rec_p, tok_p, out_p = sys.argv[1:4]
    rec = json.load(open(rec_p))
    tok = AutoTokenizer.from_pretrained(tok_p)
    run_cap = rec["args"]["answer_max"]
    ended = {(r["id"], str(r["sample"])): r["caps"] for r in rec["rows"]}
    by_cap, all_ended = {}, []
    with gzip.open(rec_p + ".traces.jsonl.gz", "rt") as f:
        for line in f:
            t = json.loads(line)
            caps = ended[(t["id"], str(t["sample"]))]
            ans = t["answers"]
            if isinstance(ans, str):
                import ast
                ans = ast.literal_eval(ans)
            for c, text in ans.items():
                n = len(tok.encode(text, add_special_tokens=False))
                e = bool(caps[c][3])
                d = by_cap.setdefault(c, {"ended": [], "unended": 0, "boxed_missing": 0, "n": 0})
                d["n"] += 1
                d["boxed_missing"] += int("\\boxed" not in text)
                if e:
                    d["ended"].append(n); all_ended.append(n)
                else:
                    d["unended"] += 1
    all_ended.sort()
    p99 = all_ended[min(len(all_ended) - 1, math.ceil(0.99 * len(all_ended)) - 1)]
    X = int(math.ceil(p99 / 256) * 256)
    out = {"receipt": rec_p, "run_answer_max": run_cap, "p99_ended": p99, "answer_max": X, "budgets": {}}
    for c, d in sorted(by_cap.items(), key=lambda kv: int(kv[0])):
        e = sorted(d["ended"])
        q = lambda p: e[min(len(e) - 1, max(0, math.ceil(p * len(e)) - 1))] if e else None  # noqa: E731
        out["budgets"][c] = {"n": d["n"], "unended_at_run_cap": d["unended"] / d["n"],
                             "unended_at_X": (d["unended"] + sum(x > X for x in e)) / d["n"],
                             "boxed_missing": d["boxed_missing"] / d["n"],
                             "answer_tokens_median": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": e[-1] if e else None}
    json.dump(out, open(out_p, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
