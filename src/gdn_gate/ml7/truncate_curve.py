#!/usr/bin/env python3
"""The step 9 curve at a smaller answer cap, derived from a run at a larger one (rule in ml_pipeline/PIPELINE.md,
answer cap, 2026-10-08): sampling is per token with fixed seeds, so an answer generated with answer_max X is the first X
tokens of the same answer generated with a larger cap. Each answer text in the traces is re-tokenized, cut at X tokens,
regraded (rl_common.correct, math_verify) and its ended/boxed flags recomputed; rows get the answer-token count as the
6th element. Output has the same schema as readout_vllm.py with args.answer_max = X.
usage: truncate_curve.py RECEIPT.json TOKENIZER_DIR X OUT.json [BROAD_DIR]  (run on the unit that holds args.data)"""
import copy
import gzip
import json
import sys


def main():
    rec_p, tok_p, X, out_p = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    sys.path.insert(0, sys.argv[5] if len(sys.argv) > 5 else __file__.rsplit("/ml7/", 1)[0] + "/broad")
    import rl_common as R
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_p)
    rec = json.load(open(rec_p))
    assert rec["args"]["answer_max"] >= X, "can only cut to a smaller answer cap"
    data = [json.loads(l) for l in open(rec["args"]["data"])]  # the readout's own data file (run where it exists)
    traces = {}
    with gzip.open(rec_p + ".traces.jsonl.gz", "rt") as f:
        for line in f:
            t = json.loads(line)
            traces[(t["id"], str(t["sample"]))] = t["answers"]
    gold = {r["id"]: R.gold_of(r) for r in data}
    out = copy.deepcopy(rec)
    out["args"]["answer_max"] = X
    out["derived_from"] = {"receipt": rec_p, "answer_max": rec["args"]["answer_max"], "method": "truncate and regrade"}
    changed = 0
    for row in out["rows"]:
        ans = traces[(row["id"], str(row["sample"]))]
        for c, v in row["caps"].items():
            text = ans[c]
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) > X:
                text2 = tok.decode(ids[:X])
                new = [int(bool(R.correct(text2, gold[row["id"]]))), v[1], v[2], 0, int(R.boxed(text2) is not None), X]
                changed += int(new[0] != v[0])
                row["caps"][c] = new
            else:
                row["caps"][c] = list(v[:5]) + [len(ids)]
    if "summary" in out:
        del out["summary"]
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from readout_vllm import summarize
    caps = [int(c) for c in out["rows"][0]["caps"]]
    out["summary"] = summarize(out["rows"], caps)
    out["truncation"] = {"answer_max": X, "grades_changed": changed}
    json.dump(out, open(out_p, "w"))
    print(json.dumps({"answer_max": X, "grades_changed": changed, "norm_auc_log2": out["summary"]["norm_auc_log2"],
                      "acc": {k: round(v["acc"], 4) for k, v in out["summary"]["per_cap"].items()}}))


if __name__ == "__main__":
    main()
