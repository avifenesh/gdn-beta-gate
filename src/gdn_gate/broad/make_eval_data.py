#!/usr/bin/env python3
"""Freeze every eval input of the broad stream into one directory, so the Sparks need no dataset downloads.

Writes to --out:
  mmlu_pro.jsonl   TIGER-Lab/MMLU-Pro test (12,032)            id = question_id
  arc_c.jsonl      allenai/ai2_arc ARC-Challenge test (1,172)  id = id
  gsm8k.jsonl      openai/gsm8k main test (1,319)              id = row index
  humaneval.jsonl  openai/openai_humaneval test (164)          id = task_id
  mbppplus.jsonl   evalplus/mbppplus test (378)                id = task_id
  swde.jsonl, fda.jsonl, squad.jsonl   hazyresearch/based-* validation (recall)  id = row index
  ruler.jsonl      synthetic RULER-style items (niah_single, niah_multikey, vt) at 4k and 16k tokens, 50 each
  math500.jsonl    HuggingFaceH4/MATH-500 (step 7 held-out)    id = unique_id
  math_train.jsonl DigitalLearningGmbH/MATH-lighteval train minus any problem in MATH-500 (step 7 training)
  nll_domains.pt   {domain: LongTensor [64, 1024]} for wiki (WikiText-103 test), web (C4 en validation),
                   code (codeparrot-clean-valid, Python), math (MATH-500 problem + solution text)
  manifest.json    sources, revisions, counts, sha256 of every file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ RULER-style synthetic items
NOISE = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")


def _rand_num(rng):
    return str(rng.randint(1_000_000, 9_999_999))


def _rand_key(rng):
    words = ["amber", "brisk", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo", "jasper", "kestrel",
             "lagoon", "meadow", "nectar", "onyx", "pebble", "quartz", "raven", "saffron", "thistle", "umber",
             "velvet", "willow", "zephyr"]
    return f"{rng.choice(words)}-{rng.choice(words)}-{rng.randint(10, 99)}"


def ruler_items(tok, wiki_paras, lengths=(4096, 16384), n=50, seed=4242):
    """niah_single: one needle in repeated noise; niah_multikey: 4 needles in WikiText paragraphs, ask one;
    vt: a 4-hop variable chain hidden in noise, answer = every variable holding the value. Lengths are the
    haystack target in tokens (the question adds a few dozen)."""
    out = []
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    noise_tok = ntok(NOISE)
    for L in lengths:
        for task in ("niah_single", "niah_multikey", "vt"):
            rng = random.Random(f"{seed}:{task}:{L}")
            for i in range(n):
                if task == "niah_single":
                    key, val = _rand_key(rng), _rand_num(rng)
                    needle = f"One of the special magic numbers for {key} is: {val}. "
                    chunks = [NOISE] * max(1, (L - ntok(needle)) // noise_tok)
                    chunks.insert(rng.randint(0, len(chunks)), needle)
                    ctx = "".join(chunks)
                    q = f"What is the special magic number for {key} mentioned in the provided text?"
                    prefix = f"The special magic number for {key} mentioned in the provided text is"
                    ans = [val]
                elif task == "niah_multikey":
                    keys = [_rand_key(rng) for _ in range(4)]
                    vals = [_rand_num(rng) for _ in range(4)]
                    needles = [f"One of the special magic numbers for {k} is: {v}. " for k, v in zip(keys, vals)]
                    paras, n_t = [], sum(ntok(x) for x in needles)
                    while n_t < L:
                        p = rng.choice(wiki_paras)
                        paras.append(p + "\n")
                        n_t += ntok(p) + 1
                    for nd in needles:
                        paras.insert(rng.randint(0, len(paras)), nd)
                    ctx = "".join(paras)
                    j = rng.randrange(4)
                    q = f"What is the special magic number for {keys[j]} mentioned in the provided text?"
                    prefix = f"The special magic number for {keys[j]} mentioned in the provided text is"
                    ans = [vals[j]]
                else:
                    val = str(rng.randint(10000, 99999))
                    names = ["".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(5)]
                    stmts = [f"VAR {names[0]} = {val}. "] + [f"VAR {names[k]} = VAR {names[k - 1]}. " for k in range(1, 5)]
                    chunks = [NOISE] * max(1, (L - sum(ntok(s) for s in stmts)) // noise_tok)
                    pos = sorted(rng.sample(range(len(chunks) + 1), 5))
                    for k, p in reversed(list(enumerate(pos))):  # keep the chain in order
                        chunks.insert(p, stmts[k])
                    ctx = "".join(chunks)
                    q = (f"Find all variables that are assigned the value {val} in the text above, "
                         f"directly or through another variable.")
                    prefix = f"Answer: According to the chain(s) of variable assignment in the text above, 5 variables are assigned the value {val}, they are:"
                    ans = names
                out.append({"id": f"{task}@{L}#{i}", "task": task, "length": L,
                            "user": f"{ctx}\n\n{q}", "answer_prefix": prefix, "answers": ans})
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    man = {"files": {}, "counts": {}}

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    dump(out / "mmlu_pro.jsonl", [{"id": str(r["question_id"]), "question": r["question"], "options": r["options"],
                                   "answer_index": r["answer_index"], "category": r["category"]} for r in ds])
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    dump(out / "arc_c.jsonl", [{"id": r["id"], "question": r["question"], "options": r["choices"]["text"],
                                "answer_index": r["choices"]["label"].index(r["answerKey"])} for r in ds])
    ds = load_dataset("openai/gsm8k", "main", split="test")
    dump(out / "gsm8k.jsonl", [{"id": str(i), "question": r["question"],
                                "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(ds)])
    ds = load_dataset("openai/openai_humaneval", split="test")
    dump(out / "humaneval.jsonl", [dict(r, id=r["task_id"]) for r in ds])
    ds = load_dataset("evalplus/mbppplus", split="test")
    dump(out / "mbppplus.jsonl", [dict(r, id=str(r["task_id"])) for r in ds])
    for name in ("swde", "fda", "squad"):
        ds = load_dataset(f"hazyresearch/based-{name}", split="validation")
        dump(out / f"{name}.jsonl", [{"id": str(i), "text": r["text"].lstrip(), "value": r["value"]}
                                     for i, r in enumerate(ds)])
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dump(out / "math500.jsonl", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
                                  "level": r["level"], "subject": r["subject"]} for r in m500])
    held = {r["problem"].strip() for r in m500}
    tr = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
    rows = [{"id": f"mtrain{i}", "problem": r["problem"], "solution": r["solution"], "level": r["level"],
             "type": r["type"]} for i, r in enumerate(tr) if r["problem"].strip() not in held]
    dump(out / "math_train.jsonl", rows)
    man["counts"]["math_train_removed_overlap"] = len(tr) - len(rows)

    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    paras = [t.strip() for t in wiki["train"]["text"][:200000] if len(t.strip()) > 400 and not t.strip().startswith("=")]
    dump(out / "ruler.jsonl", ruler_items(tok, paras))

    def pack(texts, n=64, L=1024):
        buf, rows = [], []
        for t in texts:
            buf += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
            while len(buf) >= L and len(rows) < n:
                rows.append(buf[:L])
                buf = buf[L:]
            if len(rows) == n:
                break
        assert len(rows) == n, (len(rows), n)
        return torch.tensor(rows, dtype=torch.long)

    from huggingface_hub import hf_hub_download

    c4f = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    c4 = load_dataset("json", data_files=c4f, split="train")
    stack = load_dataset("codeparrot/codeparrot-clean-valid", split="train")
    nll = {"wiki": pack(["\n".join(wiki["test"]["text"])]),
           "web": pack(c4["text"][:5000]),
           "code": pack(stack["content"][:5000]),
           "math": pack([r["problem"] + "\n" + r["solution"] for r in m500])}
    torch.save(nll, out / "nll_domains.pt")
    for f in sorted(out.iterdir()):
        if f.name != "manifest.json":
            man["files"][f.name] = sha(f)
            if f.suffix == ".jsonl":
                man["counts"][f.stem] = sum(1 for _ in open(f))
    man["tokenizer"] = a.tokenizer
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man["counts"]))


if __name__ == "__main__":
    main()
