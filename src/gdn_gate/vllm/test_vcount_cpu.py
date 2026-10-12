#!/usr/bin/env python3
"""CPU checks for the Qwen3.5 counting probe (vcount.py, olmo/ruler_paired.py --base). Needs numpy; run with the
vLLM environment (env/vllm.txt), which also has lm-eval for --lmeval. No GPU is touched. Each check has a red arm that must fail.

1. args: defaults (tasks ruler_cwe,ruler_fwe, limit 500, lengths 4k/8k/16k, max_model_len = longest + 1024); scoring
   without --engine and a negative --limit are refused.
2. jobs: step 6 names resolve to <root>/<arm>_s<seed>/ckpt_step<step>.pt with the right engine and mode (wide, wide1,
   ctrl -> stock, _strip -> stock with gates zeroed, @step, base); vscore_4b-style NAME=CKPT[:strip|:base] infer the
   engine from the name; --plan groups by engine. Red: missing checkpoint, ctrl_strip, unknown name, a job whose
   engine differs from --engine, a duplicate name.
3. state keys: a wide checkpoint scored as a stock trained run, a ctrl checkpoint on a gated engine, a strip without
   gates and a checkpoint without LoRA are refused.
4. layout: write_samples keeps ruler_paired's fields (doc_id, score under str(L)) plus doc_hash/prompt_hash and drops
   the rest; length_done needs every task at the same limit; check_digests refuses items that differ between jobs.
5. greedy: greedy_fix forces temperature 0 on a request that did not ask for sampling and counts one that did.
6. paired: ruler_paired.py --base ctrl_s0 on dumps written by write_samples gives the known per-task difference and
   macro; items whose doc_hash differs are dropped and counted. Red: without --base (reference "stock") nothing is
   paired. Default unchanged: on an Olmo-style tree (stock/half, no doc_hash) the output has no "base" or
   "hash_mismatch" keys and the means match an independent computation.
7. (--lmeval, ~5 min CPU) end to end through lm-eval 0.4.13 with a fake LM: run_ruler writes the layout, two jobs in
   one process get identical items, and data_only's digest equals run_ruler's. Red: with the cwe RNG reset disabled
   the second job's ruler_cwe items differ and check_digests refuses them.
Usage: test_vcount_cpu.py [--lmeval] [--model DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vcount as V  # noqa: E402

PAIRED = HERE.parent / "olmo" / "ruler_paired.py"
MODEL = os.path.join(os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
                     "models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b")


def raises(fn, exc=(V.VcountError, SystemExit)):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            fn()
    except exc:
        return True
    return False


def fake_root(d: Path) -> Path:
    root = d / "step6"
    for run, steps in (("wide_s0", (300, 1200)), ("wide1_s0", (1200,)), ("ctrl_s0", (1200,)), ("wide_s1", (700, 1200))):
        (root / run).mkdir(parents=True)
        for st in steps:
            (root / run / f"ckpt_step{st}.pt").write_bytes(b"x")
    return root


def check_args():
    a = V.parse_args(["--engine", "wide", "--model", "M", "--out", "O"])
    assert a.tasks_list == ["ruler_cwe", "ruler_fwe"] and a.limit == 500, a
    assert a.lengths_list == [4096, 8192, 16384] and a.max_model_len == 16384 + 1024, a
    a = V.parse_args(["--plan", "--names", "base"])
    assert a.plan and a.model is None
    assert raises(lambda: V.parse_args(["--model", "M", "--out", "O"]))           # no --engine
    assert raises(lambda: V.parse_args(["--engine", "wide", "--model", "M", "--out", "O", "--limit", "-1"]))
    assert raises(lambda: V.parse_args(["--engine", "wide", "--out", "O"]))       # no --model
    print("args: ok")


def check_jobs(d: Path):
    root = str(fake_root(d))
    j = V.resolve_name("wide_s0", root, 1200)
    assert (j.engine, j.mode, j.name, j.seed, j.step) == ("wide", "trained", "wide_s0", 0, 1200), j
    assert j.ckpt.endswith("step6/wide_s0/ckpt_step1200.pt")
    j = V.resolve_name("wide1_s0", root, 1200)
    assert (j.engine, j.mode) == ("wide1", "trained"), j
    j = V.resolve_name("ctrl_s0", root, 1200)
    assert (j.engine, j.mode, j.arm) == ("stock", "trained", "ctrl"), j
    j = V.resolve_name("wide_s0_strip", root, 1200)
    assert (j.engine, j.mode, j.ckpt.endswith("wide_s0/ckpt_step1200.pt")) == ("stock", "strip", True), j
    j = V.resolve_name("wide_s1@700", root, 1200)
    assert (j.name, j.step, j.ckpt.endswith("wide_s1/ckpt_step700.pt")) == ("wide_s1_step700", 700, True), j
    j = V.resolve_name("base", None, 1200)
    assert (j.engine, j.mode, j.ckpt) == ("stock", "base", None), j
    # red arms
    assert raises(lambda: V.resolve_name("wide1_s1", root, 1200))      # no checkpoint
    assert raises(lambda: V.resolve_name("wide_s0@999", root, 1200))   # no checkpoint at that step
    assert raises(lambda: V.resolve_name("ctrl_s0_strip", root, 1200))
    assert raises(lambda: V.resolve_name("wideg_s0", root, 1200))
    assert raises(lambda: V.resolve_name("wide_s0", None, 1200))       # no root
    ck = str(Path(root) / "wide_s0" / "ckpt_step300.pt")
    assert V.parse_job(f"main_wide_s0={ck}", None).engine == "wide"
    assert V.parse_job(f"main9_ctrl_s0={ck}", None).engine == "stock"
    assert V.parse_job(f"main_wide1_s2={ck}", None).engine == "wide1"
    jb = V.parse_job(f"main_wide_s0_strip={ck}:strip", None)
    assert (jb.engine, jb.mode) == ("stock", "strip")
    jb = V.parse_job("untouched=none:base", None)
    assert (jb.engine, jb.mode, jb.ckpt) == ("stock", "base", None)
    assert V.parse_job(f"decay_wideg_s0={ck}", "wide").engine == "wide"   # no arm token: --engine as given
    assert raises(lambda: V.parse_job(f"decay_wideg_s0={ck}", None))      # ... and no --engine
    assert raises(lambda: V.parse_job(f"x={ck}:bogus", "wide"))
    assert raises(lambda: V.parse_job("main_wide_s0=/nonexistent.pt", None))
    a = V.parse_args(["--plan", "--root", root, "--names", "wide_s0,ctrl_s0,wide1_s0,wide_s0_strip,base"])
    assert V.plan(V.resolve_jobs(a)) == {"wide": ["wide_s0"], "wide1": ["wide1_s0"],
                                         "stock": ["ctrl_s0", "wide_s0_strip", "base"]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert V.main(["--plan", "--root", root, "--names", "wide_s0,ctrl_s0,base"]) == 0
    assert buf.getvalue().splitlines() == ["wide wide_s0", "stock ctrl_s0,base"], buf.getvalue()
    # red: a job whose engine is not --engine stops main() before vLLM is imported
    assert raises(lambda: V.main(["--engine", "wide", "--root", root, "--names", "wide_s0,ctrl_s0",
                                  "--model", "M", "--out", str(d / "o")]), V.VcountError)
    assert raises(lambda: V.resolve_jobs(V.parse_args(["--plan", "--root", root, "--names", "wide_s0",
                                                       "--job", f"wide_s0={ck}"])))  # duplicate name
    assert "vllm" not in sys.modules and "torch" not in sys.modules  # resolution and the refusal import neither
    print("jobs: ok")


def gate_keys(n_layers=24, gates=True, decay=False, lora=True):
    ks = []
    for i in range(n_layers):
        p = f"base_model.model.model.layers.{i}.linear_attn"
        if lora:
            for proj in ("in_proj_qkv", "in_proj_b", "in_proj_a", "out_proj"):
                ks += [f"{p}.{proj}.lora_A.default.weight", f"{p}.{proj}.lora_B.default.weight"]
        if gates:
            ks.append(f"{p}.gate_w.weight")
        if decay:
            ks.append(f"{p}.wg_wa.weight")
    return ks


def check_state():
    W, C = V.Job("wide_s0", "x", "trained", "wide"), V.Job("ctrl_s0", "x", "trained", "stock")
    r = V.check_state_keys(gate_keys(), W)
    assert r == {"gate_tensors": 24, "decay_tensors": 0, "lora_tensors": 192}, r
    assert V.check_state_keys(gate_keys(gates=False), C)["gate_tensors"] == 0
    assert V.check_state_keys(gate_keys(), V.Job("wide_s0_strip", "x", "strip", "stock"))["gate_tensors"] == 24
    assert raises(lambda: V.check_state_keys(gate_keys(), C))                       # wide ckpt scored as ctrl
    assert raises(lambda: V.check_state_keys(gate_keys(gates=False), W))            # ctrl ckpt on a gated engine
    assert raises(lambda: V.check_state_keys(gate_keys(gates=False), V.Job("s", "x", "strip", "stock")))
    assert raises(lambda: V.check_state_keys(gate_keys(lora=False), W))
    assert V.check_state_keys(gate_keys(decay=True), W, decay=True)["decay_tensors"] == 24
    assert raises(lambda: V.check_state_keys(gate_keys(), W, decay=True))           # --decay without wg_wa
    print("state keys: ok")


def lm_sample(i, L, score, doc_hash=None):
    return {"doc_id": i, "doc": {"input": "big"}, "target": ["a", "b"], "arguments": [["ctx", {}]],
            "resps": [["a b"]], "filtered_resps": ["a b"], "filter": "none", "metrics": [str(L)],
            "doc_hash": doc_hash or f"h{i}", "prompt_hash": f"p{i}", "target_hash": "t",
            str(L): float(score), "131072": -1.0}


def write_job(out: Path, job: str, L: int, scores: dict, limit: int, hashes: dict | None = None):
    jdir = out / job
    samples = {t: [lm_sample(i, L, s, (hashes or {}).get((t, i))) for i, s in enumerate(v)] for t, v in scores.items()}
    dig = V.write_samples(jdir / f"samples_ruler{L}", samples)
    V.save_json({"results": {t: {} for t in scores}, "vcount": {"tasks": list(scores), "limit": limit, "length": L,
                                                                "digest": dig}}, jdir / f"lmeval_ruler{L}.json")
    return jdir, dig


def check_layout(d: Path):
    out = d / "o"
    jdir, dig = write_job(out, "ctrl_s0", 4096, {"ruler_cwe": [1, 0, 0.5], "ruler_fwe": [0, 1, 1]}, 3)
    rows = [json.loads(x) for x in open(jdir / "samples_ruler4096" / "ruler_cwe.jsonl")]
    assert [r["doc_id"] for r in rows] == [0, 1, 2] and rows[2]["4096"] == 0.5, rows
    assert set(rows[0]) == {"doc_id", "target", "filtered_resps", "resps", "doc_hash", "prompt_hash", "4096",
                            "131072"}, set(rows[0])
    assert dig["ruler_cwe"] == V.digest_rows(rows)
    assert V.length_done(jdir, 4096, ["ruler_cwe", "ruler_fwe"], 3)
    assert not V.length_done(jdir, 4096, ["ruler_cwe", "ruler_fwe"], 500)       # other limit
    assert not V.length_done(jdir, 4096, ["ruler_cwe", "ruler_vt"], 3)          # task missing
    assert not V.length_done(jdir, 8192, ["ruler_cwe"], 3)
    (jdir / "samples_ruler4096" / "ruler_fwe.jsonl").unlink()
    assert not V.length_done(jdir, 4096, ["ruler_cwe", "ruler_fwe"], 3)          # samples missing
    # digests: same items pass, other items are refused
    write_job(out, "wide_s0", 4096, {"ruler_cwe": [0, 0, 0], "ruler_fwe": [1, 1, 1]}, 3)
    V.check_digests(out, 4096, json.load(open(out / "wide_s0" / "lmeval_ruler4096.json"))["vcount"]["digest"])
    _, bad = write_job(d / "o2", "x", 4096, {"ruler_cwe": [0, 0, 0]}, 3, hashes={("ruler_cwe", 1): "other"})
    assert raises(lambda: V.check_digests(out, 4096, bad), V.VcountError)
    print("layout: ok")


def check_greedy():
    class FakeVLLM:
        @staticmethod
        def modify_gen_kwargs(gen_kwargs, eos=None, default_max_gen_toks=256):
            kw = {k: v for k, v in gen_kwargs.items() if k not in ("do_sample", "until", "max_gen_toks")}
            return kw, list(gen_kwargs.get("until", [])), gen_kwargs.get("max_gen_toks", default_max_gen_toks)

    kw, _, _ = FakeVLLM.modify_gen_kwargs({"do_sample": False, "max_gen_toks": 120})
    assert "temperature" not in kw  # red: unfixed, vLLM would use its default temperature 1.0
    V.GREEDY.update(requests=0, sampled=0)
    V.greedy_fix(FakeVLLM)
    V.greedy_fix(FakeVLLM)  # idempotent
    kw, _, mx = FakeVLLM.modify_gen_kwargs({"do_sample": False, "max_gen_toks": 120})
    assert kw["temperature"] == 0.0 and mx == 120, kw
    kw, _, _ = FakeVLLM.modify_gen_kwargs({"do_sample": True, "temperature": 0.7})
    assert kw["temperature"] == 0.7 and V.GREEDY == {"requests": 2, "sampled": 1}, (kw, V.GREEDY)
    print("greedy: ok")


def paired(root, out, *extra):
    r = subprocess.run([sys.executable, str(PAIRED), str(root), str(out), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.load(open(out))


def check_paired(d: Path):
    root = d / "p"
    n = 40
    base_c = [1.0] * 20 + [0.0] * 20
    wide_c = [1.0] * 30 + [0.0] * 10          # +0.25 on cwe (10 better, 0 worse)
    base_f = [0.5] * n
    wide_f = [0.5] * n                         # 0 on fwe
    write_job(root, "ctrl_s0", 4096, {"ruler_cwe": base_c, "ruler_fwe": base_f}, n)
    write_job(root, "wide_s0", 4096, {"ruler_cwe": wide_c, "ruler_fwe": wide_f}, n)
    o = paired(root, d / "p.json", "--base", "ctrl_s0", "--arms", "wide_s0", "--lengths", "4096")
    rows = {r["task"]: r for r in o["rows"]}
    assert o["base"] == "ctrl_s0"
    c = rows["ruler_cwe"]
    assert (c["n"], c["stock"], c["arm_mean"], c["diff"], c["better"], c["worse"]) == (40, 0.5, 0.75, 0.25, 10, 0), c
    assert c["hash_mismatch"] == 0 and c["p"] < 0.01, c
    assert rows["ruler_fwe"]["diff"] == 0.0 and rows["ruler_fwe"]["p"] == 1.0
    assert len(o["macro"]) == 1 and o["macro"][0]["diff"] == 0.125, o["macro"]
    # mismatched items are dropped and counted
    write_job(root, "wide1_s0", 4096, {"ruler_cwe": wide_c, "ruler_fwe": wide_f}, n,
              hashes={("ruler_cwe", 25): "zz", ("ruler_cwe", 26): "zz", ("ruler_cwe", 27): "zz"})
    o = paired(root, d / "p1.json", "--base", "ctrl_s0", "--arms", "wide1_s0", "--lengths", "4096")
    c = {r["task"]: r for r in o["rows"]}["ruler_cwe"]
    assert (c["n"], c["hash_mismatch"], c["better"]) == (37, 3, 7), c
    assert o["macro"] == [{"L": 4096, "arm": "wide1_s0", "tasks": 2, "skipped": "tasks differ in n", "n": [37, 40]}], o
    # red: without --base the reference is "stock", absent here: nothing is paired
    o = paired(root, d / "p2.json", "--arms", "wide_s0", "--lengths", "4096")
    assert o["rows"] == [] and o["macro"] == [] and "base" not in o
    # default behaviour unchanged on an Olmo-style tree (no doc_hash in the dumps)
    olmo = d / "olmo"
    rng = np.random.default_rng(1)
    S = {t: rng.integers(0, 2, 30).astype(float) for t in ("niah_single_1", "ruler_cwe")}
    H = {t: np.clip(S[t] - (rng.random(30) < 0.3), 0, 1) for t in S}
    for arm, src in (("stock", S), ("half", H)):
        sd = olmo / arm / "samples_ruler8192"
        sd.mkdir(parents=True)
        for t, v in src.items():
            with open(sd / f"{t}.jsonl", "w") as f:
                for i, x in enumerate(v):
                    f.write(json.dumps({"doc_id": i, "target": [], "filtered_resps": [], "resps": [], "4096": -1.0,
                                        "8192": float(x)}) + "\n")
    o = paired(olmo, d / "olmo.json", "--arms", "half", "--lengths", "8192")
    assert set(o) == {"root", "rows", "macro"}, set(o)
    for r in o["rows"]:
        assert "hash_mismatch" not in r and set(r) == {"L", "arm", "task", "n", "stock", "arm_mean", "diff", "p",
                                                       "better", "worse"}, r
        t = r["task"]
        assert r["stock"] == round(float(S[t].mean()), 4) and r["diff"] == round(float((H[t] - S[t]).mean()), 4), r
    print("paired: ok")


def check_lmeval(d: Path, model: str):
    """End to end through lm-eval with a fake LM (no engine)."""
    import lm_eval.api.model as LMM

    class FakeLM(LMM.LM):
        def generate_until(self, requests, disable_tqdm=False):
            return [" " + ", ".join(r.doc["outputs"]) if r.doc_id % 2 == 0 else " qqqq" for r in requests]

        def loglikelihood(self, requests, disable_tqdm=False):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError

    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    out = d / "e2e"
    tasks, L, limit = list(V.TASKS), 4096, 20
    ra = V.run_ruler(FakeLM(), tasks, out / "a", L, limit, model)
    rb = V.run_ruler(FakeLM(), tasks, out / "b", L, limit, model)
    assert ra["vcount"]["digest"] == rb["vcount"]["digest"], (ra["vcount"], rb["vcount"])
    V.check_digests(out, L, rb["vcount"]["digest"])
    for t in tasks:
        rows = [json.loads(x) for x in open(out / "a" / f"samples_ruler{L}" / f"{t}.jsonl")]
        assert len(rows) == limit and [r["doc_id"] for r in rows] == list(range(limit)), t
        sc = [r[str(L)] for r in rows]
        assert all(s == 1.0 for s in sc[::2]) and all(s == 0.0 for s in sc[1::2]), (t, sc)
        assert all(r["doc_hash"] and r["prompt_hash"] for r in rows)
    assert V.length_done(out / "a", L, tasks, limit)
    a = V.parse_args(["--data_only", "--model", model, "--out", str(out / "data"), "--lengths", str(L),
                      "--limit", str(limit)])
    with contextlib.redirect_stdout(io.StringIO()):
        V.data_only(a)
    dj = json.load(open(out / "data" / "data_only.json"))["lengths"][str(L)]["tasks"]
    assert {t: v["digest"] for t, v in dj.items()} == ra["vcount"]["digest"], (dj, ra["vcount"])
    # red: without the reset, a later build in the same process gets other ruler_cwe items
    orig = V.reset_ruler_state
    V.reset_ruler_state = lambda: False
    try:
        rc = V.run_ruler(FakeLM(), tasks, out / "c", L, limit, model)
    finally:
        V.reset_ruler_state = orig
    assert rc["vcount"]["digest"]["ruler_cwe"] != ra["vcount"]["digest"]["ruler_cwe"], rc["vcount"]
    assert rc["vcount"]["digest"]["ruler_fwe"] == ra["vcount"]["digest"]["ruler_fwe"]  # fwe seeds its own RNG
    assert raises(lambda: V.check_digests(out, L, rc["vcount"]["digest"]), V.VcountError)
    print("lmeval e2e: ok (layout, same items across jobs, data_only digest, red: no reset -> cwe items differ)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmeval", action="store_true", help="also run the lm-eval end-to-end check (~5 min CPU)")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR_VCOUNT")) as td:
        d = Path(td)
        check_args()
        check_jobs(d / "jobs")
        check_state()
        check_layout(d / "layout")
        check_greedy()
        check_paired(d / "paired")
        if a.lmeval:
            check_lmeval(d / "lmeval", a.model)
    print("ALL OK")
