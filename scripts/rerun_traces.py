#!/usr/bin/env python3
"""Rerun the CPU analyses that read the generation traces and compare their output with the released receipts.

The traces (*.traces.jsonl.gz) are not in the repository. They are a separate archive (see README.md, "Generation
traces"); extract it at the repository root, then run this script.

  text_stats.py      4B step 12 validation and step 14 test traces -> results/mech/text_stats.json
  answer_cap.py      step 9 curves at answer cap 4,096 -> results/ml7_4b/answer_cap.json (4B), results/ml7/answer_cap.json (2B)
  truncate_curve.py  the same curves cut and regraded at the chosen caps -> results/ml7_4b/step9_curve_ans1280.json,
                     results/ml7/step9_curve_ans1792_Qwen3.5-2B_gb10.json
  build_inputs.py    G3 teacher-forcing inputs from the 4B validation traces -> the "fixed" set has the 64 sequences and
                     99,769 reasoning tokens of results/mech/gate_stats/*__fixed.json; the file's sha256 equals the lab's
  example_traces.py  the paired example traces of the implementation appendix -> paper/data/example_traces.txt
                     (byte for byte; checkouts from v1.1.0 on)

Each script runs unchanged in a scratch directory. text_stats.py and example_traces.py get uncompressed copies of the
readouts they read (the repository ships them gzipped); truncate_curve.py gets a receipt copy whose args.data points at
the rebuilt data file, and that one field is put back before the comparison. Everything else must be equal.

Setup:  python scripts/rebuild_data.py --skip_fresh     (val.jsonl, test_gsm8k.jsonl, test_math500.jsonl)
Usage:  python scripts/rerun_traces.py [--verify]       (--verify first checks every trace file against
                                                          results/TRACES_LEDGER.json)
Needs:  text_stats.py runs in the CPU environment (uv sync). answer_cap.py also needs transformers, example_traces.py
        jinja2; truncate_curve.py and build_inputs.py import rl_common.py, so they need torch, transformers and
        math-verify (env/train.txt pins them). A script whose imports are missing is reported as SKIP. Tokenizers are downloaded to WG_HF_CACHE at the
        pinned revisions if absent. Exit 1 on any difference or failure, 2 if the archive or the data are missing.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SRC = ROOT / "src/gdn_gate"
LEDGER = RES / "TRACES_LEDGER.json"
QWEN4 = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
QWEN2 = ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc")
TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json", "chat_template.jinja", "special_tokens_map.json",
             "vocab.json", "merges.txt"]
# sha256 of the inputs.json the lab built with build_inputs.py (the G3 input the gate_stats receipts were read on)
LAB_INPUTS_SHA = "32e793087fd6ea12a9b36d97398e75c57fde4443a083eea0407ad533c2e29b76"
HF = os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
RUNS = Path(os.environ.get("WG_RUNS", str(ROOT / "runs")))
DATA = RUNS / "ml7/data"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(args, cwd, env=None):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, "WG_HF_CACHE": HF, "WG_RUNS": str(RUNS), "MPLBACKEND": "Agg",
                            "CUDA_VISIBLE_DEVICES": "", **(env or {})})
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def has(*mods) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def tokenizer_dir(repo_rev) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    d = None
    for f in TOK_FILES:
        try:
            d = os.path.dirname(hf_hub_download(repo_rev[0], f, revision=repo_rev[1], cache_dir=HF))
        except EntryNotFoundError:
            pass
    return d


def gunzip_readouts(dst: Path, sub: str):
    """Uncompressed readouts of results/<sub> plus links to their trace files, as the lab tree held them."""
    (dst / sub).mkdir(parents=True, exist_ok=True)
    for p in sorted((RES / sub).glob("readout_*.json.gz")):
        (dst / sub / p.name[:-3]).write_bytes(gzip.open(p).read())
    for p in sorted((RES / sub).glob("*.traces.jsonl.gz")):
        os.symlink(p, dst / sub / p.name)


def stage_receipt(dst: Path, src: Path, name: str, trace: Path, data_file: Path | None = None):
    rec = json.loads(src.read_text())
    if data_file is not None:
        rec["args"]["data"] = str(data_file)
    (dst / name).write_text(json.dumps(rec))
    os.symlink(trace, dst / (name + ".traces.jsonl.gz"))


def report(status, what, log=""):
    print(f"{status:4s} {what}" + (f"\n{log}" if log and status in ("FAIL", "DIFF") else ""))
    return status in ("FAIL", "DIFF")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check every trace file against results/TRACES_LEDGER.json")
    a = ap.parse_args()
    if not LEDGER.exists():
        sys.exit(f"{LEDGER.relative_to(ROOT)} not found: extract the trace archive at the repository root first")
    led = json.loads(LEDGER.read_text())
    missing = [f["path"] for f in led["files"] if not (ROOT / f["path"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} trace files missing, e.g. {missing[0]}")
    if a.verify:
        badsha = [f["path"] for f in led["files"] if sha256(ROOT / f["path"]) != f["sha256"]]
        print(f"{'OK  ' if not badsha else 'DIFF'} {len(led['files'])} trace files against TRACES_LEDGER.json"
              + (f": {badsha[:5]}" if badsha else ""))
        if badsha:
            sys.exit(1)
    want = json.loads((ROOT / "ml_pipeline/split_manifest.json").read_text())["files"]
    for f in ("val.jsonl", "test_gsm8k.jsonl", "test_math500.jsonl"):
        if not (DATA / f).exists() or sha256(DATA / f) != want[f]:
            sys.exit(f"{DATA / f} missing or not the frozen file: run python scripts/rebuild_data.py --skip_fresh")

    bad = 0
    tmp = Path(tempfile.mkdtemp(prefix="rerun_traces_"))
    try:
        tok4 = tokenizer_dir(QWEN4)

        # where the reasoning tokens go (4B validation and test traces)
        if not has("numpy", "matplotlib", "tokenizers"):
            report("SKIP", "results/mech/text_stats.json (text_stats.py needs numpy, matplotlib, tokenizers)")
        else:
            w = tmp / "ts"
            for sub in ("step12_box", "step12_ctrlpm", "step14", "step14_ctrlpm"):
                gunzip_readouts(w / "results", f"ml7_4b/{sub}")
            rc, log = run([SRC / "mech/text_stats.py", w / "out"], w)
            st = ("OK  " if json.loads((w / "out/text_stats.json").read_text())
                  == json.loads((RES / "mech/text_stats.json").read_text()) else "DIFF") if rc == 0 else "FAIL"
            bad += report(st, "results/mech/text_stats.json (text_stats.py)", log)

        tok2 = None
        cases = [  # (label, curve receipt, its trace file, name the lab ran it under, tokenizer repo, answer cap file,
                   #  truncated curve file, cap, receipt path given to truncate_curve.py)
            ("4B", RES / "ml7_4b/step9_curve_ans4096.json", RES / "ml7_4b/step9_curve_ans4096.json.traces.jsonl.gz",
             "step9_curve_ans4096.json", QWEN4, RES / "ml7_4b/answer_cap.json", RES / "ml7_4b/step9_curve_ans1280.json",
             1280, "step9_curve_ans4096.json"),
            ("2B", RES / "ml7/step9_curve_ans4096_Qwen3.5-2B_gb10.json",
             RES / "ml7/step9_curve_ans4096_Qwen3.5-2B_gb10.traces.jsonl.gz", "c.json", QWEN2, RES / "ml7/answer_cap.json",
             RES / "ml7/step9_curve_ans1792_Qwen3.5-2B_gb10.json", 1792, "../c.json"),
        ]
        # answer cap from the uncensored step 9 curve
        for label, rec, trace, name, repo, cap_file, _, _, _ in cases:
            if not has("transformers"):
                report("SKIP", f"{cap_file.relative_to(ROOT)} (answer_cap.py needs transformers)")
                continue
            tdir = tok4 if repo == QWEN4 else (tok2 := tok2 or tokenizer_dir(QWEN2))
            d = tmp / f"cap_{label}"
            d.mkdir()
            stage_receipt(d, rec, name, trace)
            rc, log = run([SRC / "ml7/answer_cap.py", name, tdir, d / "out.json"], d)
            st = ("OK  " if json.loads((d / "out.json").read_text()) == json.loads(cap_file.read_text()) else "DIFF") \
                if rc == 0 else "FAIL"
            bad += report(st, f"{cap_file.relative_to(ROOT)} (answer_cap.py, {label})", log)
        # the step 9 curve at the chosen answer cap: answers cut and regraded
        for label, rec, trace, name, repo, _, cut_file, cap, rec_arg in cases:
            if not has("transformers", "torch", "math_verify"):
                report("SKIP", f"{cut_file.relative_to(ROOT)} (truncate_curve.py needs torch, transformers, math-verify)")
                continue
            tdir = tok4 if repo == QWEN4 else (tok2 := tok2 or tokenizer_dir(QWEN2))
            d = tmp / f"cut_{label}"
            (d / "run").mkdir(parents=True)
            orig_data = json.loads(rec.read_text())["args"]["data"]
            stage_receipt(d / "run" if rec_arg == name else d, rec, name, trace, DATA / Path(orig_data).name)
            rc, log = run([SRC / "ml7/truncate_curve.py", rec_arg, tdir, cap, d / "out.json"], d / "run")
            st = "FAIL"
            if rc == 0:
                out = json.loads((d / "out.json").read_text())
                out["args"]["data"] = orig_data
                st = "OK  " if out == json.loads(cut_file.read_text()) else "DIFF"
            bad += report(st, f"{cut_file.relative_to(ROOT)} (truncate_curve.py, {label})", log)
        # G3 teacher-forcing inputs
        if not has("transformers", "torch", "math_verify"):
            report("SKIP", "G3 inputs (build_inputs.py needs torch, transformers, math-verify)")
        else:
            d = tmp / "g3"
            d.mkdir()
            rc, log = run([SRC / "mech/build_inputs.py", d / "inputs.json"], ROOT)
            st = "FAIL"
            if rc == 0:
                fixed = json.loads((d / "inputs.json").read_text())["sets"]["fixed"]
                gs = json.loads((RES / "mech/gate_stats/warmup__fixed.json").read_text())
                ok = (len(fixed) == gs["n_seq"] and sum(len(x["think"]) for x in fixed) == gs["n_tokens"]
                      and sha256(d / "inputs.json") == LAB_INPUTS_SHA)
                st = "OK  " if ok else "DIFF"
                log = f"fixed: {len(fixed)} sequences, {sum(len(x['think']) for x in fixed)} tokens; sha256 {sha256(d / 'inputs.json')}"
            bad += report(st, "G3 inputs: 64 sequences, 99,769 tokens, lab sha256 (build_inputs.py)", log)
        # paired example traces of the implementation appendix
        ex = ROOT / "paper/figs/example_traces.py"
        if ex.exists() and not has("tokenizers", "jinja2"):
            report("SKIP", "paper/data/example_traces.txt (example_traces.py needs tokenizers, jinja2)")
        elif ex.exists():
            d = tmp / "ex"
            (d / "paper/figs").mkdir(parents=True)
            (d / "paper/data").mkdir()
            shutil.copyfile(ex, d / "paper/figs/example_traces.py")
            gunzip_readouts(d / "results", "ml7_4b/step14")
            rc, log = run([d / "paper/figs/example_traces.py"], d)
            st = ("OK  " if (d / "paper/data/example_traces.txt").read_bytes()
                  == (ROOT / "paper/data/example_traces.txt").read_bytes() else "DIFF") if rc == 0 else "FAIL"
            bad += report(st, "paper/data/example_traces.txt (example_traces.py)", log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
