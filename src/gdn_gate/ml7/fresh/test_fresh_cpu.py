#!/usr/bin/env python3
"""test_fresh_cpu.py: CPU tests for the Phase 5 fresh-set tooling (build_fresh.py, fresh_analysis.py, fresh_guard.py,
fresh_test_box.sh), each with a red arm. Synthetic readouts only (schema wg-readout-v1 rows built here); no model,
no GPU, no read of the fresh set by any model.

  F0  frozen code: step14_analysis.py, ctrlpm_analysis.py and the readout code equal the 4B touch-2 lock sha256
  F1  dedup: build_fresh.sibling_scan finds a planted near-duplicate (a val item reworded) and none among unrelated items
  F2  analysis 4b: gated arms better and shorter -> H5 and H6 supported; red arm (wide1 = ctrl rows) -> not supported
  F3  analysis refusals: mixed GPU, data sha256 not the fresh file, ctrlpm with rank 16
  F4  analysis 2b: 6 seeds, no warm-up, H5/H6 rows marked secondary; one seed flipped -> no effect (6-seed rule)
  F5  per-source blocks are descriptive and cover the four sources
  F6  guard preflight: OK on a complete plan; refuses without the pre-registration block, with a wrong checkpoint
      sha256, with a partial 2b block, after a touch
  F7  guard final: a fake runner writes the readouts and analyses; touch recorded once (1); a second final refused
  F8  runner: fresh_test_box.sh plan lists 33 labels (14 with FRESH_RECORDS=4b) and PHASE=one refuses outside the guard
Usage: python test_fresh_cpu.py  (exit 0 = all pass)"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML7 = HERE.parent
WT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ML7))
import build_fresh as BF  # noqa: E402
import fresh_analysis as FA  # noqa: E402
import fresh_guard as FG  # noqa: E402

CAPS = [256, 512, 1024, 2048, 4096, 8192]
GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
FAILS = []


def check(name, ok, why=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
    if not ok:
        FAILS.append(name)


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def ids_fresh():
    out = []
    for src, n in (("aime2024", 30), ("aime2025", 30), ("aime2026", 30), ("amc2023", 40)):
        for k in range(n):
            out.append(f"fresh:{src}-{'I' if k < n // 2 else 'II'}-{k % (n // 2) + 1}")
    return out


def readout(arm, seed, ids, quality, length, data_sha, *, rank=16, ckpt="none", gpu=GPU, samples=8, amax=1280,
            probe=0.05, rng_seed=0):
    rng = random.Random(f"{arm}:{seed}:{rng_seed}")
    rows = []
    for i in ids:
        p_item = rng.random()
        for s in range(samples):
            tl = max(50, int(rng.gauss(length, length * 0.3)))
            closed = tl <= 9000
            caps = {}
            good = rng.random() < quality * (0.5 + p_item)
            for c in CAPS:
                within = closed and tl <= c
                caps[str(c)] = [int(good and (within or rng.random() < 0.2)), min(tl, c), int(not within), 1, 1, 100]
            rows.append({"id": i, "sample": s, "think_len": tl, "closed": int(closed), "silent_stop": 0,
                         "phase1_tokens": tl + 100, "caps": caps})
    return {"schema": "wg-readout-v1", "arm": arm, "ckpt": ckpt, "red": "", "caps": CAPS,
            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
            "args": {"seed": 20261006, "samples": samples, "answer_max": amax, "caps": ",".join(map(str, CAPS)),
                     "rank": rank, "probe_effect": probe},
            "env": {"gpu": gpu, "sha256": {"data": data_sha, "ckpt": None if ckpt == "none" else sha_text(ckpt)}},
            "rows": rows}


def write(d, p):
    Path(p).write_text(json.dumps(d))
    return str(p)


def set_4b(tmp, data_sha, red=False, mutate=None):
    ids = ids_fresh()
    files, slots = {}, []
    files["base"] = write(readout("stock", 0, ids, 0.30, 7000, data_sha, rng_seed=1), tmp / "u.json")
    files["warm"] = write(readout("ctrl", 0, ids, 0.32, 6800, data_sha, ckpt="/x/ctrl_s0_cap512_fmt_box/ckpt_step40.pt",
                                  rng_seed=2), tmp / "w.json")
    spec = {"ctrl": (0.35, 6000), "ctrlpm": (0.36, 5800), "wide": (0.50, 2500), "wide1": (0.50, 2600)}
    for a, (q, ln) in spec.items():
        for s in range(3):
            arm = "ctrl" if a == "ctrlpm" else a
            qq, ll = (spec["ctrl"] if (red and a == "wide1") else (q, ln))
            d = readout(arm, s, ids, qq, ll, data_sha, rank=20 if a == "ctrlpm" else 16,
                        ckpt=f"/x/{a}_s{s}_cap512_r_box/ckpt_step150.pt", rng_seed=(10 if (red and a == "wide1") else 3))
            if red and a == "wide1":  # identical rows to ctrl seed s (red arm: no difference at all)
                d["rows"] = readout("ctrl", s, ids, *spec["ctrl"], data_sha, rng_seed=3)["rows"]
            if mutate:
                mutate(a, s, d)
            slots.append(f"{a}_{s}=" + write(d, tmp / f"{a}_{s}.json"))
    return files, slots


def manifest(tmp, data_sha):
    p = tmp / "split_manifest.json"
    p.write_text(json.dumps({"files": {"test_fresh.jsonl": data_sha}}))
    return str(p)


def run_fa(argv):
    import importlib

    for m in ("step14_analysis", "ctrlpm_analysis", "fresh_analysis"):
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    import fresh_analysis  # noqa: PLC0415

    ns = None
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--record"); ap.add_argument("--baseline"); ap.add_argument("--warmup", default=None)
    ap.add_argument("slots", nargs="+"); ap.add_argument("--manifest"); ap.add_argument("--out")
    ap.add_argument("--seeds", default=None); ap.add_argument("--pm_rank", type=int, default=20)
    ns = ap.parse_args(argv)
    return fresh_analysis.run(ns)


def refuses(argv):
    try:
        run_fa(argv)
        return False
    except SystemExit:
        return True


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fresh_cpu_", dir=os.environ.get("TMPDIR_TEST", os.path.expanduser("~/.cache/gdn_gate/p5"))))
    # F0
    lock = {Path(ln.split()[1]).name: ln.split()[0] for ln in
            open(WT / "results/ml7_4b/step14_ctrlpm/code.sha256") if ln.strip()}
    files = {"readout_vllm.py": ML7 / "readout_vllm.py", "train_grpo_vllm.py": ML7 / "train_grpo_vllm.py",
             "step14_analysis.py": ML7 / "step14_analysis.py", "ctrlpm_analysis.py": ML7 / "ctrlpm_analysis.py",
             "gate_vllm.py": ML7.parent / "vllm/gate_vllm.py", "rl_common.py": ML7.parent / "broad/rl_common.py",
             "arms.py": ML7.parent / "broad/arms.py", "packing.py": ML7.parent / "broad/packing.py",
             "guard.py": WT / "ml_pipeline/guard.py"}
    # Published copies were path-scrubbed: a file also passes when its sha256 is the published hash that
    # results/SCRUB_LEDGER.json records for the locked lab hash.
    pub = {f["sha256_original"]: f.get("sha256_released") for f in
           json.loads((WT / "results/SCRUB_LEDGER.json").read_text())["files"] if f.get("sha256_original")}
    diff = [n for n, p in files.items() if hashlib.sha256(p.read_bytes()).hexdigest() not in (lock[n], pub.get(lock[n]))]
    # readout_vllm.py may differ by owner-approved ledger amendments only (2026-10-12 effect_visible, fresh set)
    if "readout_vllm.py" in diff and subprocess.run([sys.executable, str(ML7 / "code_ledger.py"), "accept", "--file",
                                                     str(files["readout_vllm.py"]), "--want", lock["readout_vllm.py"]],
                                                    capture_output=True).returncode == 0:
        diff.remove("readout_vllm.py")
    check("F0 frozen code equals the 4B touch-2 lock (readout_vllm.py: or its ledger chain)", not diff, diff)
    # F1
    val = [json.loads(x) for x in open(os.path.join(os.environ.get("WG_RUNS", "runs"), "ml7/data/val.jsonl"))][:50]
    planted = {"id": "fresh:plant-I-1", "problem": val[7]["problem"].replace("How many", "how many") + " ", "answer": "1"}
    other = {"id": "fresh:other-I-1", "problem": "Find the number of ordered pairs of positive integers (a, b) with "
             "a + b = 1000 such that neither a nor b has a zero digit.", "answer": "738"}
    hits = BF.sibling_scan([planted, other], val)
    check("F1 dedup finds the planted near-duplicate", any(h["left_id"] == planted["id"] for h in hits), hits)
    check("F1 red: no hit for an unrelated item", not any(h["left_id"] == other["id"] for h in hits), hits)
    dsha = sha_text("fresh-data")
    man = manifest(tmp, dsha)
    # F2
    f, sl = set_4b(tmp, dsha)
    res = run_fa(["--record", "4b", "--baseline", f["base"], "--warmup", f["warm"], *sl, "--manifest", man, "--out", str(tmp / "o.json")])
    h = res["preregistered_hypotheses"]
    check("F2 H5 supported on gated-better data", h["H5"]["supported"] is True, h["H5"].get("verdict"))
    check("F2 H6 supported on gated-shorter data", h["H6"]["supported"] is True, h["H6"].get("verdict"))
    check("F2 H1-H4 not applicable on the fresh set", all(not h[k]["applies"] for k in ("H1", "H2", "H3", "H4")))
    check("F2 four arms incl. ctrlpm, contrast wide1 - ctrlpm present", "wide1 - ctrlpm" in res["effects"], list(res["effects"]))
    r2 = tmp / "red"; r2.mkdir()
    f, sl = set_4b(r2, dsha, red=True)
    res = run_fa(["--record", "4b", "--baseline", f["base"], "--warmup", f["warm"], *sl, "--manifest", man, "--out", str(r2 / "o.json")])
    h = res["preregistered_hypotheses"]
    check("F2 red: identical wide1 and ctrl -> H5 not supported", h["H5"]["supported"] is False, h["H5"].get("verdict"))
    check("F2 red: identical wide1 and ctrl -> H6 not supported", h["H6"]["supported"] is False, h["H6"].get("verdict"))
    # F3
    r3 = tmp / "gpu"; r3.mkdir()
    f, sl = set_4b(r3, dsha, mutate=lambda a, s, d: d["env"].__setitem__("gpu", "NVIDIA GB10") if (a, s) == ("wide", 1) else None)
    check("F3 refuses mixed GPUs", refuses(["--record", "4b", "--baseline", f["base"], "--warmup", f["warm"], *sl, "--manifest", man, "--out", str(r3 / "o.json")]))
    r4 = tmp / "sha"; r4.mkdir()
    f, sl = set_4b(r4, sha_text("val"))
    check("F3 refuses data that is not the fresh file", refuses(["--record", "4b", "--baseline", f["base"], "--warmup", f["warm"], *sl, "--manifest", man, "--out", str(r4 / "o.json")]))
    r5 = tmp / "rank"; r5.mkdir()
    f, sl = set_4b(r5, dsha, mutate=lambda a, s, d: d["args"].__setitem__("rank", 16) if a == "ctrlpm" else None)
    check("F3 refuses a ctrlpm readout at rank 16", refuses(["--record", "4b", "--baseline", f["base"], "--warmup", f["warm"], *sl, "--manifest", man, "--out", str(r5 / "o.json")]))
    # F4
    r6 = tmp / "2b"; r6.mkdir()
    ids = ids_fresh()
    base = write(readout("stock", 0, ids, 0.2, 7000, dsha, amax=1792, probe=0.2, rng_seed=1), r6 / "u.json")
    sl = []
    for a, (q, ln) in {"ctrl": (0.25, 6000), "wide": (0.4, 3000), "wide1": (0.4, 3100)}.items():
        for s in range(6):
            sl.append(f"{a}_{s}=" + write(readout(a, s, ids, q, ln, dsha, amax=1792, probe=0.2,
                                                  ckpt=f"/y/{a}_s{s}_cap512_r_x/ckpt_step150.pt", rng_seed=4), r6 / f"{a}_{s}.json"))
    res = run_fa(["--record", "2b", "--baseline", base, *sl, "--manifest", man, "--out", str(r6 / "o.json")])
    h = res["preregistered_hypotheses"]
    check("F4 2b H5 supported, marked secondary", h["H5"]["supported"] and "secondary" in h["H5"]["role"], h["H5"])
    check("F4 2b uses 6 seeds", len(h["H5"]["per_seed_diff"]) == 6, h["H5"].get("per_seed_diff"))
    # red: seed 5 of wide1 reads as ctrl seed 5 -> one seed has difference 0 -> no effect by the 6-seed rule
    d = json.loads(Path(r6 / "ctrl_5.json").read_text()); d["arm"] = "wide1"; d["env"]["sha256"]["ckpt"] = sha_text("w1s5")
    write(d, r6 / "wide1_5.json")
    res = run_fa(["--record", "2b", "--baseline", base, *sl, "--manifest", man, "--out", str(r6 / "o2.json")])
    check("F4 red: one seed without a difference -> H5 not supported", res["preregistered_hypotheses"]["H5"]["supported"] is False,
          res["preregistered_hypotheses"]["H5"].get("per_seed_diff"))
    # F5
    ps = res["per_source_descriptive"]
    check("F5 per-source blocks for the four sources, descriptive", sorted(ps) == sorted(FA.SOURCES) and all("descriptive" in v for v in ps.values()), list(ps))
    # F6 / F7: guard on a scratch copy of the record and data
    g = tmp / "guard"; (g / "ml").mkdir(parents=True); (g / "data").mkdir()
    rows = [{"id": i, "problem": f"problem {i}", "answer": str(k)} for k, i in enumerate(ids)]
    (g / "data" / "test_fresh.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fsha = FG.sha(g / "data" / "test_fresh.jsonl")
    gd = g / "ml" / "fresh_test"; gd.mkdir()
    (gd / "split_manifest.json").write_text(json.dumps({"files": {"test_fresh.jsonl": fsha}}))
    guard = FG.import_guard()
    (gd / guard.STATE_FILE).write_text(json.dumps({"test_fingerprint": guard._fingerprint(FG.frame(g / "data")),
                                                   "n_test": len(rows), "test_touches": 0}))
    md = g / "ml" / "PIPELINE.md"
    md.write_text("# record\n\n## Fresh test set (pre-registration, test)\n")
    ck = g / "ck"; expect = {}
    plan = []
    for lab in FG.LABELS["4b"] + FG.LABELS["2b"]:
        if lab.endswith("untouched"):
            plan.append({"label": lab, "record": lab[:2], "arm": "stock", "ckpt": "none"})
            continue
        rd = ck / lab; rd.mkdir(parents=True)
        step = 40 if lab.endswith("warmup") else 150
        p = rd / f"ckpt_step{step}.pt"; p.write_bytes(lab.encode())
        (rd / "DONE").write_text(json.dumps({"steps": step}))
        expect[lab] = {"path": str(p), "sha256": FG.sha(p)}
        arm = lab.split("_")[1]
        plan.append({"label": lab, "record": lab[:2], "arm": {"warmup": "ctrl", "ctrlpm": "ctrl"}.get(arm, arm), "ckpt": str(p)})
    res_dir = g / "res"; res_dir.mkdir()
    for r in plan:
        rec = r["record"]
        r.update(out=str(res_dir / f"readout_fresh_{r['label']}.json"), model="m", answer_max="1280" if rec == "4b" else "1792",
                 probe="0.05" if rec == "4b" else "0.2", rank="20" if "ctrlpm" in r["label"] else "16")
    planp = g / "plan.tsv"
    planp.write_text("".join("\t".join(r[k] for k in FG.KEYS) + "\n" for r in plan))
    ok = FG.preflight_reasons(gd, md, g / "data", FG.read_plan(planp), expect)
    check("F6 preflight OK on the complete plan", not ok, ok)
    md2 = g / "ml" / "NOPRE.md"; md2.write_text("# record\n")
    check("F6 red: refuses without the pre-registration block", any("pre-registration" in x for x in FG.preflight_reasons(gd, md2, g / "data", FG.read_plan(planp), expect)))
    bad_exp = copy.deepcopy(expect); bad_exp["4b_wide1_1"]["sha256"] = "0" * 64
    check("F6 red: refuses a wrong checkpoint sha256", any("4b_wide1_1" in x for x in FG.preflight_reasons(gd, md, g / "data", FG.read_plan(planp), bad_exp)))
    part = [r for r in FG.read_plan(planp) if r["label"] != "2b_wide_4"]
    check("F6 red: refuses a partial 2b block", any("2b block" in x for x in FG.preflight_reasons(gd, md, g / "data", part, expect)))
    # F7: fake runner (readouts with the planned args, analyses as stubs)
    fake = g / "fake_runner.py"
    fake.write_text(f"""import json, os, sys, hashlib
plan = [dict(zip({FG.KEYS!r}, l.split('\\t'))) for l in open({str(planp)!r}).read().splitlines() if l.strip()]
ids = [json.loads(x)['id'] for x in open({str(g / 'data' / 'test_fresh.jsonl')!r})]
dsha = hashlib.sha256(open({str(g / 'data' / 'test_fresh.jsonl')!r}, 'rb').read()).hexdigest()
ph = os.environ['PHASE']
if ph == 'readouts':
    for r in plan:
        d = {{'arm': r['arm'], 'red': '', 'args': {{'caps': '256,512,1024,2048,4096,8192', 'seed': 20261006,
             'answer_max': int(r['answer_max']), 'samples': 8, 'rank': int(r['rank']), 'probe_effect': float(r['probe'])}},
             'env': {{'sha256': {{'data': dsha}}}}, 'rows': [{{'id': i, 'sample': s}} for i in ids for s in range(8)]}}
        open(r['out'], 'w').write(json.dumps(d))
elif ph == 'analysis':
    for rec in ('4b', '2b'):
        open(os.path.join({str(res_dir)!r}, 'analysis_fresh_' + rec + '.json'), 'w').write('{{}}')
""")
    runner = g / "runner.sh"; runner.write_text(f"#!/usr/bin/env bash\nexec {sys.executable} {fake}\n")
    argv = [sys.executable, str(HERE / "fresh_guard.py"), "final", "--gd", str(gd), "--md", str(md), "--data", str(g / "data"),
            "--plan", str(planp), "--samples", "8", "--runner", str(runner), "--res", str(res_dir), "--out", str(res_dir / "final.json")]
    p1 = subprocess.run(argv, capture_output=True, text=True)
    st = json.loads((gd / guard.STATE_FILE).read_text())
    check("F7 final records touch 1", p1.returncode == 0 and st["test_touches"] == 1 and "Step 14 final test" in md.read_text(),
          p1.stderr[-3000:])
    p2 = subprocess.run(argv, capture_output=True, text=True)
    check("F7 red: a second final is refused", p2.returncode != 0 and json.loads((gd / guard.STATE_FILE).read_text())["test_touches"] == 1,
          p2.stdout[-300:])
    check("F6 red: preflight refuses after the touch", any("already evaluated" in x for x in FG.preflight_reasons(gd, md, g / "data", FG.read_plan(planp), expect)))
    # F8: runner plan listing (no preflight pass needed: count the labels it prints)
    env = dict(os.environ, FRESH_VPY=sys.executable, FRESH_RES=str(tmp / "r8"), FRESH_SCRATCH=str(tmp / "s8"),
               FRESH_M4=str(tmp), FRESH_M2=str(tmp))
    out = subprocess.run(["bash", str(HERE / "fresh_test_box.sh"), "plan"], capture_output=True, text=True, env=env).stdout
    n = sum(1 for ln in out.splitlines() if ln.startswith("  4b_") or ln.startswith("  2b_"))
    check("F8 plan lists 33 readouts", n == 33, out[-400:])
    out = subprocess.run(["bash", str(HERE / "fresh_test_box.sh"), "plan"], capture_output=True, text=True,
                         env=dict(env, FRESH_RECORDS="4b")).stdout
    n = sum(1 for ln in out.splitlines() if ln.startswith("  4b_") or ln.startswith("  2b_"))
    check("F8 FRESH_RECORDS=4b lists 14", n == 14, out[-400:])
    p = subprocess.run(["bash", str(HERE / "fresh_test_box.sh")], capture_output=True, text=True, env=dict(env, PHASE="one", LABEL="4b_ctrl_0"))
    check("F8 red: PHASE=one outside the guard refused", p.returncode == 2, p.stderr)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
