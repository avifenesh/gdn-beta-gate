#!/usr/bin/env python3
"""test_test9b_cpu.py: CPU tests for the 9B AIME 2025 read (build_aime2025.py, test9b_analysis.py, test9b_guard.py,
test9b_spark.sh), each with a red arm. Synthetic data and readouts only: no model, no GPU, no Spark, and no read of
test_aime2025.jsonl or test_fresh.jsonl (N0 hashes code files, nothing else).

  N0  frozen instrument: the readout code, step14/ctrlpm/fresh analyses, fresh_guard.py and guard.py equal the 4B
      fresh read's code lock (results/fresh/code.sha256) by sha256
  N1  builder: selects exactly the 30 aime2025 lines, byte for byte, in file order; refuses a wrong source sha256, a
      missing line and a duplicate id; the guard state has touches 0 and the frame fingerprint
  N2  analysis: Gate(0,1) better and shorter -> H5 and H6 supported; red arm (wide1 rows = ctrl rows) -> neither;
      refusals: rank 20, probe 0.2, data sha256 not in the manifest, a wide slot, mixed GPU, a missing seed, ids
      outside AIME 2025
  N3  guard preflight: OK on a complete plan; refuses without the pre-registration block, with a frozen-code sha256
      mismatch, a checkpoint sha256 mismatch, an init_from not the warm-up, a plan probe 0.2, a missing label, touches 1
  N4  runner end to end (fake sparklock, son.sh, spark_job.sh, rsync): plan exit 0 with the 7 labels on their units;
      launch -> 7 readouts, each once, every unit its own queue in order and then labels from the end of the other's
      queue (fake jobs end at once, so stealing happens); analysis, guard touch 1, final.json; relaunch refused;
      PHASE=one outside the guard refused
  N5  runner faults: gb10b never leased -> gb10a takes gb10b's labels (own queue, then the other's from its end); one
      readout failing once -> retried and done (attempt 2); a label failing twice -> no touch recorded
Usage: <workstation vLLM venv python> test_test9b_cpu.py  (exit 0 = all pass; needs pandas and numpy)"""
from __future__ import annotations

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
sys.path.insert(0, str(ML7 / "fresh"))
sys.path.insert(0, str(ML7))
sys.path.insert(0, str(WT / "ml_pipeline"))
import build_aime2025 as B  # noqa: E402
import guard  # noqa: E402
import test9b_guard as TG  # noqa: E402

CAPS = [256, 512, 1024, 2048, 4096, 8192]
GPU = "NVIDIA GB10"
FAILS = []
IDS = [f"fresh:aime2025-{c}-{n}" for c in ("I", "II") for n in range(1, 16)]
LABELS = ["untouched"] + [f"{a}_{s}" for a in ("ctrl", "wide1") for s in range(3)]


def check(name, ok, why=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
    if not ok:
        FAILS.append(name)


def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_f(p) -> str:
    return sha_b(Path(p).read_bytes())


# ------------------------------------------------------------------------------------------------ synthetic data
def fresh_like() -> bytes:
    rows = []
    for src, n in (("aime2024", 30), ("aime2025", 30), ("aime2026", 30), ("amc2023", 40)):
        for k in range(n):
            c = "I" if k < n // 2 else "II"
            rows.append({"id": f"fresh:{src}-{c}-{k % (n // 2) + 1}", "problem": f"synthetic {src} {k} x^{k}",
                         "answer": str(k), "set": "fresh", "source": src, "contest": c, "number": k % (n // 2) + 1})
    random.Random(1).shuffle(rows)
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def readout(arm, seed, data_sha, quality, length, *, ckpt="none", ck_sha=None, rank=16, probe=0.05, gpu=GPU,
            ids=IDS, samples=8, rng_seed=0):
    rng = random.Random(f"{arm}:{seed}:{rng_seed}")
    rows = []
    for i in ids:
        p_item = rng.random()
        for s in range(samples):
            tl = max(50, int(rng.gauss(length, length * 0.3)))
            closed = tl <= 9000
            good = rng.random() < quality * (0.5 + p_item)
            caps = {}
            for c in CAPS:
                within = closed and tl <= c
                caps[str(c)] = [int(good and (within or rng.random() < 0.2)), min(tl, c), int(not within), 1, 1, 100]
            rows.append({"id": i, "sample": s, "think_len": tl, "closed": int(closed), "silent_stop": 0,
                         "phase1_tokens": tl + 100, "caps": caps})
    return {"schema": "wg-readout-v1", "arm": arm, "ckpt": ckpt, "red": "", "caps": CAPS,
            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
            "args": {"seed": 20261006, "samples": samples, "answer_max": 1280, "caps": ",".join(map(str, CAPS)),
                     "rank": rank, "probe_effect": probe},
            "env": {"gpu": gpu, "sha256": {"data": data_sha, "ckpt": ck_sha}}, "rows": rows}


SPEC = {"untouched": ("stock", 0.30, 7000), "ctrl": ("ctrl", 0.30, 6500), "wide1": ("wide1", 0.50, 3000)}


def readout_for(label, data_sha, ck_sha=None, red=False):
    a = "untouched" if label == "untouched" else label.rsplit("_", 1)[0]
    s = 0 if label == "untouched" else int(label.rsplit("_", 1)[1])
    arm, q, ln = SPEC[a]
    d = readout(arm, s, data_sha, q, ln, ckpt="none" if a == "untouched" else f"/x/{label}.pt", ck_sha=ck_sha,
                rng_seed=3)
    if red and a == "wide1":
        d["rows"] = readout("ctrl", s, data_sha, *SPEC["ctrl"][1:], rng_seed=3)["rows"]
    return d


def write(d, p):
    Path(p).write_text(json.dumps(d))
    return str(p)


# ------------------------------------------------------------------------------------------------ N0
def n0():
    lock = WT / "results/fresh/code.sha256"
    want = {Path(ln.split()[1]).name: ln.split()[0] for ln in lock.read_text().splitlines() if ln.strip()}
    files = {"readout_vllm.py": ML7 / "readout_vllm.py", "train_grpo_vllm.py": ML7 / "train_grpo_vllm.py",
             "gate_vllm.py": ML7.parent / "vllm/gate_vllm.py", "rl_common.py": ML7.parent / "broad/rl_common.py",
             "arms.py": ML7.parent / "broad/arms.py", "packing.py": ML7.parent / "broad/packing.py",
             "step14_analysis.py": ML7 / "step14_analysis.py", "ctrlpm_analysis.py": ML7 / "ctrlpm_analysis.py",
             "fresh_analysis.py": ML7 / "fresh/fresh_analysis.py", "fresh_guard.py": ML7 / "fresh/fresh_guard.py",
             "guard.py": WT / "ml_pipeline/guard.py"}
    # Published copies were path-scrubbed: a file also passes when its sha256 is the published hash that
    # results/SCRUB_LEDGER.json records for the locked lab hash.
    pub = {f["sha256_original"]: f.get("sha256_released") for f in
           json.loads((WT / "results/SCRUB_LEDGER.json").read_text())["files"] if f.get("sha256_original")}
    diff = [n for n, p in files.items() if sha_f(p) not in (want.get(n), pub.get(want.get(n)))]
    check("N0 frozen instrument equals the 4B fresh code lock", not diff, f"differ: {diff}")


# ------------------------------------------------------------------------------------------------ N1
def n1(tmp: Path):
    src = fresh_like()
    body, ids = B.select(src, src_sha=sha_b(src))
    want = b"".join(ln for ln in src.splitlines(keepends=True) if json.loads(ln)["id"].startswith("fresh:aime2025-"))
    check("N1 builder selects the 30 aime2025 lines byte for byte in file order", body == want and len(ids) == 30
          and sorted(ids) == sorted(IDS), f"{len(ids)} ids")

    def refuses(data, sha):
        try:
            B.select(data, src_sha=sha)
            return False
        except B.Refuse:
            return True
    lines = src.splitlines(keepends=True)
    drop = b"".join(x for k, x in enumerate(lines) if k != next(j for j, y in enumerate(lines) if b"aime2025-I-3\"" in y))
    dup = src + next(y for y in lines if b"aime2025-II-7\"" in y)
    check("N1 red: wrong source sha256 refused", refuses(src, "0" * 64))
    check("N1 red: a missing aime2025 line refused", refuses(drop, sha_b(drop)))
    check("N1 red: a duplicate id refused", refuses(dup, sha_b(dup)))
    out = tmp / "test_aime2025.jsonl"
    out.write_bytes(body)
    st = B.guard_state(tmp / "gd", out, sha_b(body))
    import pandas as pd  # noqa: PLC0415
    rows = [json.loads(x) for x in body.splitlines()]
    fp = guard._fingerprint(pd.DataFrame([{k: r[k] for k in ("id", "problem", "answer")} for r in rows]))
    check("N1 guard state touches 0 with the frame fingerprint", st["test_touches"] == 0 and st["test_fingerprint"] == fp
          and json.loads((tmp / "gd/split_manifest.json").read_text())["files"]["test_aime2025.jsonl"] == sha_b(body))


# ------------------------------------------------------------------------------------------------ N2
def run_an(argv):
    import importlib  # noqa: PLC0415

    for m in ("step14_analysis", "ctrlpm_analysis", "fresh_analysis", "test9b_analysis"):
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    import test9b_analysis as A  # noqa: PLC0415

    base, slots, man = argv
    return A.run(base, slots, man)


def an_refuses(argv):
    try:
        run_an(argv)
        return False
    except SystemExit:
        return True


def n2(tmp: Path):
    dsha = "d" * 64
    man = tmp / "man.json"
    man.write_text(json.dumps({"files": {"test_aime2025.jsonl": dsha}}))

    def build(red=False, mutate=None, labels=LABELS):
        base = write(readout_for("untouched", dsha), tmp / "u.json")
        slots = []
        for x in labels[1:]:
            d = readout_for(x, dsha, ck_sha=sha_b(x.encode()), red=red)
            if mutate:
                mutate(x, d)
            slots.append(f"{x}=" + write(d, tmp / f"{x}.json"))
        return base, slots, str(man)
    res = run_an(build())
    h = res["preregistered_hypotheses"]
    check("N2 gated better and shorter: H5 and H6 supported", h["H5"]["supported"] and h["H6"]["supported"]
          and set(h) == {"H5", "H6"} and res["n_items"] == 30, {k: v.get("verdict") for k, v in h.items()})
    check("N2 contrasts: wide1 - ctrl seed-paired plus each arm - untouched, nothing else",
          set(res["effects"]) == {"wide1 - ctrl", "ctrl - untouched", "wide1 - untouched"}
          and res["effects"]["wide1 - ctrl"]["seeds"] == [0, 1, 2], sorted(res["effects"]))
    res = run_an(build(red=True))
    h = res["preregistered_hypotheses"]
    check("N2 red: wide1 rows = ctrl rows -> neither supported", not h["H5"]["supported"] and not h["H6"]["supported"])

    def m_set(lab, key, val):
        def f(x, d):
            if x == lab:
                d["args"][key] = val
        return f
    check("N2 refuses rank 20", an_refuses(build(mutate=m_set("ctrl_1", "rank", 20))))
    check("N2 refuses probe 0.2", an_refuses(build(mutate=m_set("wide1_2", "probe_effect", 0.2))))

    def m_gpu(x, d):
        if x == "ctrl_0":
            d["env"]["gpu"] = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    check("N2 refuses mixed GPU", an_refuses(build(mutate=m_gpu)))
    b, s, _ = build()
    man2 = tmp / "man2.json"
    man2.write_text(json.dumps({"files": {"test_fresh.jsonl": dsha}}))
    check("N2 refuses data sha256 not the AIME 2025 file of the manifest", an_refuses((b, s, str(man2))))
    check("N2 refuses a wide slot", an_refuses((b, s + ["wide_0=" + write(readout("wide", 0, dsha, 0.5, 3000,
                                                                                    ck_sha="e" * 64), tmp / "w.json")], _)))
    check("N2 refuses a missing seed", an_refuses((b, [x for x in s if not x.startswith("wide1_2=")], _)))
    other = ["fresh:aime2024-I-1"] + IDS[1:]
    b2 = write(readout("stock", 0, dsha, 0.3, 7000, ids=other), tmp / "u2.json")
    s2 = [f"{x}=" + write(readout(x.rsplit('_', 1)[0], int(x[-1]), dsha, 0.4, 4000, ck_sha=sha_b(x.encode()), ids=other),
                          tmp / f"o_{x}.json") for x in LABELS[1:]]
    check("N2 refuses ids outside AIME 2025", an_refuses((b2, s2, _)))


# ------------------------------------------------------------------------------------------------ fixtures (N3-N5)
FROZEN = ["src/gdn_gate/ml7/readout_vllm.py", "src/gdn_gate/ml7/train_grpo_vllm.py",
          "src/gdn_gate/vllm/gate_vllm.py", "src/gdn_gate/broad/rl_common.py",
          "src/gdn_gate/broad/arms.py", "src/gdn_gate/broad/packing.py",
          "src/gdn_gate/ml7/step14_analysis.py", "src/gdn_gate/ml7/ctrlpm_analysis.py",
          "src/gdn_gate/ml7/fresh/fresh_analysis.py", "src/gdn_gate/ml7/test9b/test9b_analysis.py",
          "ml_pipeline/guard.py", "src/gdn_gate/ml7/fresh/fresh_guard.py",
          "src/gdn_gate/ml7/test9b/test9b_guard.py"]


def pipeline_md(block=True, bad_sha=False) -> str:
    lines = [f"    {('0' * 64) if (bad_sha and k == 0) else sha_f(WT / p)}  {p}" for k, p in enumerate(FROZEN)]
    head = TG.BLOCK + ", synthetic)\n\n" if block else "## Something else\n\n"
    return "# synthetic 9B record\n\n" + head + "\n".join(lines) + "\n\n## Next\n"


def fixture(tmp: Path) -> dict:
    """A complete synthetic record: data, guard dir, workstation run dirs with checkpoints, expect, model sums, plan."""
    data = tmp / "data"
    data.mkdir(parents=True)
    rows = [{"id": i, "problem": f"synthetic {i}", "answer": str(k)} for k, i in enumerate(IDS)]
    body = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (data / "test_aime2025.jsonl").write_bytes(body)
    ml = tmp / "ml"
    ml.mkdir()
    (ml / "PIPELINE.md").write_text(pipeline_md())
    B.guard_state(ml / "aime2025_test", data / "test_aime2025.jsonl", sha_b(body))
    grpo = tmp / "grpo"
    warm_run = "ctrl_s0_cap512_fmt9b_gb10"
    (grpo / warm_run).mkdir(parents=True)
    (grpo / warm_run / "ckpt_step40.pt").write_bytes(b"warm" * 100)
    (grpo / warm_run / "DONE").write_text(json.dumps({"steps": 40}))
    warm_sha = sha_f(grpo / warm_run / "ckpt_step40.pt")
    expect = {"warmup": {"run": warm_run, "step": 40, "path": str(grpo / warm_run / "ckpt_step40.pt"), "sha256": warm_sha}}
    for x in LABELS[1:]:
        a, s = x.rsplit("_", 1)
        run = f"{a}_s{s}_cap512_r9b_gb10"
        rd = grpo / run
        rd.mkdir()
        (rd / "ckpt_step150.pt").write_bytes(f"ckpt {x}".encode() * 50)
        (rd / "DONE").write_text(json.dumps({"steps": 150}))
        (rd / "meta.json").write_text(json.dumps({"args": {"arm": a, "seed": int(s), "cap": 512, "answer_max": 1280,
                                                           "rank": 16, "probe_effect": 0.05, "alpha": 0.2,
                                                           "model": "<models>/Qwen3.5-9B"},
                                                  "init_from": {"sha256": warm_sha, "applied": True}}))
        expect[x] = {"run": run, "step": 150, "path": str(rd / "ckpt_step150.pt"), "sha256": sha_f(rd / "ckpt_step150.pt")}
    (tmp / "expect.json").write_text(json.dumps(expect))
    (ml / "model_Qwen3.5-9B.sha256").write_text("a" * 64 + "  ./config.json\n")
    return {"data": data, "ml": ml, "gd": ml / "aime2025_test", "expect": tmp / "expect.json", "grpo": grpo,
            "sums": ml / "model_Qwen3.5-9B.sha256", "body": body}


def plan_rows(fx: dict, res: Path) -> list[dict]:
    exp = json.loads(fx["expect"].read_text())
    return [{"label": x, "record": "9b", "arm": TG.arm_of(x), "ckpt": "none" if x == "untouched" else exp[x]["path"],
             "out": str(res / f"readout_aime2025_{x}.json"), "model": "$NN/models/Qwen3.5-9B", "answer_max": "1280",
             "probe": "0.05", "rank": "16"} for x in LABELS]


def write_plan(rows, p: Path):
    keys = ("label", "record", "arm", "ckpt", "out", "model", "answer_max", "probe", "rank")
    p.write_text("".join("\t".join(r[k] for k in keys) + "\n" for r in rows))


def preflight(fx, plan_p: Path) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(HERE / "test9b_guard.py"), "preflight", "--gd", str(fx["gd"]), "--md",
                        str(fx["ml"] / "PIPELINE.md"), "--data", str(fx["data"]), "--plan", str(plan_p), "--expect",
                        str(fx["expect"])], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


# ------------------------------------------------------------------------------------------------ N3
def n3(tmp: Path):
    fx = fixture(tmp)
    rows = plan_rows(fx, tmp / "res")
    pp = tmp / "plan.tsv"
    write_plan(rows, pp)
    rc, out = preflight(fx, pp)
    check("N3 preflight OK on a complete plan", rc == 0, out[-600:])

    def red(name, expect_text, setup, undo):
        setup()
        rc, out = preflight(fx, pp)
        undo()
        check(f"N3 red: {name}", rc == 2 and expect_text in out, out[-400:])
    md = fx["ml"] / "PIPELINE.md"
    good_md = md.read_text()
    red("no pre-registration block", "no pre-registration block", lambda: md.write_text(pipeline_md(block=False)),
        lambda: md.write_text(good_md))
    red("frozen-code sha256 mismatch", "differs from the pre-registered", lambda: md.write_text(pipeline_md(bad_sha=True)),
        lambda: md.write_text(good_md))
    ck = Path(rows[2]["ckpt"])
    orig = ck.read_bytes()
    red("checkpoint sha256 mismatch", "differs from the recorded", lambda: ck.write_bytes(orig + b"x"),
        lambda: ck.write_bytes(orig))
    meta = ck.parent / "meta.json"
    m0 = meta.read_text()
    red("init_from not the warm-up", "need the warm-up",
        lambda: meta.write_text(m0.replace(json.loads(m0)["init_from"]["sha256"], "f" * 64)), lambda: meta.write_text(m0))
    red("plan probe 0.2", "pre-registered '0.05'",
        lambda: write_plan([{**r, "probe": "0.2"} if r["label"] == "wide1_1" else r for r in rows], pp),
        lambda: write_plan(rows, pp))
    red("a missing label", "need exactly", lambda: write_plan(rows[:-1], pp), lambda: write_plan(rows, pp))
    st = fx["gd"] / ".guard_state.json"
    s0 = st.read_text()
    red("touches 1", "touches 1", lambda: st.write_text(s0.replace('"test_touches": 0', '"test_touches": 1')),
        lambda: st.write_text(s0))


# ------------------------------------------------------------------------------------------------ N4 / N5 fakes
FAKE_SPARKLOCK = r"""#!/usr/bin/env bash
cmd=$1; shift
case $cmd in
  status) echo "gb10a: free"; echo "gb10b: free";;
  run) U=""; while [ $# -gt 0 ]; do case $1 in --unit) U=$2; shift 2;; --) shift; break;; *) shift;; esac; done
       case " ${FAKE_BUSY:-} " in *" $U "*) echo "sparklock: $U busy" >&2; exit 1;; esac
       echo "$U $LABEL" >> "$FAKE_DIR/leases.log"; "$@";;
esac
"""
FAKE_SON = r"""#!/usr/bin/env bash
case "$2" in *t9b_remote.sh*) ;; *) ;; esac
exit 0
"""
FAKE_SPARK_JOB = r"""#!/usr/bin/env bash
U=$1; NAME=$2; CMD=$4
J=$SPARK_JOB_LOCAL/$U-$NAME; rm -rf "$J"; mkdir -p "$J"; printf '%s\n' "$CMD" > "$J/cmd"
lab=${NAME#t9b-aime2025-}; lab=${lab%-*}
n=$(cat "$FAKE_DIR/calls_$lab" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$FAKE_DIR/calls_$lab"
echo "$U $lab" >> "$FAKE_DIR/jobs.log"
fails=$(echo " ${FAKE_FAIL:-} " | tr ' ' '\n' | grep -c "^$lab$")
if [ "$n" -le "$fails" ]; then echo 1 > "$J/rc"; echo "fake failure" > "$J/log"; exit 1; fi
"$FAKE_PY" "$FAKE_TEST" --fake-readout "$CMD" "$J/readout.json" > "$J/log" 2>&1; rc=$?
echo $rc > "$J/rc"; exit $rc
"""


def fake_readout(cmd: str, out: str) -> int:
    """The fake unit job: parse the readout argv in CMD and write a synthetic readout for the fixture's data."""
    toks = cmd.split()
    i = toks.index("--arm")
    arm = toks[i + 1]
    ck = toks[toks.index("--ckpt") + 1] if "--ckpt" in toks else "none"
    exp = json.loads(Path(os.environ["FAKE_EXPECT"]).read_text())
    if ck == "none":
        label, ck_sha = "untouched", None
    else:
        run = ck.split("/")[-2]
        label = next(k for k, v in exp.items() if v.get("run") == run)
        ck_sha = exp[label]["sha256"]
    dsha = sha_f(os.environ["FAKE_DATAFILE"])
    d = readout_for(label, dsha, ck_sha=ck_sha)
    assert d["arm"] == arm
    for k, cast in (("samples", int), ("answer_max", int), ("rank", int), ("probe_effect", float), ("seed", int)):
        d["args"][k] = cast(toks[toks.index("--" + k) + 1])
    d["args"]["caps"] = toks[toks.index("--caps") + 1]
    Path(out).write_text(json.dumps(d))
    return 0


def runner_env(tmp: Path, fx: dict, **extra) -> dict:
    sp = tmp / "sp"
    sp.mkdir(exist_ok=True)
    for name, body in (("son.sh", FAKE_SON), ("spark_job.sh", FAKE_SPARK_JOB), ("sync_code.sh", "#!/bin/sh\nexit 0\n"),
                       ("rsh.sh", "#!/bin/sh\nexit 0\n")):
        (sp / name).write_text(body)
        (sp / name).chmod(0o755)
    sl = tmp / "sparklock"
    sl.write_text(FAKE_SPARKLOCK)
    sl.chmod(0o755)
    fd = tmp / "fake"
    fd.mkdir(exist_ok=True)
    env = dict(os.environ, NINE_SP=str(sp), NINE_SPARKLOCK=str(sl), NINE_RSYNC="true", NINE_UNIT_CHECKS="0",
               NINE_VPY=sys.executable, NINE_DATA=str(fx["data"]), NINE_ML=str(fx["ml"]), NINE_GD=str(fx["gd"]),
               NINE_RES=str(tmp / "res"), NINE_EXPECT=str(fx["expect"]), NINE_MODEL_SUMS=str(fx["sums"]),
               NINE_SCRATCH=str(tmp / "scratch"), SPARK_JOB_LOCAL=str(tmp / "jobs"), FAKE_DIR=str(fd),
               FAKE_PY=sys.executable, FAKE_TEST=str(Path(__file__).resolve()), FAKE_EXPECT=str(fx["expect"]),
               FAKE_DATAFILE=str(fx["data"] / "test_aime2025.jsonl"), NINE_TTL="20m", NINE_WAIT="1m")
    env.pop("PHASE", None)
    env.pop("FRESH_UNDER_GUARD", None)
    env.update(extra)
    return env


def runner(env, *args):
    return subprocess.run(["bash", str(HERE / "test9b_spark.sh"), *args], env=env, capture_output=True, text=True)


def touches(fx) -> int:
    return json.loads((fx["gd"] / ".guard_state.json").read_text())["test_touches"]


QUEUES = {"gb10a": ["untouched", "ctrl_0", "wide1_0"], "gb10b": ["ctrl_1", "wide1_1", "ctrl_2", "wide1_2"]}


def placement_ok(jobs) -> bool:
    """Every label read once; per unit, its own labels are a prefix of its queue in order, and the labels it took
    from the other unit are a suffix of that unit's queue (taken from the end)."""
    labs = [lab for _, lab in jobs]
    if sorted(labs) != sorted(LABELS):
        return False
    for u, q in QUEUES.items():
        other = next(v for k, v in QUEUES.items() if k != u)
        seq = [lab for x, lab in jobs if x == u]
        own = [x for x in seq if x in q]
        stolen = [x for x in seq if x in other]
        if own != q[:len(own)] or seq != own + stolen or stolen != other[::-1][:len(stolen)]:
            return False
    return True


def n4(tmp: Path):
    fx = fixture(tmp)
    env = runner_env(tmp, fx)
    r = runner(env, "plan")
    units = {ln.split()[0]: ln.split()[2] for ln in r.stdout.splitlines() if ln.startswith("  ") and len(ln.split()) > 5
             and ln.split()[0] in LABELS}
    want = {"untouched": "gb10a", "ctrl_0": "gb10a", "wide1_0": "gb10a", "ctrl_1": "gb10b", "wide1_1": "gb10b",
            "ctrl_2": "gb10b", "wide1_2": "gb10b"}
    check("N4 plan exit 0 with the 7 labels on their units", r.returncode == 0 and units == want
          and "plan: launchable" in r.stdout, (r.returncode, units, r.stdout[-800:] + r.stderr[-800:]))
    r = subprocess.run(["bash", str(HERE / "test9b_spark.sh")], env={**env, "PHASE": "one", "LABEL": "ctrl_0",
                                                                     "UNIT": "gb10a"}, capture_output=True, text=True)
    check("N4 PHASE=one outside the guard refused", r.returncode == 2 and "only under" in r.stderr, r.stderr[-300:])
    r = runner(env)
    res = tmp / "res"
    jobs = [ln.split() for ln in (tmp / "fake/jobs.log").read_text().splitlines()] if (tmp / "fake/jobs.log").exists() else []
    ran = {lab: u for u, lab in jobs}
    fin = json.loads((res / "final.json").read_text()) if (res / "final.json").exists() else {}
    check("N4 launch: 7 readouts, each once, own queue first then the other's from its end; analysis, touch 1",
          r.returncode == 0 and placement_ok(jobs) and len(jobs) == 7 and touches(fx) == 1 and fin.get("score") == 1.0
          and (res / "analysis_fresh_9b.json").exists() and (res / "DONE").exists()
          and all((res / f"done/{x}.sha256").exists() for x in LABELS)
          and "fresh_rows_read_by_all_models=1.0000 (touch 1" in (fx["ml"] / "PIPELINE.md").read_text()
          and fin.get("what", "").startswith("AIME 2025"),
          (r.returncode, ran, touches(fx), r.stdout[-1500:], r.stderr[-1500:]))
    an = json.loads((res / "analysis_fresh_9b.json").read_text()) if (res / "analysis_fresh_9b.json").exists() else {}
    check("N4 analysis: H5 and H6 supported on the synthetic gated-better-shorter readouts",
          an.get("preregistered_hypotheses", {}).get("H5", {}).get("supported") is True
          and an["preregistered_hypotheses"]["H6"]["supported"] is True)
    r = runner(env)
    check("N4 relaunch after the touch refused", r.returncode == 2 and "recorded" in r.stdout and touches(fx) == 1,
          (r.returncode, r.stdout[-300:]))


def n5(tmp: Path):
    fx = fixture(tmp / "a")
    env = runner_env(tmp / "a", fx, FAKE_BUSY="gb10b", FAKE_FAIL="ctrl_1")
    r = runner(env)
    jobs = [ln.split() for ln in (tmp / "a/fake/jobs.log").read_text().splitlines()] if (tmp / "a/fake/jobs.log").exists() else []
    order = [lab for u, lab in jobs]
    steps = (tmp / "a/res/steps.log").read_text() if (tmp / "a/res/steps.log").exists() else ""
    check("N5 gb10b never leased: gb10a runs its queue, then gb10b's from the end; ctrl_1 fails once and is retried",
          r.returncode == 0 and all(u == "gb10a" for u, _ in jobs) and touches(fx) == 1
          and order == ["untouched", "ctrl_0", "wide1_0", "wide1_2", "ctrl_2", "wide1_1", "ctrl_1", "ctrl_1"]
          and "attempt 1 of 2" in steps, (r.returncode, order, steps[-1200:], r.stderr[-800:]))
    fx = fixture(tmp / "b")
    env = runner_env(tmp / "b", fx, FAKE_FAIL="wide1_1 wide1_1")
    r = runner(env)
    check("N5 red: a label failing twice -> no touch recorded, FAIL marker, rerun resumes",
          r.returncode != 0 and touches(fx) == 0 and (tmp / "b/res/FAIL").exists()
          and not (tmp / "b/res/final.json").exists(), (r.returncode, touches(fx)))
    r = runner(env)
    check("N5 rerun resumes: only wide1_1 read again, then touch 1",
          r.returncode == 0 and touches(fx) == 1
          and (tmp / "b/fake/jobs.log").read_text().split().count("wide1_1") == 3, (r.returncode, r.stdout[-600:]))


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--fake-readout":
        return fake_readout(sys.argv[2], sys.argv[3])
    n0()
    for f in (n1, n2, n3, n4, n5):
        base = Path(os.environ.get("TMPDIR_T9B", Path.home() / ".cache/gdn_gate/ml7/test9b_cpu"))
        base.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"t9b_{f.__name__}_", dir=base))
        try:
            f(tmp)
        finally:
            if not os.environ.get("KEEP_T9B"):
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
