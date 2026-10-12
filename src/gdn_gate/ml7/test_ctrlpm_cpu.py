#!/usr/bin/env python3
"""CPU checks for the parameter-matched control ctrlpm of the Qwen3.5-4B record (owner approval 2026-10-11 ~23:55
Israel): ctrlpm = trainer --arm ctrl --rank 20 from the 4B warm-up widened to rank 20 by pad_rank.py. Each check that
could pass vacuously carries a red arm that must fail it. The instrument (train_grpo_vllm.py, readout_vllm.py,
gate_vllm.py, rl_common.py, arms.py, packing.py, step14_analysis.py) is not edited: P0 checks its sha256 against the
4B touch-1 code lock.

  P0 instrument unchanged: the six readout/trainer files and step14_analysis.py equal results/ml7_4b/
     step14/code.sha256 by sha256.
  P1 parameter counts on the real Qwen3.5-4B config (meta device, the trainer's own arms.install + arms.add_lora):
     ctrl r16 8,675,328; wide1 r16 + gate W 8,675,328 + 1,966,080 = 10,641,408; ctrlpm r20 10,844,160 (+1.91%); rank
     20 is the closest uniform rank (19: -3.19%, 21: +7.0%) and the only one within 5% above or below. Red arm: ctrl at
     r16 misses wide1 by -18.5%, outside 5%.
  P2 widening on the real 4B warm-up checkpoint (ctrl_s0_cap512_fmt_box ckpt_step40, sha256 ee5cc709...): the first 16
     components are bit-equal, the new B columns are exactly zero, the new A rows lie inside PEFT's init bound and
     differ between seeds and are reproducible for one seed; PEFT's own rank-20 LoRA A init on the tiny model has the
     same bound. Red arms: a padded file with one new B column made nonzero, or new A rows zeroed, fails verify().
  P3 forward identity on the tiny Qwen3.5 model: a rank-16 "trained" LoRA (random A and B) and its rank-20 widening,
     each loaded through train_grpo_vllm.init_lora_from into arms.add_lora(model, r), give the same logits (max abs
     difference < 1e-5 x the logit scale). Red arm: the widening with one new B column at 1e-2 gives a difference
     above 100 x that tolerance.
  P4 trainer end to end on CPU (the CPU engine of test_grpo_vllm_cpu.py): --arm ctrl --rank 20 --init_from <widened>
     runs, its startup parity check passes (LoRA effect visible and agreeing, red arm failing), meta.json records the
     widened init (sha256) and n_lora = the tiny model's rank-20 count, the checkpoints hold rank-20 tensors. Red arms:
     --rank 20 with the unwidened rank-16 file is refused by init_lora_from; an engine that drops the LoRA fails parity.
  P5 vLLM plugin at rank 20 (vLLM imported on CPU): set_lora maps a rank-20 PEFT state and the engine insertions equal
     base + 2 B(A x) on every projection. Red arm: a rank-16 state into the rank-20 engine raises.
  P6 ctrlpm_analysis.py on the real 4B validation readouts (step12_box) and the MATH-500 test readouts of touch 1
     (step14), with stand-in ctrlpm readouts made from copies (no new data): ctrlpm := ctrl copy gives wide1 - ctrlpm =
     wide1 - ctrl exactly and ctrlpm - ctrl = 0 with no effect, the touch-1 self-check equal, H3 applying on MATH-500
     test with the touch-1 wide1 - ctrl verdict; ctrlpm := wide1 copy gives wide1 - ctrlpm = 0 and H3 not supported.
     Red arms: a ctrlpm file at rank 16, a ctrlpm file with arm wide1, and a touch-1 receipt with one changed number
     are refused.
Usage: CUDA_VISIBLE_DEVICES= ~/.cache/gdn_gate/vllm/venv/bin/python test_ctrlpm_cpu.py --out RECEIPT.json
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["WG_NO_FLA"] = "1"
HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "broad"))
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("TEST_THREADS", "4")))
import arms  # noqa: E402
import pad_rank as P  # noqa: E402
import rl_common as R  # noqa: E402
import test_cpu  # noqa: E402
import test_grpo_vllm_cpu as TG  # noqa: E402
import train_grpo_vllm as T  # noqa: E402

SCR = Path.home() / ".cache/gdn_gate/ml7/test_ctrlpm_cpu"
SNAP4 = os.path.join(os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub")), "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
WARM4 = Path(os.environ.get("WG_RUNS", str(Path(__file__).resolve().parents[3] / "runs"))) / "ml7_4b/grpo/ctrl_s0_cap512_fmt_box/ckpt_step40.pt"
WARM4_SHA = "ee5cc70988f943cb8b2bfe9b19b4dd9c1f5416ac3e4c9957f0e81e94afd4779f"
# Content digest of the same tensors (keys, dtypes, shapes, bytes): the published warm-up adapter converted with
# scripts/adapters.py to-pt holds equal tensors in a file with other bytes.
WARM4_CONTENT_SHA = "02fe77c6f01a674bbde0fab587eb7117a1963a8688e5b51bf9e86a5ba80805af"


def content_sha(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
R4 = WT / "results/ml7_4b"
COUNTS = {"ctrl": 8_675_328, "gate": 1_966_080, "wide1": 10_641_408, "ctrlpm": 10_844_160}
checks: list[dict] = []


def check(name, ok, **detail):
    checks.append({"check": name, "pass": bool(ok), **detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {json.dumps(detail, default=str)[:300]}", flush=True)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def p0():
    lock = {ln.split()[1].rsplit("/", 1)[1]: ln.split()[0] for ln in (R4 / "step14/code.sha256").read_text().splitlines()}
    files = {"readout_vllm.py": HERE / "readout_vllm.py", "train_grpo_vllm.py": HERE / "train_grpo_vllm.py",
             "gate_vllm.py": HERE.parent / "vllm/gate_vllm.py", "rl_common.py": HERE.parent / "broad/rl_common.py",
             "arms.py": HERE.parent / "broad/arms.py", "packing.py": HERE.parent / "broad/packing.py",
             "step14_analysis.py": HERE / "step14_analysis.py"}
    # Published copies: the lab files were path-scrubbed for release, so a file also passes when its sha256 is the
    # published hash that results/SCRUB_LEDGER.json records for the locked lab hash.
    pub = {f["sha256_original"]: f.get("sha256_released") for f in
           json.loads((WT / "results/SCRUB_LEDGER.json").read_text())["files"]}
    diff = {k: (sha(p)[:12], lock.get(k, "")[:12]) for k, p in files.items()
            if sha(p) != lock.get(k) and sha(p) != pub.get(lock.get(k))}
    # readout_vllm.py may differ by owner-approved ledger amendments only (2026-10-12 effect_visible, fresh set)
    if "readout_vllm.py" in diff and subprocess.run([sys.executable, str(HERE / "code_ledger.py"), "accept", "--file",
                                                     str(files["readout_vllm.py"]), "--want", lock["readout_vllm.py"]],
                                                    capture_output=True).returncode == 0:
        diff.pop("readout_vllm.py")
    check("P0 instrument unchanged: trainer, readout, plugin, rl_common, arms, packing, step14_analysis equal the 4B "
          "touch-1 code lock", not diff, diff=diff or "all equal")


def count_4b(arm: str, rank: int) -> tuple[int, int]:
    from transformers import AutoConfig, AutoModelForCausalLM

    tc = json.load(open(f"{SNAP4}/config.json"))["text_config"]
    tc.update(model_type="qwen3_5_text", mtp_num_hidden_layers=0)
    cfg = AutoConfig.for_model(**tc)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    arms.install(m, arm)
    m = arms.add_lora(m, rank)
    n_l = sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and "gate_w" not in n)
    n_w = sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and "gate_w" in n)
    return n_l, n_w


def p1():
    c16, _ = count_4b("ctrl", 16)
    l16, w = count_4b("wide1", 16)
    by_rank = {r: count_4b("ctrl", r)[0] for r in (19, 20, 21)}
    wide1 = l16 + w
    rel = {r: (n - wide1) / wide1 for r, n in by_rank.items()}
    best = min(rel, key=lambda r: abs(rel[r]))
    ok = (c16 == COUNTS["ctrl"] and w == COUNTS["gate"] and wide1 == COUNTS["wide1"]
          and by_rank[20] == COUNTS["ctrlpm"] and best == 20 and abs(rel[20]) < 0.05)
    check("P1 counts on the real 4B config: ctrl 8,675,328; gate 1,966,080; wide1 10,641,408; ctrlpm r20 10,844,160, "
          "the closest uniform rank, within 5%", ok, ctrl=c16, gate=w, wide1=wide1, by_rank=by_rank,
          rel={r: round(v, 4) for r, v in rel.items()})
    red = (c16 - wide1) / wide1
    check("P1 red arm: ctrl at r16 misses wide1's count by more than 5% (the matching test can fail)", abs(red) > 0.05,
          rel=round(red, 4))


def p2():
    src = torch.load(WARM4, map_location="cpu")
    out0, rec = P.pad_state(src, 20, 0)
    out0b, _ = P.pad_state(src, 20, 0)
    out1, _ = P.pad_state(src, 20, 1)
    v = P.verify(src, out0, 16, 20)
    same_seed = all(torch.equal(out0[k], out0b[k]) for k in out0)
    a_keys = [k for k in out0 if ".lora_A." in k]
    seeds_differ = all(not torch.equal(out0[k][16:], out1[k][16:]) for k in a_keys)
    n = sum(x.numel() for x in out0.values())
    check("P2 widening of the real 4B warm-up: first 16 components bit-equal, new B zero, new A inside the init bound, "
          "reproducible per seed, different across seeds; 10,844,160 numbers",
          (sha(WARM4) == WARM4_SHA or content_sha(src) == WARM4_CONTENT_SHA) and not v
          and same_seed and seeds_differ and n == COUNTS["ctrlpm"] and rec["rank_from"] == 16, verify=v, numel=n,
          pairs=rec["pairs"])
    k_b = next(k for k in out0 if ".lora_B." in k)
    bad_b = dict(out0)
    bad_b[k_b] = out0[k_b].clone()
    bad_b[k_b][0, 17] = 1e-3
    bad_a = dict(out0)
    bad_a[a_keys[0]] = out0[a_keys[0]].clone()
    bad_a[a_keys[0]][16:] = 0
    rb, ra = P.verify(src, bad_b, 16, 20), P.verify(src, bad_a, 16, 20)
    check("P2 red arms: a new B column made nonzero, or new A rows zeroed, fail verify()", bool(rb) and bool(ra),
          red_b=rb[:1], red_a=ra[:1])
    # PEFT's own LoRA A init has the bound the widening draws from
    md = test_cpu.tiny_model_dir()
    torch.manual_seed(5)
    m, _ = arms.load_model(md, "ctrl", device_map={"": "cpu"}, dtype=torch.float32)
    m = arms.add_lora(m, 20)
    ratios = [float(p.abs().max()) * p.shape[1] ** 0.5 for n, p in m.named_parameters() if ".lora_A." in n]
    check("P2 PEFT's own rank-20 LoRA A init lies in U(-1/sqrt(in), 1/sqrt(in)) (max |A| sqrt(in) in (0.9, 1])",
          all(0.9 < r <= 1.0 + 1e-6 for r in ratios), ratio_min=min(ratios), ratio_max=max(ratios))


def trained_r16(md: str) -> dict:
    torch.manual_seed(11)
    m, _ = arms.load_model(md, "ctrl", device_map={"": "cpu"}, dtype=torch.float32)
    m = arms.add_lora(m, 16)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if ".lora_B." in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return T.state_of(m)


def logits_with(md: str, rank: int, path: Path, x: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(99)
    m, _ = arms.load_model(md, "ctrl", device_map={"": "cpu"}, dtype=torch.float32)
    m = arms.add_lora(m, rank)
    T.init_lora_from(m, path)
    with torch.no_grad():
        return m(input_ids=x).logits.float()


def p3(md: str):
    st = trained_r16(md)
    f16, f20, fbad = SCR / "tiny_r16.pt", SCR / "tiny_r20.pt", SCR / "tiny_r20_bad.pt"
    torch.save(st, f16)
    w, _ = P.pad_state(st, 20, 0)
    torch.save(w, f20)
    bad = {k: v.clone() for k, v in w.items()}
    kb = next(k for k in bad if ".lora_B." in k and "in_proj_qkv" in k)
    bad[kb][:, 16:] = 1e-2
    torch.save(bad, fbad)
    x = torch.randint(0, 1000, (2, 24), generator=torch.Generator().manual_seed(0))
    l16, l20, lbad = logits_with(md, 16, f16, x), logits_with(md, 20, f20, x), logits_with(md, 20, fbad, x)
    scale = float(l16.abs().max())
    d, dbad = float((l20 - l16).abs().max()), float((lbad - l16).abs().max())
    tol = 1e-5 * scale
    check("P3 forward identity: the rank-20 widening gives the rank-16 LoRA's logits", d < tol, max_abs=d, tol=tol)
    check("P3 red arm: one new B column at 1e-2 moves the logits by more than 100 x the tolerance", dbad > 100 * tol,
          max_abs=dbad, tol=tol)
    return f16, f20


def p4(md: str, f16: Path, f20: Path):
    real = R.correct

    def grader(text, gold):
        return real(text, gold) if "\\boxed" in text else (len(text) % 3 == 0)

    R.correct = grader
    try:
        o = SCR / "run_ctrlpm"
        shutil.rmtree(o, ignore_errors=True)
        a = TG.args_for(md, "ctrl", o, rank=20, init_from=str(f20))
        res = T.run(a, make_engine=lambda a, mml, th: TG.CpuEngine(a, th))
        meta = res["meta"]
        par = meta["parity"]
        torch.manual_seed(0)
        m, _ = arms.load_model(md, "ctrl", device_map={"": "cpu"}, dtype=torch.float32)
        n20 = sum(p.numel() for n, p in arms.add_lora(m, 20).named_parameters() if p.requires_grad)
        ck = torch.load(o / "ckpt_step4.pt")
        ranks = {tuple(v.shape)[0] for k, v in ck.items() if ".lora_A." in k} | \
                {tuple(v.shape)[1] for k, v in ck.items() if ".lora_B." in k}
        check("P4 trainer --arm ctrl --rank 20 --init_from <widened>: parity passes, meta records the widened init, "
              "n_lora = rank-20 count, rank-20 checkpoints, DONE",
              par["pass"] and par["start_pass"] and meta["init_from"]["sha256"] == sha(f20)
              and meta["init_from"]["applied"] and meta["n_lora"] == n20 and meta["n_w"] == 0 and ranks == {20}
              and (o / "DONE").exists(), checks=par["checks"], n_lora=meta["n_lora"], n20=n20, ranks=sorted(ranks))
        r = SCR / "run_ctrlpm_unpadded"
        shutil.rmtree(r, ignore_errors=True)
        try:
            T.run(TG.args_for(md, "ctrl", r, rank=20, init_from=str(f16)),
                  make_engine=lambda a, mml, th: TG.CpuEngine(a, th))
            refused, msg = False, ""
        except SystemExit as e:
            refused, msg = "shapes differ" in str(e), str(e)[:160]
        check("P4 red arm: --rank 20 with the unwidened rank-16 warm-up is refused by init_lora_from", refused, msg=msg)
        r = SCR / "run_ctrlpm_droplora"
        shutil.rmtree(r, ignore_errors=True)
        try:
            T.run(TG.args_for(md, "ctrl", r, rank=20, init_from=str(f20)),
                  make_engine=lambda a, mml, th: TG.CpuEngine(a, th, drop="lora"))
            failed, msg = False, ""
        except SystemExit as e:
            failed, msg = "parity FAILED" in str(e), str(e)[:160]
        check("P4 red arm: an engine that drops the rank-20 LoRA fails the parity check", failed, msg=msg)
    finally:
        R.correct = real


P5_RED = r"""
bad_state = {}
for p, (A, B) in P.items():
    bad_state[pre + p + ".lora_A.default.weight"] = A[:16]
    bad_state[pre + p + ".lora_B.default.weight"] = B[:, :16]
try:
    NV.set_lora(root, bad_state, gate=False)
    refused = False
except NV.GateError as e:
    refused = "rank 20" in str(e)
print(json.dumps({"red_rank16_refused": refused}))
"""


def p5():
    code = TG.G8_CODE.replace('NV.install("stock", lora_rank=4)', 'NV.install("stock", lora_rank=20)')
    code = code.replace("r, H, kd, vd, nv, T = 4, 32, 8, 16, 4, 5", "r, H, kd, vd, nv, T = 20, 32, 8, 16, 4, 5")
    assert code != TG.G8_CODE and "lora_rank=20" in code and "= 20, 32" in code
    r = subprocess.run([sys.executable, "-c", code + P5_RED, str(HERE.parent / "vllm"), str(SCR / "vllm_cache")],
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    try:
        lines = [json.loads(x) for x in r.stdout.strip().splitlines() if x.startswith("{")]
        d = {**lines[0], **lines[1]}
    except Exception:  # noqa: BLE001
        check("P5 plugin LoRA at rank 20", False, rc=r.returncode, err=r.stderr[-400:])
        return
    check("P5 plugin at rank 20: set_lora maps a rank-20 state; insertions equal base + 2 B(A x); lora=False zeroes",
          d["err"] < 1e-4 and d["zeroed"] == 0.0 and d["modules"] == 4 and d["rewired"], **d)
    check("P5 red arm: a rank-16 state into the rank-20 engine raises", d["red_rank16_refused"], **d)


def plain(p: Path) -> Path:
    """The released readouts are gzipped (readout_*.json.gz); decompress one into the scratch dir when needed."""
    p = Path(p)
    if p.exists() or not Path(str(p) + ".gz").exists():
        return p
    out = SCR / "plain" / p.parent.name / p.name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(gzip.open(str(p) + ".gz").read())
    return out


def fake_pm(src: Path, dst: Path, seed: int, rank: int = 20, arm: str = "ctrl") -> str:
    """A stand-in ctrlpm readout: a copy of an existing readout's rows, labeled as readout_vllm.py --arm ctrl
    --rank 20 writes a ctrlpm readout (no new data)."""
    d = json.loads(plain(src).read_text())
    d["args"]["rank"] = rank
    d["args"]["arm"] = d["arm"] = arm
    d["ckpt"] = f"/x/grpo/ctrlpm_s{seed}_cap512_r_box/ckpt_step150.pt"
    d["env"]["sha256"]["ckpt"] = hashlib.sha256(f"stand-in ctrlpm {seed} {src.name}".encode()).hexdigest()
    dst.write_text(json.dumps(d))
    return str(dst)


def analysis(split: str, base: Path, warm: Path, files: dict, touch1: Path, out: Path) -> tuple[int, dict, str]:
    slots = [f"{k}={v}" for k, v in files.items()]
    p = subprocess.run([sys.executable, str(HERE / "ctrlpm_analysis.py"), "--split", split, "--baseline", str(base),
                        "--warmup", str(warm), *slots, "--manifest", str(WT / "ml_pipeline_4b/split_manifest.json"),
                        "--touch1", str(touch1), "--out", str(out)], capture_output=True, text=True, timeout=600)
    res = json.loads(out.read_text()) if p.returncode == 0 and out.is_file() else {}
    return p.returncode, res, p.stderr[-400:]


def p6():
    s = SCR / "an"
    shutil.rmtree(s, ignore_errors=True)
    s.mkdir(parents=True)
    for split, rd, prefix in (("val", R4 / "step12_box", "readout_val_"), ("test", R4 / "step14", "readout_math500_")):
        touch1 = rd / ("analysis_val.json" if split == "val" else "analysis_math500.json")
        base, warm = plain(rd / f"{prefix}untouched.json"), plain(rd / f"{prefix}warmup.json")
        orig = {f"{a}_{x}": str(plain(rd / f"{prefix}{a}_{x}.json")) for a in ("ctrl", "wide", "wide1") for x in (0, 1, 2)}
        f_ctrl = {f"ctrlpm_{x}": fake_pm(rd / f"{prefix}ctrl_{x}.json", s / f"{split}_pm_ctrl_{x}.json", x) for x in (0, 1, 2)}
        f_w1 = {f"ctrlpm_{x}": fake_pm(rd / f"{prefix}wide1_{x}.json", s / f"{split}_pm_w1_{x}.json", x) for x in (0, 1, 2)}
        rc, A, err = analysis(split, base, warm, {**orig, **f_ctrl}, touch1, s / f"{split}_a.json")
        rc2, B, err2 = analysis(split, base, warm, {**orig, **f_w1}, touch1, s / f"{split}_b.json")
        ok = rc == 0 and rc2 == 0
        detail = {"rc": [rc, rc2], "err": (err or err2)[-200:]}
        if ok:
            e, le = A["effects"], A["length_effects"]
            ok = (e["wide1 - ctrlpm"]["per_seed_diff"] == e["wide1 - ctrl"]["per_seed_diff"]
                  and e["wide1 - ctrlpm"]["ci95"] == e["wide1 - ctrl"]["ci95"]
                  and le["wide1 - ctrlpm"]["per_seed_diff"] == le["wide1 - ctrl"]["per_seed_diff"]
                  and e["ctrlpm - ctrl"]["seed_mean_diff"] == 0 and e["ctrlpm - ctrl"]["verdict"] == "no effect"
                  and A["ctrlpm"]["touch1_check"]["equal_on_original_arms"]
                  and B["effects"]["wide1 - ctrlpm"]["seed_mean_diff"] == 0
                  and B["effects"]["wide1 - ctrlpm"]["verdict"] == "no effect")
            if split == "test":
                t1 = json.loads(touch1.read_text())["effects"]["wide1 - ctrl"]
                h3a, h3b = A["preregistered_hypotheses"]["H3"], B["preregistered_hypotheses"]["H3"]
                ok = ok and h3a["applies"] and h3a["rule_verdict"] == t1["verdict"] and h3a["seed_mean_diff"] == \
                    t1["seed_mean_diff"] and h3b["verdict"].startswith("not supported")
                detail.update(h3_ctrl_copy=h3a["verdict"], h3_wide1_copy=h3b["verdict"], touch1=t1["verdict"])
            detail.update(w1_pm=e["wide1 - ctrlpm"]["seed_mean_diff"], pm_ctrl=e["ctrlpm - ctrl"]["verdict"])
        check(f"P6 ctrlpm_analysis on the 4B {split} readouts: stand-in ctrlpm = ctrl copy gives wide1 - ctrlpm = "
              "wide1 - ctrl, ctrlpm - ctrl = 0 (no effect), touch-1 self-check equal; = wide1 copy gives 0 (H3 not "
              "supported on test)", ok, **detail)
        reds = {}
        r16 = {**orig, **f_ctrl, "ctrlpm_1": fake_pm(rd / f"{prefix}ctrl_1.json", s / f"{split}_r16.json", 1, rank=16)}
        rc, _, err = analysis(split, base, warm, r16, touch1, s / f"{split}_r16_out.json")
        reds["rank16"] = rc == 2 and "args.rank 16" in err
        aw = {**orig, **f_ctrl, "ctrlpm_1": fake_pm(rd / f"{prefix}ctrl_1.json", s / f"{split}_aw.json", 1, arm="wide1")}
        rc, _, err = analysis(split, base, warm, aw, touch1, s / f"{split}_aw_out.json")
        reds["arm wide1"] = rc == 2 and "file arm 'wide1'" in err
        t = json.loads(touch1.read_text())
        t["effects"]["wide - ctrl"]["per_seed_diff"][0] += 1e-9
        bad_t1 = s / f"{split}_touch1_bad.json"
        bad_t1.write_text(json.dumps(t))
        rc, _, err = analysis(split, base, warm, {**orig, **f_ctrl}, bad_t1, s / f"{split}_t1_out.json")
        reds["touch1 changed"] = rc == 3 and "touch-1" in err
        check(f"P6 red arms ({split}): a rank-16 ctrlpm file, a ctrlpm file with arm wide1, a touch-1 receipt with one "
              "changed number are refused", all(reds.values()), **reds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    TG.SCR.mkdir(parents=True, exist_ok=True)
    md = test_cpu.tiny_model_dir()
    want = set(a.only.split(",")) if a.only else {"p0", "p1", "p2", "p3", "p4", "p5", "p6"}
    if "p0" in want:
        p0()
    if "p1" in want:
        p1()
    if "p2" in want:
        p2()
    f16 = f20 = None
    if want & {"p3", "p4"}:
        f16, f20 = p3(md)
    if "p4" in want:
        p4(md, f16, f20)
    if "p5" in want:
        p5()
    if "p6" in want:
        p6()
    out = {"pass": all(c["pass"] for c in checks), "n_checks": len(checks), "checks": checks,
           "seconds": round(time.time() - t0, 1), "torch": torch.__version__, "host": os.uname().nodename,
           "cmd": " ".join(sys.argv)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print("ALL PASS" if out["pass"] else "SOME FAILED", len(checks))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
