#!/usr/bin/env python3
"""CPU checks for the broad stream (step 6 code), on a tiny random Qwen3.5 text model with the real tokenizer.

Each check can fail and carries a red arm where a check could otherwise pass vacuously.
  T1 wide1 bound (arms.check_wide1_bound): wide1 beta stays in [0, 1]; red arm: wide exceeds 1 on the same probe.
  T2 real forward, three arms: at W = 0 every arm's logits equal ctrl exactly; with the same random W, the betas
     the patched forward actually feeds the kernel stay <= 1 for wide1 and exceed 1 for wide, and the logits of
     wide, wide1 and ctrl differ pairwise (the gate reaches the output).
  T3 cached decode carries the gate: greedy generate with the KV/state cache vs a full no-cache forward on the same
     tokens agree for wide, while wide vs its stripped forward (W = 0) differs by much more (red arm).
  T4 train_sft.py end to end (wide and ctrl, 4 steps): the gate W moves off zero for wide, ctrl has no gate;
     resume from step 2 reproduces step 4 bit for bit; red arm: step 2 and step 4 tensors differ.
  T5 battery.py on the trained wide checkpoint, its strip twin and the untouched base (every bench but
     humaneval, which calls .cuda() and runs in the GPU pilot; 3 items each):
     strip drops exactly one gate tensor per GDN layer; the explore/confirm split of MMLU-Pro ids is 50/50 +- 2%.
  T6 analyze_broad.py on synthetic seed results: an injected +5-point wide - ctrl effect (seed SD 1) is a hit in
     explore and confirmed; red arm: the same pipeline on null data flags nothing.

Usage: CUDA_VISIBLE_DEVICES= python test_cpu.py --out results/broad/test_cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["WG_NO_FLA"] = "1"  # torch GDN path on CPU; inherited by the train_sft / battery subprocesses
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms  # noqa: E402
import patch  # noqa: E402

HF = os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
RUNS = os.environ.get("WG_RUNS", str(HERE.parents[2] / "runs"))
BASE = f"{HF}/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b"
POST = f"{HF}/models--Qwen--Qwen3.5-4B/snapshots"
TULU = f"{RUNS}/broad/data/tulu3_permissive.parquet"
EVAL = f"{RUNS}/broad/eval_data"
SCR = Path.home() / ".cache/gdn_gate/broad/test_cpu"
PY = sys.executable

checks: list[dict] = []
# humaneval goes through retrofit/general_eval.humaneval unchanged, which calls .cuda(): GPU pilot only
CPU_BENCHES = "mmlu_cloze,arc_cloze,gsm8k,mbppplus,swde,fda,squad,ruler,nll"


def check(name, ok, **kw):
    if "pass" in kw:  # a detail dict's own verdict must not overwrite this check's
        kw["detail_pass"] = kw.pop("pass")
    checks.append({"check": name, "pass": bool(ok), **kw})
    print(("PASS " if ok else "FAIL ") + name, json.dumps(kw)[:300], flush=True)


def tiny_model_dir() -> str:
    """A 4-layer (3 GDN + 1 attention) Qwen3.5 text model, random init, with the real tokenizer and the post-trained
    chat template (so the thinking template is testable on it)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    d = SCR / "tiny"
    if (d / "config.json").exists():
        return str(d)
    d.mkdir(parents=True, exist_ok=True)
    tc = json.load(open(f"{BASE}/config.json"))["text_config"]
    tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
              layer_types=["linear_attention"] * 3 + ["full_attention"], linear_num_key_heads=2,
              linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, num_attention_heads=2,
              num_key_value_heads=1, head_dim=16, mtp_num_hidden_layers=0, dtype="float32")
    tc["model_type"] = "qwen3_5_text"
    cfg = AutoConfig.for_model(**tc)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    m.save_pretrained(d)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(f"{BASE}/{f}", d / f)
    post = sorted(Path(POST).glob("*/chat_template.jinja"))
    if post:
        shutil.copy(post[0], d / "chat_template.jinja")
    return str(d)


def tiny_data() -> str:
    import pyarrow.parquet as pq

    p = SCR / "tulu_small.parquet"
    if not p.exists():
        f = pq.ParquetFile(TULU)
        t = f.read_row_group(0)
        pq.write_table(t.slice(0, 3000), p)
    return str(p)


def t1():
    r = arms.check_wide1_bound()
    check("T1 wide1 bound, red arm wide > 1", r["pass"], **r)


def _betas(model):
    seen = []
    for layer in patch.gdn_layers(model):
        if hasattr(layer, "_wg_beta"):
            f = layer._wg_beta

            def wrap(b, x, f=f):
                out = f(b, x)
                seen.append(out.detach().float())
                return out
            layer._wg_beta = wrap
    return seen


def t2_t3(md):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(md)
    ids = torch.tensor([tok.encode("The quick brown fox jumps over the lazy dog. " * 6)])
    logits, betas, models = {}, {}, {}
    for arm in ("ctrl", "wide", "wide1"):
        m, new = arms.load_model(md, arm, device_map={"": "cpu"}, dtype=torch.float32)
        models[arm] = m.eval()
        with torch.no_grad():
            logits[(arm, 0)] = m(ids).logits
    same0 = all(torch.equal(logits[("ctrl", 0)], logits[(a, 0)]) for a in ("wide", "wide1"))
    check("T2 W=0: wide and wide1 logits equal ctrl exactly", same0)
    g = torch.Generator().manual_seed(1)
    ws = None
    for arm in ("wide", "wide1"):
        layers = patch.gdn_layers(models[arm])
        if ws is None:
            ws = [torch.randn(l.gate_w.weight.shape, generator=g) * 3.0 for l in layers]
        with torch.no_grad():
            for l, w in zip(layers, ws):
                l.gate_w.weight.copy_(w)
        seen = _betas(models[arm])
        with torch.no_grad():
            logits[(arm, 1)] = models[arm](ids).logits
        betas[arm] = (max(b.max().item() for b in seen), min(b.min().item() for b in seen), len(seen))
    check("T2 forward: wide1 betas fed to the kernel stay in [0,1]", betas["wide1"][0] <= 1.0 and betas["wide1"][1] >= 0.0,
          max=betas["wide1"][0], min=betas["wide1"][1], calls=betas["wide1"][2])
    check("T2 red arm: wide betas exceed 1 on the same W", betas["wide"][0] > 1.0, max=betas["wide"][0])
    d = lambda a, b: (logits[a] - logits[b]).abs().max().item()  # noqa: E731
    check("T2 gate reaches the logits: wide, wide1, ctrl pairwise differ",
          min(d(("wide", 1), ("ctrl", 0)), d(("wide1", 1), ("ctrl", 0)), d(("wide", 1), ("wide1", 1))) > 1e-4,
          wide_ctrl=d(("wide", 1), ("ctrl", 0)), wide1_ctrl=d(("wide1", 1), ("ctrl", 0)), wide_wide1=d(("wide", 1), ("wide1", 1)))

    # T3: cached decode vs full forward, wide with the random W
    m = models["wide"]
    for l in patch.gdn_layers(m):
        l._wg_beta = l._wg_beta.__defaults__[0] if hasattr(l._wg_beta, "__defaults__") and l._wg_beta.__defaults__ else l._wg_beta
    with torch.no_grad():
        out = m.generate(input_ids=ids, max_new_tokens=12, do_sample=False, output_logits=True,
                         return_dict_in_generate=True, pad_token_id=tok.eos_token_id, eos_token_id=None)
        cached = torch.stack(out.logits, 1)[0]  # [12, V], logits that produced each new token
        full = m(out.sequences).logits[0, ids.shape[1] - 1 : -1]
        for l in patch.gdn_layers(m):
            l.gate_w.weight.zero_()
        strip = m(out.sequences).logits[0, ids.shape[1] - 1 : -1]
    e_cf = (cached - full).abs().max().item()
    e_fs = (full - strip).abs().max().item()
    check("T3 cached decode carries the gate: |cached - full| << |gated - stripped|", e_cf < 1e-3 and e_fs > 100 * e_cf,
          cached_vs_full=e_cf, gated_vs_stripped=e_fs)


def run(cmd, log):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    Path(log).write_text(r.stdout + "\n---stderr---\n" + r.stderr)
    return r.returncode, time.time() - t


def t4(md, data):
    common = ["--model", md, "--data", data, "--steps", "4", "--ckpt_steps", "2,4", "--step_tokens", "1500",
              "--micro_tokens", "1600", "--max_len", "400", "--val_n", "6", "--device", "cpu", "--seed", "3"]
    outs = {}
    for arm in ("wide", "ctrl"):
        o = SCR / f"sft_{arm}"
        shutil.rmtree(o, ignore_errors=True)
        rc, s = run([PY, "train_sft.py", "--arm", arm, "--out", str(o)] + common, SCR / f"sft_{arm}.log")
        check(f"T4 train_sft {arm} 4 steps exit 0 with DONE", rc == 0 and (o / "DONE").exists(), seconds=round(s, 1))
        outs[arm] = o
    a = torch.load(outs["wide"] / "ckpt_step4.pt")
    w = [v for k, v in a.items() if "gate_w" in k]
    wn = float(sum(x.float().norm() ** 2 for x in w) ** 0.5) if w else 0.0
    check("T4 wide: 3 gate tensors trained off zero", len(w) == 3 and wn > 0, n=len(w), w_norm=wn)
    c = torch.load(outs["ctrl"] / "ckpt_step4.pt")
    check("T4 ctrl: no gate tensors, same LoRA tensor names as wide",
          not any("gate_w" in k for k in c) and set(c) == {k for k in a if "gate_w" not in k}, n=len(c))
    o = SCR / "sft_wide_resume"
    shutil.rmtree(o, ignore_errors=True)
    rc1, _ = run([PY, "train_sft.py", "--arm", "wide", "--out", str(o), "--stop_after", "2"] + common, SCR / "r1.log")
    (o / "ckpt_step4.pt").unlink(missing_ok=True)
    rc2, _ = run([PY, "train_sft.py", "--arm", "wide", "--out", str(o), "--resume"] + common, SCR / "r2.log")
    b = torch.load(o / "ckpt_step4.pt") if (o / "ckpt_step4.pt").exists() else {}
    eq = bool(b) and set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    check("T4 resume from step 2 reproduces step 4 bit for bit", rc1 == 0 and rc2 == 0 and eq)
    a2 = torch.load(outs["wide"] / "ckpt_step2.pt")
    check("T4 red arm: step 2 and step 4 tensors differ", any(not torch.equal(a[k], a2[k]) for k in a))
    v = [json.loads(l) for l in open(outs["wide"] / "val_log.jsonl")]
    check("T4 val log at steps 0, 2, 4 with gate stats", [x["step"] for x in v] == [0, 2, 4] and "gate_stat_mean" in v[-1],
          val=[round(x["val_loss"], 4) for x in v])
    return outs["wide"] / "ckpt_step4.pt"


def t5(md, ck):
    res = {}
    for name, extra in (("wide", ["--ckpt", str(ck), "--arm", "wide"]), ("strip", ["--ckpt", str(ck), "--arm", "wide", "--strip"]),
                        ("base", ["--ckpt", "none"])):
        o = SCR / f"bat_{name}.json"
        o.unlink(missing_ok=True)
        rc, s = run([PY, "battery.py", "--model", md, "--data", EVAL, "--out", str(o), "--limit", "3", "--device", "cpu",
                     "--benches", CPU_BENCHES] + extra, SCR / f"bat_{name}.log")
        j = json.loads(o.read_text()) if o.exists() else {}
        res[name] = j
        ok = rc == 0 and set(j.get("bench", {})) == set(CPU_BENCHES.split(","))
        check(f"T5 battery {name}: every bench scored", ok, seconds=round(s, 1),
              n={b: v["n"] for b, v in j.get("bench", {}).items()})
    check("T5 strip dropped one gate tensor per GDN layer (3)", res["strip"].get("load", {}).get("dropped_gate_tensors") == 3)
    nl_w = res["wide"]["bench"]["nll"]["items"]
    nl_s = res["strip"]["bench"]["nll"]["items"]
    check("T5 strip changes the model (nll items differ)", any(abs(nl_w[k] - nl_s[k]) > 1e-6 for k in nl_w))
    import battery

    ids = [json.loads(l)["id"] for l in open(f"{EVAL}/mmlu_pro.jsonl")]
    fa = sum(battery.split_of("mmlu_cloze", i) == "A" for i in ids) / len(ids)
    check("T5 MMLU-Pro explore/confirm split is 50/50 +- 2%", abs(fa - 0.5) < 0.02, frac_A=fa, n=len(ids))


def t6():
    import analyze_broad as A

    rng = np.random.default_rng(7)
    for label, eff in (("effect", 0.05), ("null", 0.0)):
        root = SCR / f"syn_{label}"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        n_items = 4000
        ids = [str(i) for i in range(n_items)]
        for seed in range(6):
            seed_off = {b: rng.normal(0, 0.01) for b in ("mmlu_cloze", "gsm8k", "arc_cloze")}
            for arm in ("wide", "wide1", "ctrl"):
                bench = {}
                for b in ("mmlu_cloze", "gsm8k", "arc_cloze"):
                    p = 0.5 + seed_off[b] + rng.normal(0, 0.01) + (eff if (arm == "wide" and b == "mmlu_cloze") else 0)
                    items = {i: int(rng.random() < p) for i in ids}
                    bench[b] = {"items": items, "split": {i: A_split(b, i) for i in ids}}
                (root / f"{arm}_s{seed}_step1200.json").write_text(json.dumps({"load": {"strip": False}, "bench": bench}))
        out = root / "explore.json"
        r = subprocess.run([PY, "analyze_broad.py", "--root", str(root), "--mode", "explore", "--out", str(out)],
                           capture_output=True, text=True, cwd=HERE)
        j = json.loads(out.read_text())
        hits = [(h["bench"], h["contrast"]) for h in j["hits"]]
        if label == "effect":
            check("T6 explore finds the injected wide - ctrl MMLU effect", ("mmlu_cloze", "wide-ctrl") in hits, hits=hits)
            hyp = root / "hyp.json"
            hyp.write_text(json.dumps([{"bench": "mmlu_cloze", "step": 1200, "contrast": "wide-ctrl", "sign": 1}]))
            outc = root / "confirm.json"
            subprocess.run([PY, "analyze_broad.py", "--root", str(root), "--mode", "confirm", "--hyp", str(hyp),
                            "--out", str(outc)], capture_output=True, text=True, cwd=HERE)
            jc = json.loads(outc.read_text())
            check("T6 confirm (half B, Holm) confirms it", jc["results"][0]["confirmed"], mean=jc["results"][0]["mean_diff"])
        else:
            check("T6 red arm: null data gives no explore hit", not hits, hits=hits, n_tests=j["n_tests"], rc=r.returncode)


def A_split(b, i):
    import battery

    return battery.split_of(b, i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    SCR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    md = tiny_model_dir()
    data = tiny_data()
    only = set(a.only.split(",")) if a.only else None
    if not only or "t1" in only:
        t1()
    if not only or "t2" in only:
        t2_t3(md)
    ck = None
    if not only or "t4" in only:
        ck = t4(md, data)
    if not only or "t5" in only:
        t5(md, ck or SCR / "sft_wide/ckpt_step4.pt")
    if not only or "t6" in only:
        t6()
    import transformers

    out = {"pass": all(c["pass"] for c in checks), "n_checks": len(checks), "checks": checks,
           "seconds": round(time.time() - t0, 1), "torch": torch.__version__, "transformers": transformers.__version__,
           "host": os.uname().nodename, "python": sys.version.split()[0],
           "git_describe": subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty"], capture_output=True,
                                          text=True).stdout.strip(),
           "cmd": " ".join(sys.argv)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("ALL PASS" if out["pass"] else "SOME FAILED", len(checks))


if __name__ == "__main__":
    main()
