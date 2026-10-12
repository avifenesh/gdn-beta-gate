#!/usr/bin/env python3
"""CPU checks for train_grpo_vllm.py, on the tiny random Qwen3.5 text model of broad/test_cpu.py. Each check that
could pass vacuously carries a red arm that must fail it.

  G1 reward and advantage: wrong = 0, shorter correct > longer; advantage = reward - group mean (sums to 0 per
     group) with no std scaling (doubling the rewards doubles the advantages). Red arm: a std-scaled advantage is
     scale-invariant and fails the doubling check.
  G2 rollout bookkeeping with engine logprobs: closed+ended, closed but cut (phase 2), never closed (forced close,
     no logprob on forced tokens), silent stop (stop and its logprob dropped). Red arm: a missing phase-2 logprob
     is caught by finish().
  G3 loss shape: forward_logps gradients and log-probs equal a direct full-logit reference
     -sum A log p / norm, packed and padded; doubling norm halves the gradients. Red arm: the reference with the
     one-token shift off by one disagrees.
  G5 data refusal: the frozen pool passes; a one-byte-tampered copy, a file named test_*.jsonl and a wrong name
     are refused; --cap is a required argument.
  G6 grader fail-closed: with math_verify importable require_grader() passes; with it blocked the process fails.
  G8 plugin unmerged LoRA arithmetic and set_lora mapping (vLLM imported on CPU), with a swapped-slot red arm.
  G7 run end to end (wide and ctrl, CPU engine with its own model copy and real logprobs): parity passes and is
     decisive, logs, checkpoints and DONE are written; a run killed at step 3 (after the step-2 checkpoint and the
     step-2 log line) resumes to a step-4 state equal bit for bit to the uninterrupted run's, and its logs hold each
     step once. Red arms: an engine that drops the gate (wide) or the LoRA (ctrl) on sync fails the parity check.
  G9 --init_from: a wide run started from a ctrl warm-up checkpoint holds exactly the warm-up LoRA at its first step
     (bit-equal), the gate at its usual init and an empty optimizer (step 0, schedule 0); meta.json records the path
     and sha256; its checkpoints hold the full LoRA (moved off the warm-up) plus the gate, the set readout_vllm.py
     loads. Red arms: a checkpoint with a key missing, a shape changed (rank 8 instead of 16) or a gate tensor added
     is refused, by run() too. A run killed at step 3 and resumed with --init_from pointing at a refused checkpoint
     ignores it and reaches the uninterrupted run's step-4 state bit for bit.
  G10 --length_on: the reasoning count is readout_vllm.split_trace's think_len and the answer count its answer
     tokens; with reasoning the reward is computed on reasoning tokens. In a group where the reasoning and completion
     orderings differ the rewards differ (red arm: completion mode ranks the group differently). On a real CPU run (an
     engine that cuts each phase at a seed-fixed length, so the two lengths vary) with
     --length_on reasoning every logged reward recomputes from the logged reasoning counts and, in at least one
     group, not from the completion counts. --length_on has no default.
Usage: CUDA_VISIBLE_DEVICES= python test_grpo_vllm_cpu.py --out results/ml7/test_grpo_vllm_cpu.json
(run with the workstation vLLM venv: math_verify and peft are there; vLLM itself is not needed)
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
os.environ["WG_NO_FLA"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "broad"))
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("TEST_THREADS", "4")))  # a loaded 24-core box: 24 threads ran 25x slower
import train_grpo_vllm as T  # noqa: E402
import arms  # noqa: E402
import rl_common as R  # noqa: E402
import test_cpu  # noqa: E402

POOL = os.path.join(os.environ.get("WG_RUNS", str(Path(__file__).resolve().parents[3] / "runs")), "ml7/data/rl_pool.jsonl")
SCR = Path.home() / ".cache/gdn_gate/ml7/test_grpo_cpu"
RANK = 16  # args_for leaves --rank at the trainer default
checks = test_cpu.checks
check = test_cpu.check


def g1():
    r = R.length_reward([1, 1, 0, 1], [100, 300, 50, 200], 0.2)
    ok = r[2] == 0 and r[0] > r[3] > r[1]
    corr, lens = [1, 0, 1, 1, 0, 0, 1, 0], [10, 20, 30, 15, 5, 5, 40, 9]
    rew, adv = T.group_rewards(corr, lens, 4, 0.2)
    sums = [sum(adv[i:i + 4]) for i in (0, 4)]
    mu = [sum(rew[i:i + 4]) / 4 for i in (0, 4)]
    exact = all(abs(adv[i] - (rew[i] - mu[i // 4])) < 1e-12 for i in range(8))
    adv2 = [2 * x for x in adv]  # what doubled rewards give without std scaling
    check("G1 reward gated by correctness, shorter correct wins; advantage = reward - group mean, sums to 0",
          ok and exact and all(abs(s) < 1e-12 for s in sums), r=r, adv=adv)

    def std_adv(rw):
        out = []
        for i in (0, 4):
            g = rw[i:i + 4]
            m = sum(g) / 4
            sd = (sum((x - m) ** 2 for x in g) / 4) ** 0.5
            out += [(x - m) / (sd + 1e-6) for x in g]
        return out

    ok_ours = all(abs(2 * x - y) < 1e-12 for x, y in zip(adv, adv2))
    s1, s2 = std_adv(rew), std_adv([2 * x for x in rew])
    red = all(abs(2 * x - y) < 1e-9 for x, y in zip(s1, s2))
    check("G1 no std scaling: doubled rewards double the advantage; red arm (std-scaled) fails it", ok_ours and not red)


def g2(md):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(md)
    th = R.Thinker(tok)
    E, IM, a, b = th.end_think, th.im_end, 100, 101
    p = th.prompt("1+1?")
    g1 = [[a, a, E, b, IM], [a, E, b, b], [a, a, a, a], [a, a, IM]]
    lp1 = [[-0.1 * (i + 1)] * len(g) for i, g in enumerate(g1)]
    rolls, need = T.assemble([p] * 4, g1, lp1, th)
    g2 = [[b, IM], [b, b, IM], [b, IM]]
    lp2 = [[-2.0] * len(g) for g in g2]
    T.finish(rolls, need, g2, lp2, lambda t: tok.decode(t, skip_special_tokens=True))
    r0, r1, r2, r3 = rolls
    F = len(th.force)
    ok = (need == [1, 2, 3] and not r0.forced and r0.lp == lp1[0] and T.sampled_lp(r0) == lp1[0]
          and r1.completion == g1[1] + g2[0] and r1.lp == lp1[1] + lp2[0]
          and r2.forced and r2.lp == lp1[2] + [None] * F + lp2[1] and T.sampled_lp(r2) == lp1[2] + lp2[1]
          and r3.forced and r3.think == [a, a] and r3.lp == lp1[3][:2] + [None] * F + lp2[2]
          and all(len(r.lp) == len(r.completion) for r in rolls))
    check("G2 rollouts: closed, cut, forced and silent-stop paths keep one engine logprob per sampled token", ok)
    rolls, need = T.assemble([p] * 4, g1, lp1, th)
    try:
        T.finish(rolls, need, g2, [lp2[0], lp2[1], lp2[2][:-1]], lambda t: "")
        caught = False
    except AssertionError:
        caught = True
    check("G2 red arm: a missing phase-2 logprob is caught", caught)


def g3(md):
    torch.manual_seed(0)
    model, _ = arms.load_model(md, "wide", device_map={"": "cpu"}, dtype=torch.float32)
    model = arms.add_lora(model, 4)
    import packing

    packing.install(model)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.normal_(0, 0.02)
    items = [([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, [True, True, True, True, False, True, True], 0.7),
             ([11, 12, 13, 14, 15], 2, [True, True, True], -0.3),
             ([3, 3, 3, 3, 5, 6], 4, [True, False], 0.5)]
    norm = 9.0
    params = [p for p in model.parameters() if p.requires_grad]
    dev = torch.device("cpu")

    def grads():
        g = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        for p in params:
            p.grad = None
        return g

    def ref(shift, nrm=norm):
        loss, lps = 0.0, []
        for ids, plen, mask, A in items:
            lp = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), -1)
            cur = []
            for j, m in enumerate(mask):
                if m:
                    x = lp[plen + j - shift, ids[plen + j]]
                    loss = loss - A * x
                    cur.append(float(x))
            lps.append(cur)
        (loss / nrm).backward()
        return grads(), lps

    g_ref, lp_ref = ref(1)
    scale = max(y.abs().max().item() for y in g_ref)
    res = {}
    for pk in (True, False):
        _, lp = T.forward_logps(model, items, dev, micro_tokens=12, pack=pk, norm=norm)
        g = grads()
        err = max((x - y).abs().max().item() for x, y in zip(g, g_ref))
        lerr = max(abs(u - v) for a, b in zip(lp, lp_ref) for u, v in zip(a, b))
        res[pk] = (err, lerr)
        check(f"G3 forward_logps ({'packed' if pk else 'padded'}): grads and log p equal the full-logit reference",
              err < 1e-5 * max(1, scale) and lerr < 1e-4 and scale > 0, grad_err=err, lp_err=lerr, scale=scale)
    T.forward_logps(model, items, dev, micro_tokens=12, pack=True, norm=2 * norm)
    g_half = grads()
    herr = max((2 * x - y).abs().max().item() for x, y in zip(g_half, g_ref))
    check("G3 loss normalization: doubling norm halves the gradients", herr < 1e-5 * max(1, scale), err=herr)
    g_bad, _ = ref(0)
    errb = max((x - y).abs().max().item() for x, y in zip(g_ref, g_bad))
    check("G3 red arm: the reference with the shift off by one disagrees", errb > 100 * max(res[True][0], 1e-9), err_bad=errb)


def g5():
    man = T.MANIFEST
    ok = T.verify_frozen(POOL, "rl_pool.jsonl", man)["rows"] == 6965
    d = SCR / "data"
    shutil.rmtree(d, ignore_errors=True)
    (d / "bad").mkdir(parents=True)
    raw = bytearray(Path(POOL).read_bytes())
    raw[100] = ord("Z") if raw[100] != ord("Z") else ord("Y")
    (d / "bad" / "rl_pool.jsonl").write_bytes(bytes(raw))
    shutil.copy(POOL, d / "test_gsm8k.jsonl")
    shutil.copy(POOL, d / "pool_copy.jsonl")
    refused = {}
    for name, path in (("tampered", d / "bad" / "rl_pool.jsonl"), ("test_name", d / "test_gsm8k.jsonl"),
                       ("wrong_name", d / "pool_copy.jsonl")):
        try:
            T.verify_frozen(path, "rl_pool.jsonl", man)
            refused[name] = False
        except SystemExit as e:
            refused[name] = str(e)[:80]
    check("G5 frozen pool passes; tampered, test-named and wrong-named files are refused",
          ok and all(refused.values()), refused=refused)
    base = ["--model", "m", "--pool", POOL, "--arm", "wide", "--seed", "0", "--out", "o", "--gpu_mem", "0.3",
            "--micro_tokens", "8192", "--length_on", "completion"]
    try:
        T.build_parser().parse_args(base)
        req = False
    except SystemExit:
        req = True
    okp = T.build_parser().parse_args(base + ["--cap", "1024"]).cap == 1024
    check("G5 --cap has no default (parse fails without it)", req and okp)
    nolen = [x for x in base if x != "--length_on" and x != "completion"] + ["--cap", "1024"]
    try:
        T.build_parser().parse_args(nolen)
        reql = False
    except SystemExit:
        reql = True
    okl = T.build_parser().parse_args(base + ["--cap", "1024"]).length_on == "completion"
    check("G5 --length_on has no default (parse fails without it)", reql and okl)


def g6():
    code = ("import sys; sys.path.insert(0, %r); %s import train_grpo_vllm as T; print(T.require_grader())")
    good = subprocess.run([sys.executable, "-c", code % (str(HERE), "")], capture_output=True, text=True)
    bad = subprocess.run([sys.executable, "-c", code % (str(HERE), "sys.modules['math_verify'] = None;")],
                         capture_output=True, text=True)
    check("G6 grader: passes with math_verify, the process fails with math_verify blocked",
          good.returncode == 0 and bad.returncode != 0, good=good.stdout.strip()[-40:], bad_rc=bad.returncode,
          bad_err=bad.stderr.strip()[-120:])


class CpuEngine:
    """The VEngine interface on CPU: its own copy of the model, weights by sync(), multinomial sampling at
    temperature 1 with the sampled token's exact log p. drop: 'gate' or 'lora' zeroes that part on every sync
    (red arms: an engine the gate or LoRA never reaches)."""

    def __init__(self, a, th, drop="", varlen=False):
        torch.manual_seed(1234)
        m, _ = arms.load_model(a.model, a.arm, device_map={"": "cpu"}, dtype=torch.float32)
        self.m = arms.add_lora(m, a.rank).eval()
        self.gated, self.stops, self.drop, self.varlen = a.arm != "ctrl", set(th.stops), drop, varlen

    def info(self):
        return {"engine": "cpu-test"}

    def sync(self, state, gate=None, lora=True):
        gate = self.gated if gate is None else gate
        st = {}
        for k, v in state.items():
            if ("gate_w" in k and (not gate or self.drop == "gate")) or ("lora_B" in k and (self.drop == "lora" or not lora)):
                v = torch.zeros_like(v)
            st[k] = v
        T.load_state(self.m, st)
        return {"sync_s": 0.0}

    @torch.no_grad()
    def score(self, items):
        out = []
        for ids, plen, mask, _ in items:
            l = torch.log_softmax(self.m(input_ids=torch.tensor([ids])).logits[0].float(), -1)
            out.append([float(l[plen + j - 1, ids[plen + j]]) for j, m in enumerate(mask) if m])
        return out

    @torch.no_grad()
    def sample(self, contexts, max_tokens, seeds):
        G, LP = [], []
        for c, s in zip(contexts, seeds):
            gen = torch.Generator().manual_seed(int(s) % (2**62))
            ids, g, lp = list(c), [], []
            # varlen: a seed-fixed shorter length, as a natural stop would give (the random model never closes or
            # ends), so reasoning and answer lengths vary independently between the two phases
            n = max_tokens - (int(s) % (max_tokens // 2 + 1) if self.varlen else 0)
            for _ in range(n):
                l = torch.log_softmax(self.m(input_ids=torch.tensor([ids]), logits_to_keep=1).logits[0, -1].float(), -1)
                t = int(torch.multinomial(l.exp(), 1, generator=gen))
                g.append(t)
                lp.append(float(l[t]))
                ids.append(t)
                if t in self.stops:
                    break
            G.append(g)
            LP.append(lp)
        return G, LP


def args_for(md, arm, out, **kw):
    argv = ["--model", md, "--pool", POOL, "--arm", arm, "--seed", "3", "--out", str(out), "--cap", "10",
            "--gpu_mem", "0.3", "--micro_tokens", "256", "--steps", "4", "--prompts", "2",
            "--group", "4", "--answer_max", "5", "--ckpt_every", "2", "--parity_prompts", "2", "--parity_tokens", "8",
            "--device", "cpu", "--length_on", "completion"]
    a = T.build_parser().parse_args(argv)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def lines(p):
    return [json.loads(x) for x in open(p)] if Path(p).exists() else []


def g7(md):
    real = R.correct

    def grader(text, gold):  # random-token texts never hold \boxed: a deterministic pseudo grade gives advantages
        return real(text, gold) if "\\boxed" in text else (len(text) % 3 == 0)

    R.correct = grader
    try:
        for arm in ("wide", "ctrl"):
            o = SCR / f"run_{arm}"
            shutil.rmtree(o, ignore_errors=True)
            t = time.time()
            res = T.run(args_for(md, arm, o), make_engine=lambda a, mml, th: CpuEngine(a, th))
            par = res["meta"]["parity"]
            log = lines(o / "train_log.jsonl")
            check(f"G7 {arm}: parity passes (pairs, LoRA and gate effects agree, red arms fail); 4 steps logged, ckpts, DONE",
                  par["pass"] and par["start_pass"] and len(log) == 4 and (o / "DONE").exists()
                  and (o / "ckpt_step4.pt").exists() and (o / "ckpt_step2.pt").exists()
                  and any(x["nonzero_adv"] > 0 for x in log), seconds=round(time.time() - t, 1),
                  checks=par["checks"], lora=par["lora_effect"], red=par["red_lora_zeroed"],
                  ratio_abs=[x.get("ratio_logr_abs_mean") for x in log], nonzero=[x["nonzero_adv"] for x in log])
            # killed at step 3: rollouts() raises on its 4th call (steps 0, 1, 2 ran; checkpoint at 2)
            k = SCR / f"kill_{arm}"
            shutil.rmtree(k, ignore_errors=True)
            orig, n = T.rollouts, {"c": 0}

            def dying(*x, **y):
                n["c"] += 1
                if n["c"] == 4:
                    raise RuntimeError("simulated kill")
                return orig(*x, **y)

            T.rollouts = dying
            try:
                T.run(args_for(md, arm, k), make_engine=lambda a, mml, th: CpuEngine(a, th))
                killed = False
            except RuntimeError:
                killed = True
            finally:
                T.rollouts = orig
            before = [x["step"] for x in lines(k / "train_log.jsonl")]
            T.run(args_for(md, arm, k, resume=True), make_engine=lambda a, mml, th: CpuEngine(a, th))
            after = [x["step"] for x in lines(k / "train_log.jsonl")]
            roll_steps = sorted({x["step"] for x in lines(k / "rollouts.jsonl")})
            n_roll = len(lines(k / "rollouts.jsonl"))
            A = torch.load(o / "ckpt_step4.pt")
            B = torch.load(k / "ckpt_step4.pt")
            same = set(A) == set(B) and all(torch.equal(A[x], B[x]) for x in A)
            moved = any(not torch.equal(A[x], torch.load(o / "ckpt_step2.pt")[x]) for x in A)
            meta_r = json.loads((k / "meta_resume2.json").read_text())
            check(f"G7 {arm}: killed at step 3, resumed from step 2: step-4 state bit-equal to the uninterrupted run, "
                  "each step logged once", killed and before == [0, 1, 2] and after == [0, 1, 2, 3] and same and moved
                  and roll_steps == [0, 1, 2, 3] and n_roll == 4 * 8 and (k / "DONE").exists(),
                  before=before, after=after, dropped=meta_r["dropped_log_lines"])
            # red arm: the engine never receives the gate (wide) / the LoRA (ctrl)
            r = SCR / f"red_{arm}"
            shutil.rmtree(r, ignore_errors=True)
            drop = "gate" if arm == "wide" else "lora"
            try:
                T.run(args_for(md, arm, r), make_engine=lambda a, mml, th: CpuEngine(a, th, drop=drop))
                failed, msg = False, ""
            except SystemExit as e:
                failed, msg = "parity FAILED" in str(e), str(e)[:200]
            check(f"G7 red arm {arm}: an engine without the {drop} fails the parity check", failed, msg=msg)
    finally:
        R.correct = real


def pseudo_grader():
    real = R.correct

    def grader(text, gold):  # as in g7: a deterministic pseudo grade on random-token texts
        return real(text, gold) if "\\boxed" in text else (len(text) % 3 == 0)

    return real, grader


def kill_at_step3(fn):
    """Run fn() with rollouts() raising on its 4th call (steps 0, 1, 2 ran; checkpoint at 2). True if it died."""
    orig, n = T.rollouts, {"c": 0}

    def dying(*x, **y):
        n["c"] += 1
        if n["c"] == 4:
            raise RuntimeError("simulated kill")
        return orig(*x, **y)

    T.rollouts = dying
    try:
        fn()
        return False
    except RuntimeError:
        return True
    finally:
        T.rollouts = orig


def g9(md):
    real, grader = pseudo_grader()
    R.correct = grader
    eng = lambda a, mml, th: CpuEngine(a, th)  # noqa: E731
    try:
        # the shared warm-up: ctrl, correctness only (alpha 0), 2 steps
        wd = SCR / "init_warm"
        shutil.rmtree(wd, ignore_errors=True)
        T.run(args_for(md, "ctrl", wd, steps=2, alpha=0.0), make_engine=eng)
        W = wd / "ckpt_step2.pt"
        warm = torch.load(W)
        # an arm branched from it: wide, reasoning-only length penalty
        o = SCR / "init_wide"
        shutil.rmtree(o, ignore_errors=True)
        res = T.run(args_for(md, "wide", o, init_from=str(W), length_on="reasoning"), make_engine=eng)
        st, meta = res["start_state"], res["meta"]
        torch.manual_seed(3)  # the run's seed: what the model holds before any init
        fm, _ = arms.load_model(md, "wide", device_map={"": "cpu"}, dtype=torch.float32)
        fresh = T.state_of(arms.add_lora(fm, RANK))
        lora_keys = sorted(k for k in st if "gate_w" not in k)
        gate_keys = sorted(k for k in st if "gate_w" in k)
        lora_eq = lora_keys == sorted(warm) and all(torch.equal(st[k], warm[k]) for k in lora_keys)
        gate_eq = bool(gate_keys) and all(torch.equal(st[k], fresh[k]) for k in gate_keys)
        differs = any(not torch.equal(warm[k], fresh[k]) for k in lora_keys)  # the warm-up is not the default init
        oas, ir = meta["opt_at_start"], meta["init_from"]
        fresh_opt = oas["state_entries"] == 0 and oas["sched_last_epoch"] == 0 and oas["start_step"] == 0
        steps = [x["step"] for x in lines(o / "train_log.jsonl")]
        check("G9 init_from: first-step state = warm-up LoRA bit for bit, gate at its usual init, optimizer and "
              "schedule fresh; meta.json records path and sha256",
              lora_eq and gate_eq and differs and fresh_opt and steps == [0, 1, 2, 3] and ir["applied"]
              and ir["sha256"] == T.sha256_file(W) and ir["path"] == str(W.resolve())
              and json.loads((o / "meta.json").read_text())["init_from"]["sha256"] == ir["sha256"]
              and meta["length_on"] == "reasoning",
              lora_eq=lora_eq, gate_eq=gate_eq, warm_differs_from_default=differs, opt_at_start=oas, steps=steps)
        C = torch.load(o / "ckpt_step4.pt")
        full = set(C) == set(st) and bool(gate_keys)
        moved = all(not torch.equal(C[k], warm[k]) for k in lora_keys if "lora_B" in k)
        canon = lambda k: k[k.find("layers."):]  # noqa: E731  (readout_vllm.py's key mapping)
        readable = {canon(k) for k in C} == {canon(n) for n in fresh}
        check("G9 init_from: the arm checkpoint holds the full LoRA (init plus update, every lora_B moved off the "
              "warm-up) and the gate, the tensor set readout_vllm.py --ckpt loads", full and moved and readable,
              full=full, moved=moved, readable=readable)
        # red arms: mismatched checkpoints are refused
        bad_dir = SCR / "init_bad"
        shutil.rmtree(bad_dir, ignore_errors=True)
        bad_dir.mkdir(parents=True)
        k0 = lora_keys[0]
        ka = next(k for k in lora_keys if "lora_A" in k)
        bads = {"missing_key": {k: v for k, v in warm.items() if k != k0},
                "rank_8": {k: (v[:8] if k == ka else v) for k, v in warm.items()},
                "gate_added": {**warm, gate_keys[0]: torch.zeros_like(fresh[gate_keys[0]])}}
        refused = {}
        for name, sd in bads.items():
            torch.save(sd, bad_dir / f"{name}.pt")
            torch.manual_seed(3)
            m, _ = arms.load_model(md, "wide", device_map={"": "cpu"}, dtype=torch.float32)
            m = arms.add_lora(m, RANK)
            try:
                T.init_lora_from(m, bad_dir / f"{name}.pt")
                refused[name] = False
            except SystemExit as e:
                refused[name] = str(e)[:90]
        rr = SCR / "init_red_run"
        shutil.rmtree(rr, ignore_errors=True)
        try:
            T.run(args_for(md, "wide", rr, init_from=str(bad_dir / "rank_8.pt"), length_on="reasoning"),
                  make_engine=eng)
            refused["run_rank_8"] = False
        except SystemExit as e:
            refused["run_rank_8"] = "--init_from" in str(e) and str(e)[:90]
        torch.manual_seed(3)
        m, _ = arms.load_model(md, "wide", device_map={"": "cpu"}, dtype=torch.float32)
        good = T.init_lora_from(arms.add_lora(m, RANK), W)["applied"]
        check("G9 red arms: a missing key, a rank-8 shape and an added gate tensor are refused (run() too); the "
              "matching checkpoint loads", good and all(refused.values()), refused=refused)
        # resume ignores init_from: killed at step 3, resumed with a refused checkpoint as --init_from
        k = SCR / "init_kill"
        shutil.rmtree(k, ignore_errors=True)
        killed = kill_at_step3(lambda: T.run(args_for(md, "wide", k, init_from=str(W), length_on="reasoning"),
                                              make_engine=eng))
        T.run(args_for(md, "wide", k, init_from=str(bad_dir / "rank_8.pt"), length_on="reasoning", resume=True),
              make_engine=eng)
        B = torch.load(k / "ckpt_step4.pt")
        same = set(B) == set(C) and all(torch.equal(B[x], C[x]) for x in C)
        mr = json.loads((k / "meta_resume2.json").read_text())["init_from"]
        check("G9 resume ignores --init_from once the run has checkpoints: resumed with a refused checkpoint, the "
              "step-4 state equals the uninterrupted run bit for bit",
              killed and same and mr["applied"] is False and mr["original"]["sha256"] == T.sha256_file(W),
              killed=killed, same=same, init_from_resume=mr)
    finally:
        R.correct = real


def g10(md):
    import readout_vllm as RV
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(md)
    th = R.Thinker(tok)
    E, IM, a, b = th.end_think, th.im_end, 100, 101
    p = th.prompt("1+1?")
    # one group: short reasoning + long answer, long reasoning + short answer, a forced close, a silent stop
    g1 = [[a, a, E] + [b] * 9 + [IM], [a] * 7 + [E, b, IM], [a] * 8, [a] * 5 + [IM]]
    rolls, need = T.assemble([p] * 4, g1, [[-0.5] * len(g) for g in g1], th)
    g2 = [[b, b, IM], [b] * 6 + [IM]]
    T.finish(rolls, need, g2, [[-0.5] * len(g) for g in g2], lambda t: tok.decode(t, skip_special_tokens=True))
    think = [RV.split_trace(g, E, th.stops)["think_len"] for g in g1]
    ans_ro = [len(g) - t if E in g else len(o) for g, t, o in zip(g1, think, [None, None] + g2)]
    reas = T.reward_lengths(rolls, "reasoning")
    comp = T.reward_lengths(rolls, "completion")
    ans = [T.answer_tokens(r) for r in rolls]
    check("G10 reasoning count = readout_vllm.split_trace think_len (closed, forced, silent stop); answer count = "
          "its answer tokens", reas == think and ans == ans_ro, reasoning=reas, readout=think, answer=ans,
          readout_answer=ans_ro, completion=comp)
    corr = [True] * 4
    rr, _ = T.group_rewards(corr, reas, 4, 0.2)
    rc, _ = T.group_rewards(corr, comp, 4, 0.2)
    order = lambda v: sorted(range(4), key=lambda i: (v[i], i))  # noqa: E731
    differ_ord = order(reas) != order(comp)
    ok = rr == R.length_reward(corr, reas, 0.2) and max(range(4), key=lambda i: rr[i]) == 0
    red = rc != rr and max(range(4), key=lambda i: rc[i]) != 0
    check("G10 length_on reasoning: reward on reasoning tokens (shortest reasoning wins); red arm: completion mode "
          "ranks the same group differently", differ_ord and ok and red, rew_reasoning=rr, rew_completion=rc)
    # a real CPU run with --length_on reasoning
    real, grader = pseudo_grader()
    R.correct = grader
    try:
        o = SCR / "len_reasoning"
        shutil.rmtree(o, ignore_errors=True)
        T.run(args_for(md, "ctrl", o, length_on="reasoning", answer_max=8),
              make_engine=lambda a_, mml, t_: CpuEngine(a_, t_, varlen=True))
    finally:
        R.correct = real
    ro = lines(o / "rollouts.jsonl")
    tl = lines(o / "train_log.jsonl")
    from_reas, from_comp, n_grp = True, 0, 0
    for i in range(0, len(ro), 4):
        g = ro[i : i + 4]
        c = [x["correct"] for x in g]
        want = R.length_reward(c, [x["reasoning"] for x in g], 0.2)
        alt = R.length_reward(c, [x["completion"] for x in g], 0.2)
        from_reas &= all(abs(x["reward"] - w) < 1e-12 for x, w in zip(g, want))
        from_comp += any(abs(x["reward"] - w) > 1e-9 for x, w in zip(g, alt))
        n_grp += 1
    logged = all("answer_mean" in x and "reasoning_mean" in x and "length_mean" in x for x in tl) and all(
        abs(x["length_mean"] - x["reasoning_mean"]) < 1e-9 for x in tl)
    check("G10 run with --length_on reasoning: logged rewards recompute from reasoning counts; red arm: from "
          "completion counts they disagree in at least one group; reasoning and answer means logged per step",
          from_reas and from_comp > 0 and logged and n_grp == 8, groups=n_grp, groups_completion_disagrees=from_comp,
          log_keys=sorted(tl[0]) if tl else None)


G8_CODE = r"""
import os, sys, types, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["WG_VLLM_CACHE"] = sys.argv[2]
sys.path.insert(0, sys.argv[1])
import torch, torch.nn as nn
import gate_vllm as NV
NV.install("stock", lora_rank=4)
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
C = Q.QwenGatedDeltaNetAttention
r, H, kd, vd, nv, T = 4, 32, 8, 16, 4, 5
qkv = 2 * kd + vd
g = torch.Generator().manual_seed(0)
R = lambda *s: torch.randn(*s, generator=g)
m = C.__new__(C); nn.Module.__init__(m)
for n, shp in (("wg_lA", (3 * r, H)), ("wg_lBqkv", (qkv, r)), ("wg_lBb", (nv, r)), ("wg_lBa", (nv, r)),
               ("wg_lAo", (r, vd)), ("wg_lBo", (H, r))):
    setattr(m, n, nn.Parameter(torch.zeros(*shp), requires_grad=False))
m._wg_r, m._wg_lora = r, True
root = nn.Module(); holder = nn.Module(); holder.linear_attn = m; root.layers = nn.ModuleList([holder])
pre = "base_model.model.model.layers.0.linear_attn."
P = {p: (R(r, din), R(dout, r)) for p, din, dout in (("in_proj_qkv", H, qkv), ("in_proj_b", H, nv), ("in_proj_a", H, nv), ("out_proj", vd, H))}
state = {}
for p, (A, B) in P.items():
    state[pre + p + ".lora_A.default.weight"] = A
    state[pre + p + ".lora_B.default.weight"] = B
rec = NV.set_lora(root, state, gate=False)
x, y = R(T, H), R(T, vd)
qkvz, ba, out = R(T, qkv + vd), R(T, 2 * nv), R(T, H)
q2, ba2 = C._wg_lora_in(m, x, qkvz, ba)
o2 = C._wg_lora_out(m, y, out)
d = lambda p, inp: 2.0 * inp @ P[p][0].T @ P[p][1].T
ref_q = torch.cat([qkvz[:, :qkv] + d("in_proj_qkv", x), qkvz[:, qkv:]], -1)
ref_ba = torch.cat([ba[:, :nv] + d("in_proj_b", x), ba[:, nv:] + d("in_proj_a", x)], -1)
ref_o = out + d("out_proj", y)
err = max(float((q2 - ref_q).abs().max()), float((ba2 - ref_ba).abs().max()), float((o2 - ref_o).abs().max()))
swap = torch.cat([ba[:, :nv] + d("in_proj_a", x), ba[:, nv:] + d("in_proj_b", x)], -1)
red = float((ba2 - swap).abs().max())
NV.set_lora(root, state, gate=False, lora=False)
q3, ba3 = C._wg_lora_in(m, x, qkvz, ba)
zeroed = float((q3 - qkvz).abs().max()) + float((ba3 - ba).abs().max())
rewired = C.forward_cuda.__code__.co_filename.endswith("wg_forward_cuda.py") and "_wg_lora_out" in C._output_projection.__code__.co_names
print(json.dumps({"err": err, "red_swapped_ba": red, "zeroed": zeroed, "modules": rec["lora_modules"], "rewired": rewired}))
"""


def g8():
    """G8 plugin unmerged LoRA: _wg_lora_in / _wg_lora_out equal PEFT's base + 2 B(A x) on the qkv rows (z
    untouched), on b and a, and on out_proj, after set_lora maps a PEFT state; lora=False zeroes it; the class methods
    are the rewritten ones. Red arm: the b/a slots swapped disagree."""
    r = subprocess.run([sys.executable, "-c", G8_CODE, str(HERE.parent / "vllm"), str(SCR / "vllm_cache")],
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        check("G8 plugin LoRA arithmetic", False, rc=r.returncode, err=r.stderr[-400:])
        return
    check("G8 plugin LoRA: engine insertions equal base + 2 B(A x) per projection; lora=False zeroes; rewired",
          d["err"] < 1e-4 and d["zeroed"] == 0.0 and d["modules"] == 4 and d["rewired"], **d)
    check("G8 red arm: b and a LoRA slots swapped disagree", d["red_swapped_ba"] > 100 * max(d["err"], 1e-6), **d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    md = test_cpu.tiny_model_dir()
    want = set(a.only.split(",")) if a.only else {"g1", "g2", "g3", "g5", "g6", "g7", "g8", "g9", "g10"}
    for name, fn in (("g1", g1), ("g2", lambda: g2(md)), ("g3", lambda: g3(md)), ("g5", g5), ("g6", g6),
                     ("g7", lambda: g7(md)), ("g8", g8), ("g9", lambda: g9(md)), ("g10", lambda: g10(md))):
        if name in want:
            fn()
    out = {"pass": all(c["pass"] for c in checks), "n_checks": len(checks), "checks": checks,
           "seconds": round(time.time() - t0, 1), "torch": torch.__version__, "host": os.uname().nodename,
           "cmd": " ".join(sys.argv)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print("ALL PASS" if out["pass"] else "SOME FAILED", len(checks))


if __name__ == "__main__":
    main()
