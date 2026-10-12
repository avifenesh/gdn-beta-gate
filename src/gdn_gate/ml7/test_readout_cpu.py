#!/usr/bin/env python3
"""CPU checks for readout_vllm.py and compare_readout.py. Each check has a red arm that must fail it.

1. prefix: run_readout against a fake engine: phase-2 contexts are exactly prompt + reasoning[:c] + force, the prior
   is reasoning[:c] + force, closed-within-budget traces keep the phase-1 answer, silent stops drop the stop token,
   budgets are prefixes of one trace. Red: --red half must fail the same assertions.
2. presence (needs vllm importable; run with the vLLM venv): PriorPresence + vLLM's output-token presence penalty
   give the same logits as rl_common.PresencePenalty(prior=...) step by step. Red: vLLM's penalty alone, and a
   prior penalty that ignores the generated tokens (double count), must both differ.
3. metrics: norm_auc and tokens_at_acc on hand-computed curves.
4. compare: same per-item rates -> AGREE (and the false alarm rate over 300 synthetic repeats stays near alpha);
   accuracy read at half the budget -> DISAGREE; HF readout_len.py JSON loads into the same rows.
Usage: test_readout_cpu.py [--skip_presence] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readout_vllm as RV  # noqa: E402
import compare_readout as CR  # noqa: E402

END, STOP, NL = 1, 2, 3


class FakeThinker:
    end_think, stops, force = END, [STOP], [NL, END, NL, NL]

    def prompt(self, problem):
        return [100 + ord(ch) % 50 for ch in problem]


class FakeEngine:
    """Phase 1: scripted traces by first prompt token. Phase 2: the answer encodes the context length."""

    def __init__(self, traces):
        self.traces, self.calls = traces, []

    def generate(self, contexts, specs):
        self.calls.append((contexts, specs))
        out = []
        for c, s in zip(contexts, specs):
            if s["prior"] is None:
                out.append(list(self.traces[tuple(c)])[: s["max_tokens"]])
            else:
                out.append([9000 + len(c), STOP])
        return out


def check_prefix(red: str) -> dict:
    th = FakeThinker()
    caps, amax = [4, 8, 16], 3
    items = [{"id": "closed5", "problem": "a", "gold": "A"}, {"id": "closed12", "problem": "b", "gold": "B"},
             {"id": "unclosed", "problem": "c", "gold": "C"}, {"id": "silent6", "problem": "d", "gold": "D"}]
    think = lambda n, base: [base + k for k in range(n)]  # noqa: E731
    tr = {tuple(th.prompt("a")): think(5, 500) + [END, 77, STOP],
          tuple(th.prompt("b")): think(12, 600) + [END, 78, STOP],
          tuple(th.prompt("c")): think(40, 700),
          tuple(th.prompt("d")): think(6, 800) + [STOP]}
    eng = FakeEngine(tr)
    decode = lambda t: " ".join(map(str, t))  # noqa: E731
    grade = lambda text, gold: "77" in text.split() or "78" in text.split()  # noqa: E731  (phase-1 answers)
    rows, traces, _ = RV.run_readout(eng, items, th, caps, amax, 1, 7, grade, decode, red)
    R = {r["id"]: r for r in rows}
    ctx2, spec2 = eng.calls[1]
    errs = []
    if eng.calls[0][1][0]["max_tokens"] != max(caps) + amax:
        errs.append("phase-1 length")
    exp_forced = {("closed5", 4), ("closed12", 4), ("closed12", 8), ("unclosed", 4), ("unclosed", 8), ("unclosed", 16),
                  ("silent6", 4), ("silent6", 8), ("silent6", 16)}
    got = set()
    seedmap = {RV.seed_of(7, it["id"], 0, b): (it["id"], b) for it in items for b in caps}
    by_id = {it["id"]: it for it in items}
    for c, s in zip(ctx2, spec2):
        iid, b = seedmap[s["seed"]]
        got.add((iid, b))
        p = th.prompt(by_id[iid]["problem"])
        full = tr[tuple(p)]
        body = full[: full.index(END)] if END in full else (full[:-1] if full[-1] == STOP else full)
        if c != p + body[:b] + th.force:
            errs.append(f"{iid}@{b}: phase-2 context is not prompt + reasoning[:{b}] + force")
        if s["prior"] != body[:b] + th.force:
            errs.append(f"{iid}@{b}: prior is not reasoning[:{b}] + force")
    if got != exp_forced:
        errs.append(f"forced set {sorted(got)} != {sorted(exp_forced)}")
    if R["closed5"]["caps"]["8"][:3] != [1, 5, 0] or R["closed12"]["caps"]["16"][:3] != [1, 12, 0]:
        errs.append("kept phase-1 answer wrong")
    if R["unclosed"]["caps"]["8"][1:3] != [8, 1] or R["silent6"]["think_len"] != 6 or not R["silent6"]["silent_stop"]:
        errs.append("forced / silent bookkeeping wrong")
    if R["closed5"]["caps"]["4"][0] != 0:
        errs.append("forced answer graded as the phase-1 answer")
    return {"red": red, "errors": errs, "pass": not errs}


def check_presence() -> dict:
    """PriorPresence (Model Runner V2 interface) against rl_common.PresencePenalty(prior=...), step by step, with a fake
    request state: slots out of order and reshuffled rows each step, one request resumed after preemption (partial
    output already in its slot), one request without a prior (must stay untouched)."""
    import types

    import numpy as np
    import torch
    from vllm import SamplingParams

    import rl_common as R

    PP = RV.make_prior_presence()
    V, p, n_slots, maxlen = 64, 1.5, 8, 64
    g = torch.Generator().manual_seed(0)
    # slot: (prompt, prior, generated sequence, tokens already generated before a preemption)
    reqs = {5: ([50, 51, 52], [5, 6, 7, 7, 9], [6, 30, 5, 31, 30, 9], 0),
            2: ([53, 54], [11, 12], [40, 11, 12, 41, 42, 40], 0),
            7: ([55], [20], [21, 22, 20, 23, 24, 25], 2),      # resumed: 21, 22 were generated before preemption
            0: ([56, 57], None, [3, 3, 4, 50, 6, 7], 0)}       # phase-1 style request: no prior, no processor rows
    T = len(reqs[5][2])

    def run(variant):
        st = types.SimpleNamespace(
            max_num_reqs=n_slots, vocab_size=V, device=torch.device("cpu"),
            prompt_len=types.SimpleNamespace(np=np.zeros(n_slots, np.int32)),
            prefill_len=types.SimpleNamespace(np=np.zeros(n_slots, np.int32)),
            all_token_ids=types.SimpleNamespace(gpu=torch.zeros(n_slots, maxlen, dtype=torch.int32)))
        lp = PP(None, st)
        for slot, (prompt, prior, gen, done) in reqs.items():
            ctx_toks = prompt + (prior or [])
            st.prompt_len.np[slot] = len(ctx_toks)
            st.prefill_len.np[slot] = len(ctx_toks) + done
            full = ctx_toks + gen[:done]
            st.all_token_ids.gpu[slot, : len(full)] = torch.tensor(full, dtype=torch.int32)
            ea = {"wg_prior": prior, "wg_prior_p": p} if (prior and variant != "native_only") else None
            sp = SamplingParams(presence_penalty=0.0 if ea else p, extra_args=ea)
            PP.validate_params(sp)
            lp.add_request(slot, sp)
        st.prompt_len.gpu = torch.from_numpy(st.prompt_len.np.copy())
        lp.apply_staged_writes()
        worst, untouched = 0.0, True
        rng = random.Random(3)
        for step in range(T - 2):
            slots = list(reqs)
            rng.shuffle(slots)
            inp, pos, outs = [], [], {}
            for slot in slots:
                prompt, prior, gen, done = reqs[slot]
                k = done + step                         # tokens generated so far
                L0 = len(prompt) + len(prior or [])
                inp.append(gen[k - 1] if k > 0 else (prompt + (prior or []))[-1])
                pos.append(L0 + k - 1)
                outs[slot] = gen[:k]
            ctx = types.SimpleNamespace(expanded_idx_mapping=torch.tensor(slots, dtype=torch.int32),
                                        idx_mapping_np=np.array(slots), input_ids=torch.tensor(inp, dtype=torch.int32),
                                        pos=torch.tensor(pos, dtype=torch.int64))
            base = torch.randn(len(slots), V, generator=g)
            x = base.clone()
            for r, slot in enumerate(slots):  # vLLM's own presence penalty (output tokens), for requests that use it
                prior = reqs[slot][1]
                if not prior or variant in ("native_only", "double"):
                    for t in set(outs[slot]):
                        x[r, t] -= p
            x = lp.apply(x, ctx)
            for r, slot in enumerate(slots):
                prompt, prior, gen, done = reqs[slot]
                ref = R.PresencePenalty(p, 1, prior=[prior] if prior else None)(
                    torch.tensor([[0] + outs[slot]]), base[r: r + 1].clone())
                d = float((x[r] - ref[0]).abs().max())
                if prior:
                    worst = max(worst, d)
                else:
                    untouched &= d == 0.0
        return worst, untouched

    ours, untouched = run("ours")
    native, _ = run("native_only")
    double, _ = run("double")
    ok = ours == 0.0 and untouched and native > 0 and double > 0
    return {"max_abs_diff_ours": ours, "no_prior_request_untouched": untouched, "red_native_only": native,
            "red_double_count": double, "pass": ok}


def check_metrics() -> dict:
    caps = [256, 512, 1024, 2048, 4096, 8192]
    errs = []
    if abs(RV.norm_auc(caps, [1] * 6) - 1) > 1e-12 or abs(RV.norm_auc(caps, [0] * 6)) > 1e-12:
        errs.append("auc bounds")
    if abs(RV.norm_auc(caps, [0, 0, 0, 0, 0, 1]) - 0.1) > 1e-12:  # one half-unit triangle over 5 units
        errs.append("auc triangle")
    if abs(RV.norm_auc(caps, [0, 0.2, 0.4, 0.6, 0.8, 1.0]) - 0.5) > 1e-12:
        errs.append("auc ramp")
    t = RV.tokens_at_acc(caps, [0.1, 0.2, 0.4, 0.6, 0.7, 0.8], 0.5)
    if abs(t - 2 ** 10.5) > 1e-6:
        errs.append(f"tokens_at_acc {t}")
    if RV.tokens_at_acc(caps, [0.1] * 6, 0.5) is not None:
        errs.append("tokens_at_acc unreachable")
    # red: a curve one budget late must change the area
    if abs(RV.norm_auc(caps, [0, 0.2, 0.4, 0.6, 0.8, 1.0]) - RV.norm_auc(caps, [0, 0, 0.2, 0.4, 0.6, 0.8])) < 0.1:
        errs.append("auc insensitive to a shifted curve")
    return {"errors": errs, "pass": not errs}


def synth(n_items, samples, rng, shift_half=False, base=None):
    caps = [256, 512, 1024, 2048, 4096, 8192]
    base = base or [(rng.lognormvariate(6.8, 0.6), rng.random() * 0.4 + 0.55) for _ in range(n_items)]
    rows = []
    for i, (L, pmax) in enumerate(base):
        for s in range(samples):
            tl = int(L * rng.lognormvariate(0, 0.35))
            r = {"id": f"q{i}", "sample": s, "think_len": tl, "closed": int(tl <= 8192), "silent_stop": 0, "caps": {}}
            for c in caps:
                eff = c // 2 if shift_half else c
                pc = pmax * (1 if tl <= eff else 0.25 * min(1, eff / tl))
                r["caps"][str(c)] = [int(rng.random() < pc), min(tl, c), int(tl > eff), 1, 1]
            rows.append(r)
    return rows, caps, base


def check_compare() -> dict:
    rng = random.Random(1)
    out = {}
    A, caps, base = synth(50, 8, rng)
    B, _, _ = synth(50, 1, rng, base=base)
    Rd, _, _ = synth(50, 1, rng, shift_half=True, base=base)
    g, r = CR.compare(A, B, caps), CR.compare(A, Rd, caps)
    out["green_agree"], out["red_disagree"] = g["agree"], r["disagree"]
    out["red_z"] = {k: round(v["z"], 2) for k, v in r["tests"].items()}
    fa = 0
    for k in range(300):
        rr = random.Random(100 + k)
        A, caps, base = synth(50, 8, rr)
        B, _, _ = synth(50, 1, rr, base=base)
        fa += not CR.compare(A, B, caps)["agree"]
    out["false_alarm_rate_300"] = fa / 300
    power = 0
    for k in range(100):
        rr = random.Random(1000 + k)
        A, caps, base = synth(50, 8, rr)
        Rd, _, _ = synth(50, 1, rr, shift_half=True, base=base)
        power += CR.compare(A, Rd, caps)["disagree"]
    out["red_detect_rate_100"] = power / 100
    # HF readout_len.py JSON -> rows
    hf = {"args": {"sample_seed": 1}, "think_len": {"q0": 300, "q1": 9000},
          "items": {str(c): {"q0": {"correct": 1, "reasoning": min(300, c), "forced": int(c < 300), "answer_ended": 1},
                             "q1": {"correct": 0, "reasoning": c, "forced": 1, "answer_ended": 1}} for c in caps}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(hf, f)
    rows, c2, _ = CR.load_rows([f.name])
    Path(f.name).unlink()
    out["hf_load"] = c2 == caps and len(rows) == 2 and rows[1]["caps"]["8192"][:3] == [0, 8192, 1]
    out["pass"] = (out["green_agree"] and out["red_disagree"] and out["false_alarm_rate_300"] <= 0.10
                   and out["red_detect_rate_100"] >= 0.8 and out["hf_load"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_presence", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {"prefix": check_prefix(""), "prefix_red_half": check_prefix("half"), "prefix_red_noclose": check_prefix("noclose"),
           "metrics": check_metrics(),
           "compare": check_compare()}
    res["prefix_red_half"]["pass_expected"] = False
    if not a.skip_presence:
        res["presence"] = check_presence()
    ok = (res["prefix"]["pass"] and not res["prefix_red_half"]["pass"] and not res["prefix_red_noclose"]["pass"]
          and res["metrics"]["pass"]
          and res["compare"]["pass"] and (a.skip_presence or res["presence"]["pass"]))
    res["verdict"] = "PASS" if ok else "FAIL"
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
