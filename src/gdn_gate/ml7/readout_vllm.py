#!/usr/bin/env python3
"""Accuracy vs reasoning budget on vLLM, with the prefix method of broad/readout_len.py (the HF readout).

Method (identical to readout_len.py, batched in vLLM):
* Phase 1: one long trace per (item, sample) at the Qwen3.5 card's thinking sampling (rl_common.VENDOR: temperature
  1.0, top_p 0.95, top_k 20, presence 1.5), up to max(caps) + --answer_max tokens, prompt = rl_common.Thinker.prompt
  (enable_thinking, boxed-answer suffix).
* For each budget c: a trace whose thinking closed (</think>) within c tokens keeps its phase-1 answer; any other
  trace is cut at c reasoning tokens, closed with the same text ("\\n</think>\\n\\n", Thinker.force) and its answer is
  sampled in phase 2 (up to --answer_max tokens). Smaller budgets are exact prefixes of the one long trace.
* Presence penalty in phase 2 covers the whole completion, as readout_len.py does (rl_common.PresencePenalty with
  prior = the kept reasoning + the forced close). vLLM's own penalty sees output tokens only, so phase-2 requests run
  with it off and the PriorPresence logits processor (Model Runner V2 interface) applies the penalty to prior OR
  generated tokens, once per token.
* Grading: rl_common.correct (last \\boxed{}, math_verify; the run refuses to start without math_verify).

Arms: --arm stock (untouched model, stock vLLM). A trained checkpoint (train_grpo.py trainable state, LoRA r16 with or
without gate tensors) goes through the gate plugin installed with lora_rank: the LoRA stays UNMERGED in the engine
(gate_vllm.set_lora, the trainer's sync path; a bf16 merge drops most of a small RL LoRA move) with the gate for
--arm wide|wide1, the stock kernels for --arm ctrl; --strip loads a gated checkpoint on the stock engine without its
gate. Before any generation the engine must reproduce the trainer's log-probs for the loaded checkpoint:
train_grpo_vllm.parity_check (floors, probe LoRA / gate effects with red arms, and the checkpoint's own decode pair
and effect against the untouched model), with the training forward built in this process (HF model + PEFT LoRA,
built before the engine); the readout refuses to run on a failure and records the result under "parity". Set
WG_ML7_PYLIB to a dir with fla so the training forward runs the trainer's kernels.

Output (--out JSON, schema wg-readout-v1): receipt (command, versions, git describe, model revision, host, GPU,
code and data sha256, wall times), per-budget summary (accuracy with item-clustered SE, mean reasoning tokens, cut-off
share, missing-box share, answer-not-ended share), natural length (reasoning tokens at the largest budget, closed
share, silent stops), the primary metric (normalized area under accuracy vs log2 budget, with SE), and `rows`: one
record per (item, sample) with think_len, closed, silent_stop and per budget [correct, reasoning, forced,
answer_ended, boxed]. A sidecar OUT.traces.jsonl.gz keeps the phase-1 reasoning tokens and every answer text (error
analysis, regrading). load_rows() in compare_readout.py reads this file and readout_len.py JSON alike.

Usage:
  readout_vllm.py --model M --data val.jsonl --out OUT.json [--caps 256,...,8192] [--samples 1] [--stride 1 --limit 0]
                  [--arm stock | --arm wide|wide1|ctrl --ckpt trainable.pt [--strip]] [--seed 20261006]
  --red half|noclose   (red arms, never for results: half cuts and forces at c//2 while labelling the budget c;
                        noclose cuts at c but omits the forced close, so the answer is sampled mid-thought)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEXT = HERE.parent
for p in (NEXT / "vllm", NEXT / "broad"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CAPS = "256,512,1024,2048,4096,8192"
SCHEMA = "wg-readout-v1"


# ------------------------------------------------------------------------------------------------ pure logic (CPU)
def seed_of(base: int, *parts) -> int:
    """Stable per-request seed: the same (seed, item, sample[, budget]) draws the same seed in any subset or order."""
    return int(hashlib.sha256(":".join(map(str, (base,) + parts)).encode()).hexdigest()[:12], 16)


def split_trace(g: list[int], end_think: int, stops) -> dict:
    """readout_len.py phase-1 bookkeeping, line for line."""
    closed = end_think in g
    k = g.index(end_think) if closed else len(g)
    silent = (not closed) and bool(g) and g[-1] in stops
    think = g[:k] if closed else (g[:-1] if silent else g)
    return {"closed": closed, "think_len": len(think), "silent_stop": silent, "think": think}


def plan(rec: dict, c: int, force: list[int], red: str = "") -> tuple[str, int]:
    """('keep', reasoning) when the phase-1 answer stands at budget c, else ('force', cut)."""
    eff = c // 2 if red == "half" else c
    if rec["closed"] and rec["think_len"] <= eff:
        return "keep", rec["think_len"]
    return "force", eff


def norm_auc(caps, accs) -> float:
    """Trapezoid area under accuracy vs log2(budget), divided by the log2 span: 1.0 = right at every budget."""
    x = [math.log2(c) for c in caps]
    area = sum((x[j + 1] - x[j]) * (accs[j] + accs[j + 1]) / 2 for j in range(len(x) - 1))
    return area / (x[-1] - x[0])


def tokens_at_acc(caps, accs, target) -> float | None:
    """Smallest budget (log2-interpolated) at which the curve reaches `target`; None if it never does."""
    if accs[0] >= target:
        return float(caps[0])
    for j in range(len(caps) - 1):
        a0, a1 = accs[j], accs[j + 1]
        if a1 >= target > a0:
            f = (target - a0) / (a1 - a0)
            return 2 ** (math.log2(caps[j]) + f * (math.log2(caps[j + 1]) - math.log2(caps[j])))
    return None


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _se_clustered(rows, val) -> float:
    """SE of the mean of val(row), rows clustered by item (samples of one item are not independent draws of items)."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["id"], []).append(val(r))
    m = [_mean(v) for v in by.values()]
    n = len(m)
    if n < 2:
        return float("nan")
    mu = _mean(m)
    return math.sqrt(sum((x - mu) ** 2 for x in m) / (n - 1) / n)


def summarize(rows: list[dict], caps: list[int]) -> dict:
    """rows: normalized records {id, sample, think_len, closed, silent_stop, caps: {str(c): [correct, reasoning,
    forced, answer_ended, boxed]}} (boxed/closed may be None for HF receipts)."""
    big = max(caps)
    per = {}
    for c in caps:
        k = str(c)
        v = [r["caps"][k] for r in rows]
        boxed = [x[4] for x in v if x[4] is not None]
        per[k] = {"acc": _mean([x[0] for x in v]), "acc_se": _se_clustered(rows, lambda r: r["caps"][k][0]),
                  "reasoning_mean": _mean([x[1] for x in v]), "forced_frac": _mean([x[2] for x in v]),
                  "answer_unended_frac": _mean([1 - x[3] for x in v]),
                  "boxed_missing_frac": _mean([1 - b for b in boxed]) if boxed else None, "n": len(v)}
    accs = [per[str(c)]["acc"] for c in caps]
    auc_row = lambda r: norm_auc(caps, [r["caps"][str(c)][0] for c in caps])  # noqa: E731
    nat = sorted(min(r["think_len"], big) for r in rows)
    closed = [r["closed"] for r in rows if r.get("closed") is not None]
    return {
        "per_cap": per,
        "norm_auc_log2": norm_auc(caps, accs), "norm_auc_log2_se": _se_clustered(rows, auc_row),
        "tokens_at_own_max_acc": tokens_at_acc(caps, accs, accs[-1]),
        "natural": {"reasoning_mean_at_max": _mean(nat), "reasoning_mean_at_max_se": _se_clustered(rows, lambda r: min(r["think_len"], big)),
                    "median": nat[len(nat) // 2], "p90": nat[min(len(nat) - 1, int(0.9 * len(nat)))],
                    "cut_off_share_at_max": per[str(big)]["forced_frac"],
                    "closed_share_phase1": _mean(closed) if closed else None,
                    "silent_stops": sum(bool(r.get("silent_stop")) for r in rows)},
        "n_rows": len(rows), "n_items": len({r["id"] for r in rows}),
    }


def run_readout(engine, items: list[dict], th, caps: list[int], answer_max: int, samples: int, seed: int,
                grade, decode, red: str = "") -> tuple[list[dict], list[dict], dict]:
    """The prefix method against any engine with
         engine.generate(contexts: list[list[int]], specs: list[dict]) -> list[list[int]]
       spec = {"max_tokens", "seed", "prior": list[int] | None}. Returns (rows, traces, timing)."""
    big = max(caps)
    jobs = [(it, s) for it in items for s in range(samples)]
    prompts = {it["id"]: th.prompt(it["problem"]) for it in items}
    t0 = time.time()
    g1 = engine.generate([prompts[it["id"]] for it, s in jobs],
                         [{"max_tokens": big + answer_max, "seed": seed_of(seed, it["id"], s), "prior": None}
                          for it, s in jobs])
    t_p1 = time.time() - t0
    recs = []
    for (it, s), g in zip(jobs, g1):
        recs.append({"it": it, "s": s, "g": g, **split_trace(g, th.end_think, th.stops)})
    ctx, specs, where = [], [], []
    for ri, x in enumerate(recs):
        for c in caps:
            how, cut = plan(x, c, th.force, red)
            if how == "force":
                prior = x["think"][:cut] + ([] if red == "noclose" else th.force)
                ctx.append(prompts[x["it"]["id"]] + prior)
                specs.append({"max_tokens": answer_max, "seed": seed_of(seed, x["it"]["id"], x["s"], c), "prior": prior})
                where.append((ri, c, cut))
    t1 = time.time()
    g2 = engine.generate(ctx, specs) if ctx else []
    t_p2 = time.time() - t1
    forced_out = {(ri, c): (cut, o) for (ri, c, cut), o in zip(where, g2)}
    rows, traces = [], []
    for ri, x in enumerate(recs):
        row = {"id": x["it"]["id"], "sample": x["s"], "think_len": x["think_len"], "closed": int(x["closed"]),
               "silent_stop": int(x["silent_stop"]), "phase1_tokens": len(x["g"]), "caps": {}}
        texts = {}
        for c in caps:
            if (ri, c) in forced_out:
                cut, o = forced_out[(ri, c)]
                text = decode(o)
                ended = int(bool(o) and o[-1] in th.stops)
                reasoning, forced = min(x["think_len"], cut), 1
                ans_tok = len(o)  # sampled answer tokens after the forced close
            else:
                text = decode(x["g"][x["think_len"]:])
                ended = int(x["g"][-1] in th.stops)
                reasoning, forced = x["think_len"], 0
                ans_tok = len(x["g"]) - x["think_len"]  # natural close: the closing tag plus the answer
            # 6th element (2026-10-09): answer tokens, so reasoning carried into the answer is visible (step 14)
            row["caps"][str(c)] = [int(grade(text, x["it"]["gold"])), reasoning, forced, ended,
                                   int(boxed_of(text) is not None), ans_tok]
            texts[str(c)] = text
        rows.append(row)
        traces.append({"id": x["it"]["id"], "sample": x["s"], "think": x["think"], "answers": texts})
    timing = {"phase1_s": t_p1, "phase2_s": t_p2, "phase1_tokens": sum(len(x["g"]) for x in recs),
              "phase2_tokens": sum(len(o) for o in g2), "phase2_requests": len(ctx)}
    return rows, traces, timing


def boxed_of(text):
    import rl_common as R

    return R.boxed(text)


# ------------------------------------------------------------------------------------------------ vLLM side
def make_prior_presence():
    """vLLM Model Runner V2 logits processor carrying the whole phase-2 presence penalty: subtract p from every token
    in (prior tokens of the request, from extra_args wg_prior) OR (tokens the request has generated), once per
    token, exactly as rl_common.PresencePenalty(prior=...) in readout_len.py. The request's own presence_penalty must
    be 0 (validate_params enforces it), so vLLM's built-in penalty never adds a second subtraction.

    Generated tokens are tracked on device without a host sync: each step, the row's input token (the token sampled
    last step) is OR-ed into the slot's mask when its position is at or past the prompt length (an output token);
    on (re)admission the mask is rebuilt from the prior plus any partial output already in the slot (resumption)."""
    import numpy as np
    import torch
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

    class PriorPresence(LogitsProcessor):
        ENGAGED = {"apply_calls": 0, "row_steps": 0, "requests": 0}  # engagement receipt, read by the caller

        def __init__(self, vllm_config, req_states):
            self.rs = req_states
            n, V, dev = req_states.max_num_reqs, req_states.vocab_size, req_states.device
            self.mask = torch.zeros(n, V, dtype=torch.bool, device=dev)
            self.p = torch.zeros(n, dtype=torch.float32, device=dev)
            self.p_np = np.zeros(n, dtype=np.float32)
            self.pending: dict = {}
            self.dirty = False

        @classmethod
        def validate_params(cls, sp):
            ea = sp.extra_args or {}
            if ea.get("wg_prior") and float(ea.get("wg_prior_p", 0.0)) and sp.presence_penalty:
                raise ValueError("wg_prior carries the whole presence penalty: set the request's presence_penalty to 0")

        def add_request(self, req_idx, sampling_params):
            ea = sampling_params.extra_args or {}
            pr, p = ea.get("wg_prior"), float(ea.get("wg_prior_p", 0.0))
            if not pr or not p:
                if self.p_np[req_idx]:
                    self.p_np[req_idx] = 0.0
                    self.dirty = True
                self.pending.pop(req_idx, None)
                return False
            PriorPresence.ENGAGED["requests"] += 1
            self.p_np[req_idx] = p
            self.pending[req_idx] = sorted(set(int(t) for t in pr))
            self.dirty = True
            return True

        def apply_staged_writes(self):
            if self.pending:
                for slot, prior in self.pending.items():
                    self.mask[slot].zero_()
                    self.mask[slot, torch.tensor(prior, dtype=torch.long, device=self.mask.device)] = True
                    pl, fl = int(self.rs.prompt_len.np[slot]), int(self.rs.prefill_len.np[slot])
                    if fl > pl:  # resumed after preemption: outputs so far are already in the slot
                        self.mask[slot, self.rs.all_token_ids.gpu[slot, pl:fl].long()] = True
                self.pending.clear()
            if self.dirty:
                self.p.copy_(torch.from_numpy(self.p_np), non_blocking=False)
                self.dirty = False

        def apply(self, logits, ctx):
            act = self.p_np[ctx.idx_mapping_np] > 0
            if not act.any():
                return logits
            PriorPresence.ENGAGED["apply_calls"] += 1
            PriorPresence.ENGAGED["row_steps"] += int(act.sum())
            slots = ctx.expanded_idx_mapping.long()
            toks = ctx.input_ids.long()
            out_tok = ctx.pos.long() >= self.rs.prompt_len.gpu[slots].long()
            self.mask[slots, toks] = self.mask[slots, toks] | out_tok
            V = self.mask.shape[1]
            logits[:, :V] -= self.p[slots].unsqueeze(1) * self.mask[slots].to(logits.dtype)
            return logits

    return PriorPresence


class VEngine:
    def __init__(self, llm, th, sampling):
        self.llm, self.th, self.S = llm, th, sampling

    def generate(self, contexts, specs):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        S = self.S
        sps = []
        for sp in specs:
            # phase 2: PriorPresence carries the whole presence penalty (prior + generated), vLLM's own is off
            ea = {"wg_prior": sp["prior"], "wg_prior_p": S["presence_penalty"]} if sp["prior"] else None
            sps.append(SamplingParams(temperature=S["temperature"], top_p=S["top_p"], top_k=S["top_k"],
                                      presence_penalty=0.0 if ea else S["presence_penalty"], max_tokens=sp["max_tokens"],
                                      stop_token_ids=list(self.th.stops), seed=sp["seed"], skip_special_tokens=False,
                                      extra_args=ea))
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=c) for c in contexts], sps, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git_describe() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty", "--tags"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def grader_preflight(R) -> str:
    """Fail closed unless rl_common.correct grades through math_verify (same cases as preflight_grader.py)."""
    import math_verify  # noqa: F401  (ImportError = refuse)

    accept = [("\\boxed{\\$18}", "18"), ("\\boxed{18.00}", "18"), ("\\boxed{1,000}", "1000"), ("\\boxed{18}", "18"),
              ("\\boxed{\\dfrac{14}{3}}", "\\frac{14}{3}")]
    reject = [("\\boxed{19}", "18"), ("18", "18"), ("\\boxed{}", "18"), ("\\boxed{18} then \\boxed{19}", "18")]
    bad = [x for x in accept if not R.correct(*x)] + [x for x in reject if R.correct(*x)]
    if bad:
        raise SystemExit(f"grader preflight FAIL {bad}")
    from importlib.metadata import version

    return version("math_verify")


PARITY_K_OVERRIDE = {"wide_s2_cap512_r_gb10/ckpt_step150.pt": 2.5}
# effect_visible 5.0, except the (checkpoint, data file) pairs below: owner approval 2026-10-12 (relayed by the lane
# wg coordinator, "approve"): 4.0 for fresh-set readouts (test_fresh.jsonl) of Qwen3.5-4B checkpoints whose
# pre-readout self-test fails on the visibility checks alone (4B ctrl s1: lora effect 0.0405 vs bar 0.0419; 4B wide s2:
# lora 0.0433 vs 0.0436, gate 0.0398 vs 0.0436; every agreement, red-arm and start check passing). Every other check
# keeps its bar; the self-test runs before generation, so scored generations and scoring are unchanged.
EFFECT_VISIBLE_OVERRIDE = {("ml7_4b/grpo/ctrl_s1_cap512_r_box/ckpt_step150.pt", "test_fresh.jsonl"): 4.0,
                           ("ml7_4b/grpo/wide_s2_cap512_r_box/ckpt_step150.pt", "test_fresh.jsonl"): 4.0}


def effect_visible_of(ckpt: str, data: str) -> float:
    """The self-test's visibility bar (x the prefill floor) for this checkpoint and data file."""
    return next((v for (c, d), v in EFFECT_VISIBLE_OVERRIDE.items() if ckpt.endswith(c) and data.endswith(d)), 5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=("stock", "wide", "wide1", "ctrl"), default="stock")
    ap.add_argument("--ckpt", default="none")
    ap.add_argument("--strip", action="store_true")
    ap.add_argument("--caps", default=CAPS)
    ap.add_argument("--answer_max", type=int, default=512)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--gpu_mem", type=float, default=0.35)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank of --ckpt")
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--red", choices=("", "half", "noclose"), default="")
    ap.add_argument("--probe_effect", type=float, default=0.2,
                    help="probe size (nats per token) of the pre-readout parity self-test; 0.2 = the 2B record, "
                         "0.05 = the Qwen3.5-4B record (owner approval 2026-10-11 ~14:55)")
    ap.add_argument("--gate_stats", action="store_true")
    ap.add_argument("--no_traces", action="store_true")
    a = ap.parse_args()
    if a.arm in ("wide", "wide1") and a.ckpt == "none":
        raise SystemExit("a gated arm needs --ckpt (the untouched model is --arm stock)")
    if a.arm == "ctrl" and a.ckpt == "none":
        raise SystemExit("--arm ctrl needs --ckpt (the untouched model is --arm stock)")
    if a.strip and a.arm not in ("wide", "wide1"):
        raise SystemExit("--strip applies to a gated checkpoint")
    engine_arm = "stock" if (a.arm in ("stock", "ctrl") or a.strip) else a.arm

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_GDN_DECODE_KERNEL", "triton")
    import train_grpo_vllm as T  # puts WG_ML7_PYLIB (fla) on sys.path before transformers is imported
    import gate_vllm as NV

    # sets the per-arm VLLM_CACHE_ROOT before vllm is imported; a checkpoint gets the unmerged LoRA
    NV.install(engine_arm, lora_rank=a.rank if a.ckpt != "none" else 0)
    import rl_common as R
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM

    mv = grader_preflight(R)
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    th = R.Thinker(tok)
    items = R.load_jsonl(a.data)[:: a.stride][: a.limit or None]
    for it in items:
        it["gold"] = R.gold_of(it)
    caps = sorted(int(c) for c in a.caps.split(","))
    max_prompt = max(len(th.prompt(it["problem"])) for it in items)
    PP = make_prior_presence()
    hf = None
    if a.ckpt != "none":  # the training forward for the parity check, built BEFORE vLLM (train_grpo_vllm.run notes why)
        import arms
        import packing

        hf_arm = "ctrl" if engine_arm == "stock" else a.arm
        torch.manual_seed(0)
        hf, _ = arms.load_model(a.model, hf_arm, device_map={"": 0}, dtype=torch.bfloat16)
        hf = arms.add_lora(hf, a.rank).eval()
        packing.install(hf)
        packing.CONV_BACKEND = "torch"
    llm = LLM(model=a.model, **NV.engine_kwargs(engine_arm != "stock", gpu_memory_utilization=a.gpu_mem,
                                                 max_model_len=max_prompt + max(caps) + a.answer_max + len(th.force) + 8,
                                                 max_num_seqs=a.max_num_seqs, max_num_batched_tokens=8192,
                                                 logits_processors=[PP]))
    t_engine = time.time() - t_start
    rec_ck = None
    if a.ckpt != "none":
        import types as _types

        import packing

        state = torch.load(a.ckpt, map_location="cpu")
        canon = lambda k: k[k.find("layers."):] if "layers." in k else k  # noqa: E731
        by = {canon(k): v for k, v in state.items()}
        n_gate = sum("gate_w" in k for k in state)
        if engine_arm != "stock" and not n_gate:
            raise SystemExit("gated arm: the checkpoint has no gate tensors")
        current = {}
        for n, p in T.trainable(hf):
            if canon(n) not in by:
                raise SystemExit(f"trainable tensor {n} absent from {a.ckpt}")
            current[n] = by[canon(n)].float().clone()
        dropped = sum(1 for k in by if "gate_w" in k and not any(canon(n) == k for n in current))
        if a.strip and dropped != n_gate:
            raise SystemExit("strip: the gate tensors were not all dropped")
        eng = T.VEngine(None, a.model, "ctrl" if engine_arm == "stock" else a.arm, None, th.stops, llm=llm)
        # parity_k 2.0, except the checkpoints listed in PARITY_K_OVERRIDE (owner approval 2026-10-11 ~10:50: step 14
        # GSM8K wide s2 failed the decode-path check at 2.03x vs 2.0x with every other check passing; 2.5 as in the
        # wide s1 training retry). The check is a pre-readout self-test; it does not touch the scored generations.
        pk = next((v for k, v in PARITY_K_OVERRIDE.items() if a.ckpt.endswith(k)), 2.0)
        # probe_effect: --probe_effect (default 0.2, the value hard-coded before; owner approval 2026-10-11 ~14:55:
        # 0.05 for every readout of the Qwen3.5-4B record, where the 0.2 probe calibrator does not converge)
        pa = _types.SimpleNamespace(seed=a.seed, parity_samples=2, parity_tokens=256, micro_tokens=8192, pack=True,
                                    parity_k=pk, parity_floor=1e-3, probe_effect=a.probe_effect, effect_corr=0.9,
                                    effect_slope_tol=0.1, effect_visible=effect_visible_of(a.ckpt, a.data))
        prows = items[:8]
        rec_ck = T.parity_check(eng, hf, "ctrl" if engine_arm == "stock" else a.arm, prows, th, pa,
                                torch.device("cuda:0"), current)
        rec_ck.update(train_kernel="fla" if packing.fla_active() else "torch_reference", gate_tensors=n_gate,
                      gate_dropped=dropped, sync=eng.sync(current))
        del hf
        torch.cuda.empty_cache()
        if not (rec_ck["pass"] and rec_ck["start_pass"]):
            Path(a.out + ".parity_fail.json").write_text(json.dumps(rec_ck, indent=1, default=str))
            raise SystemExit(f"engine/trainer parity FAILED on {a.ckpt}: {rec_ck['checks']} start {rec_ck['start']}")
    if a.gate_stats:
        NV.STATS["on"] = True
    rows, traces, timing = run_readout(VEngine(llm, th, R.VENDOR), items, th, caps, a.answer_max, a.samples, a.seed,
                                       R.correct, lambda t: tok.decode(t, skip_special_tokens=True), a.red)
    summ = summarize(rows, caps)
    out = {
        "schema": SCHEMA, "method": "prefix (readout_len.py): one vendor-sampled trace per (item, sample), exact "
                                    "truncation at each budget, forced close, sampled answer",
        "args": vars(a), "argv": sys.argv, "arm": a.arm, "engine": engine_arm, "ckpt": a.ckpt, "strip": a.strip,
        "red": a.red, "caps": caps, "sampling": R.VENDOR, "force_text": tok.decode(th.force), "stops": th.stops,
        "set_checkpoint": None, "parity": rec_ck, "gate_calls": dict(NV.CALLS), "prior_presence": dict(PP.ENGAGED), "gate_stats": dict(NV.STATS) if a.gate_stats else None,
        "summary": summ,
        "env": {"vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                "math_verify": mv, "python": platform.python_version(), "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0), "git_describe": _git_describe(),
                "model_revision": (Path(a.model) / "REVISION").read_text().strip() if (Path(a.model) / "REVISION").exists() else None,
                "sha256": {"readout_vllm.py": _sha(__file__), "rl_common.py": _sha(R.__file__),
                           "gate_vllm.py": _sha(NV.__file__), "data": _sha(a.data),
                           "ckpt": _sha(a.ckpt) if a.ckpt != "none" else None}},
        "timing": {**timing, "engine_s": t_engine, "wall_s": time.time() - t_start},
        "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out + ".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, a.out)
    if not a.no_traces:
        with gzip.open(a.out + ".traces.jsonl.gz", "wt") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
    brief = {c: round(v["acc"], 4) for c, v in summ["per_cap"].items()}
    print(json.dumps({"acc": brief, "norm_auc_log2": round(summ["norm_auc_log2"], 4), "natural": summ["natural"],
                      "timing": out["timing"]}), flush=True)


if __name__ == "__main__":
    main()
