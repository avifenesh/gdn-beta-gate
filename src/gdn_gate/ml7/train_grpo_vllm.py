#!/usr/bin/env python3
"""Step 7 full-run trainer: length-cost GRPO on Qwen3.5-2B (thinking), rollouts on a colocated vLLM engine, training
forward in PyTorch, one arm and seed per run. Recipe: the approved model design in ml_pipeline/PIPELINE.md.

Design: option (b). The loop and loss are broad/train_grpo.py's (receipted: broad/test_rl_cpu.py R5), with HF generate
replaced by a vLLM 0.31 engine in the same process carrying the gate plugin (vllm/gate_vllm.py) installed with
lora_rank: the engine keeps the LoRA unmerged (scale * B(A x) added to the GDN projection outputs, as PEFT does in the
trainer) and the gate W, both as static fp32 buffers. After every optimizer step gate_vllm.set_lora copies the
trainable tensors into them in place (no adapter reload, CUDA graphs stay valid). The merged path
(set_checkpoint: W + scale B A rounded to bf16) is not used: on the first GRPO steps it kept 1.3-4.0% of the LoRA
move (workstation smoke 2026-10-08), so rollouts would come from a policy lagging the trained one.
Why not TRL 1.14.2 GRPOTrainer (option a), checked against its source in the workstation vLLM venv:
  * its colocate engine is built inside trl.generation.VLLMGeneration with TRL's own LLM kwargs. The gate needs the
    plugin installed and VLLM_CACHE_ROOT set before vLLM is imported, the Triton GDN prefill backend, an fp32 Mamba
    state cache and VLLM_ENABLE_V1_MULTIPROCESSING=0, so the engine would be replaced;
  * its weight sync (VLLMGeneration.sync_weights) merges the PEFT adapter and pushes tensors by name through vLLM's
    load_weights, which has no mapping for the gate tensors (gate_w) and does not know the fused in_proj_qkvz /
    in_proj_ba layout, so the sync would be replaced by set_lora;
  * the contract's two-phase budget forcing (forced close excluded from the loss, then a sampled answer) needs the
    experimental rollout_func plus an env_mask, so the rollouts would be replaced;
  * our loss (Dr. GRPO advantage, token sum over the batch's sampled tokens, ratio 1, no KL, no importance-sampling
    correction) maps to loss_type="dapo", scale_rewards="none", vllm_importance_sampling_correction=False and
    num_iterations=1, a mapping that would need its own proof while ours is already tested.
  With engine, sync and rollouts replaced, TRL would only add its optimizer loop. Speed lives in the rollouts.
Unsloth for the training forward: not used and not measured in-process. Unsloth 2026.10.2 lives in venv-fast
(torch 2.14, datasets < 5) and vLLM 0.31 is built against torch 2.13 (workstation vLLM venv: datasets 5.1); one process
cannot hold both without a rebuild. The training forward is our packed path (broad/packing.py, the step 6 packing,
x1.65 over padding there) on fla 0.5.2 kernels. train_log.jsonl splits every step into gen_s, train_s and sync_s,
so the measured training share decides whether a two-process Unsloth trainer is worth building.

Per step (identical for every arm): `--prompts` rows of the frozen RL pool (rl_pool.jsonl, sha256 checked against
ml_pipeline/split_manifest.json, refused on mismatch; test files are refused by name) in a seed-fixed permutation,
`--group` samples each at temperature 1.0 and nothing else (TRAIN_SAMPLING), thinking capped at `--cap` (required)
with rl_common budget forcing: still thinking at the cap -> "\\n</think>\\n\\n" appended (forced, out of the loss),
answer sampled up to `--answer_max`. Grading: rl_common.correct through math_verify only (the run refuses to start
otherwise). Reward: rl_common.length_reward (alpha `--alpha`) over the lengths `--length_on` names (required, no
default): `completion` = every completion token (sampled thinking, the close forced or natural, the answer), or
`reasoning` = the sampled thinking tokens before the close, forced or natural (Rollout.reasoning_tokens, set in
assemble(); the same count as readout_vllm.split_trace's think_len), so the answer's length is never penalized.
Advantage: reward minus the group mean (Dr. GRPO, no std scaling). One gradient step per batch, so the policy ratio
is 1 and the loss is -sum_i sum_t A_i log p(y_t) / (sampled tokens in the batch), over sampled tokens only. No KL. LoRA r16 (alpha 32) on
the GDN projections, plus the gate W for wide (beta in (0,2)) and wide1 (beta in (0,1)) (broad/arms.py); AdamW
(0.9, 0.95), wd 0, lr 2e-5 for both groups, constant after 5% warmup, grad clip 1.0.

Init (`--init_from CKPT`, owner design 2026-10-09 05:20: every arm branches from the shared format warm-up): the
arm's LoRA A/B start from CKPT, a ckpt_step{N}.pt of this trainer (state_of names). Its key set must be exactly this
model's LoRA tensors (no gate tensors) with the same shapes, else the run refuses before the engine is built. The gate
W (wide, wide1) keeps its usual init (zeros); AdamW state, step counter, schedule and RNG start fresh. The values are
copied into the PEFT parameters themselves, so every ckpt_step{N}.pt this run writes holds the full LoRA (init plus
update) and readout_vllm.py --ckpt reads it unchanged. meta.json records the path, its sha256 and the optimizer state
at the first step. --resume ignores --init_from once the run has its own checkpoints (meta_resume{N}.json says so).

Engine/trainer parity (every process start, fail closed; the same check runs in readout_vllm.py on a loaded
checkpoint). Token sequences are sampled by the engine; the trainer scores the same tokens.
  floors      untouched model on both sides: mean |log p_train - log p_engine| on the decode path (the sampled
              tokens' own logprobs, what the importance ratio uses) and on the prefill path (prompt logprobs).
  probe       a LoRA (random B direction) and, for gated arms, a gate W (random direction), each scaled on the trainer
              until it moves log p by about --probe_effect nats per token (visible above the floor).
  pairs       decode pair at the probe state <= K x decode floor; prefill pairs at the LoRA and full probe states
              <= K x prefill floor (K = --parity_k = 2).
  effects     the engine's log p shift from the LoRA (and from the gate) against the trainer's own shift on the same
              tokens: slope within 1 +- --effect_slope_tol (0.1), correlation >= --effect_corr (0.9), effect size
              >= --effect_visible (5) x the prefill floor. A lagging or missing LoRA shows as a slope near 0.
  red arms    the engine synced with A and B zeroed (set_lora(lora=False)), and with the gate not synced
              (set_lora(gate=False)): their effect agreement must FAIL, so the check can see a missing component.
  start       the run's real state is written back; its decode pair must stay under the bound, and its own effect
              (state vs untouched) must agree whenever it is visible (a resumed run, a readout checkpoint).
K = 2: FINDINGS 2026-10-08 measured the gated bf16 path at about 1.8-2x the stripped path's engine-vs-HF gap (0.070-0.098
vs 0.039-0.050 on the 4B). Every step logs the importance-ratio statistics of the trained rollouts
(log p_train - log p_engine on every sampled token) and the sync time.

GDN kernels: every arm runs the shipped bf16 path on both sides (the Olmo 2x2 of 2026-10-08 measured fp32 vs bf16
fla at about 0.013 nats on HF and vLLM alike, so no precision option is kept). On GPU the training forward refuses
to run without fla (the torch reference path is far slower); provide fla through WG_ML7_PYLIB (a --target dir
prepended to sys.path for this process only).

Memory: one process holds the engine (gpu_memory_utilization `--gpu_mem` of the device, created after the training
model, max_num_seqs
`--max_num_seqs`, 128 = one step of 16 x 8 rollouts; GDN needs one Mamba cache block per sequence and 128 is the
known-good count) and the training model (packed micro-rows of at most `--micro_tokens` tokens). A startup memory
probe runs one forward+backward of a full micro-row of cap-length sequences and records the process peak.

Writes to --out: meta.json (meta_resume{N}.json on resume), train_log.jsonl (every step: reasoning_mean and
answer_mean, the sampled answer tokens after the reasoning, logged separately, and length_mean, the reward's length),
rollouts.jsonl (every rollout: correctness, reasoning, answer and completion tokens, reward, advantage; two texts
per step), val_log.jsonl (only with --val_every: report
only, never selects anything), ckpt_step{N}.pt (trainable tensors, train_grpo.py names, read by readout_vllm.py) and
opt_step{N}.pt every --ckpt_every steps and at the last step, DONE at the last step. --resume restarts from the newest
step with both files, drops log lines from that step on, and reruns the parity check on the restored state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEXT = HERE.parent
for _p in (NEXT / "vllm", NEXT / "broad", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if os.environ.get("WG_ML7_PYLIB"):  # fla (and peft on the Sparks) for this process only; before transformers
    sys.path.insert(0, os.environ["WG_ML7_PYLIB"])
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_GDN_DECODE_KERNEL", "triton")

MANIFEST = Path(__file__).resolve().parents[3] / "ml_pipeline" / "split_manifest.json"
TRAIN_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0}
TEST_FILES = ("test_gsm8k.jsonl", "test_math500.jsonl")


# ------------------------------------------------------------------------------------------------ data and grading
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_frozen(path, name: str, manifest) -> dict:
    """Refuse unless `path` is the frozen split file `name`: same basename, sha256 and row count as
    ml_pipeline/split_manifest.json. A test split is refused by name before it is opened."""
    base = Path(path).name
    if base in TEST_FILES or base.startswith("test_"):
        raise SystemExit(f"refusing to read a test split in the trainer: {path}")
    if base != name:
        raise SystemExit(f"expected the frozen {name}, got {path}")
    man = json.loads(Path(manifest).read_text())
    want, got = man["files"][name], sha256_file(path)
    if got != want:
        raise SystemExit(f"{name} sha256 {got} does not match split_manifest.json ({want}): refusing")
    with open(path) as f:
        n = sum(1 for _ in f)
    if n != man["counts"][name]:
        raise SystemExit(f"{name} has {n} rows, split_manifest.json says {man['counts'][name]}")
    return {"file": str(path), "sha256": got, "rows": n, "manifest": str(manifest)}


def load_rows(path):
    import rl_common as R

    rows = R.load_jsonl(path)
    for r in rows:
        r["gold"] = R.gold_of(r)
    return rows


def require_grader():
    """Fail closed unless rl_common.correct grades through math_verify (readout_vllm.grader_preflight cases)."""
    import readout_vllm as RV
    import rl_common as R

    return RV.grader_preflight(R)


def seed_of(base: int, *parts) -> int:
    import readout_vllm as RV

    return RV.seed_of(base, *parts)


# ------------------------------------------------------------------------------------------------ rollouts (pure)
_ROLL = {}


def _rollout_cls():
    if "cls" not in _ROLL:
        import rl_common as R

        @dataclass
        class Roll(R.Rollout):
            lp: list = field(default_factory=list)  # engine log p per completion token, None for forced tokens

        _ROLL["cls"] = Roll
    return _ROLL["cls"]


def assemble(prompts, g1, lp1, th):
    """Phase 1 bookkeeping, rl_common.Thinker.generate line for line, carrying the engine logprobs.
    Returns (rolls, need): need = rollouts that get a phase-2 answer."""
    Roll = _rollout_cls()
    rolls, need = [], []
    for p, g, lp in zip(prompts, g1, lp1):
        g, lp = list(g), list(lp)
        assert len(g) == len(lp)
        if th.end_think in g:
            i = g.index(th.end_think)
            r = Roll(p, g[:i], g[i:], [True] * (len(g) - i), lp=lp)
            if g[-1] not in th.stops:
                need.append(len(rolls))  # thinking closed but the answer ran into the cap: finish it
        else:
            if g and g[-1] in th.stops:
                g, lp = g[:-1], lp[:-1]  # silent stop inside the thinking: drop the stop, force the close
            r = Roll(p, g, list(th.force), [False] * len(th.force), forced=True, lp=lp + [None] * len(th.force))
            need.append(len(rolls))
        r.reasoning_tokens = len(r.think)
        rolls.append(r)
    return rolls, need


def finish(rolls, need, g2, lp2, decode):
    for i, g, lp in zip(need, g2, lp2):
        assert len(g) == len(lp)
        rolls[i].tail += list(g)
        rolls[i].tail_sampled += [True] * len(g)
        rolls[i].lp += list(lp)
    for r in rolls:
        r.text = decode(r.tail)
        assert len(r.lp) == len(r.completion) == len(r.sampled_mask)
        assert all((x is None) != m for x, m in zip(r.lp, r.sampled_mask)), "engine logprob missing on a sampled token"
    return rolls


def sampled_lp(r) -> list:
    return [x for x, m in zip(r.lp, r.sampled_mask) if m]


def group_rewards(corr, lens, group: int, alpha: float):
    """rl_common.length_reward per prompt group, advantage = reward - group mean (Dr. GRPO, no std scaling)."""
    import rl_common as R

    assert len(corr) == len(lens) and len(corr) % group == 0
    rew, adv = [], []
    for gi in range(len(corr) // group):
        sl = slice(gi * group, (gi + 1) * group)
        rg = R.length_reward(corr[sl], lens[sl], alpha)
        mu = sum(rg) / len(rg)
        rew += rg
        adv += [x - mu for x in rg]
    return rew, adv


def answer_tokens(r) -> int:
    """Sampled tokens after the reasoning: a natural close's </think> plus the answer, or the answer after a forced
    close (forced tokens excluded), readout_vllm's answer-token count."""
    return sum(r.tail_sampled)


def reward_lengths(rolls, length_on: str) -> list:
    """The lengths rl_common.length_reward sees (--length_on)."""
    if length_on == "reasoning":
        return [r.reasoning_tokens for r in rolls]
    if length_on == "completion":
        return [len(r.completion) for r in rolls]
    raise ValueError(f"length_on {length_on!r}")


def ratio_stats(per_seq: list, prefix: str = "ratio_") -> dict:
    """per_seq: per rollout, log p_train - log p_engine on each sampled token."""
    import numpy as np

    flat = np.array([x for s in per_seq for x in s], dtype=np.float64)
    if flat.size == 0:
        return {prefix + "n": 0}
    seq = np.array([sum(s) for s in per_seq if s], dtype=np.float64)
    ex = np.exp(np.clip(flat, -50, 50))
    return {prefix + "n": int(flat.size), prefix + "seqs": int(seq.size), prefix + "logr_mean": float(flat.mean()),
            prefix + "logr_std": float(flat.std()), prefix + "logr_abs_mean": float(np.abs(flat).mean()),
            prefix + "logr_abs_p99": float(np.quantile(np.abs(flat), 0.99)),
            prefix + "logr_abs_max": float(np.abs(flat).max()), prefix + "r_mean": float(ex.mean()),
            prefix + "r_min": float(ex.min()), prefix + "r_max": float(ex.max()),
            prefix + "frac_abs_gt_0p2": float((np.abs(flat) > 0.2).mean()),
            prefix + "seq_logr_abs_mean": float(np.abs(seq).mean())}


def diff_stats(lp_a: list, lp_b: list) -> dict:
    import numpy as np

    d = np.array([x - y for a, b in zip(lp_a, lp_b) for x, y in zip(a, b)], dtype=np.float64)
    if d.size == 0:
        return {"n": 0, "mean": float("nan"), "p99": float("nan"), "max": float("nan"), "bias": float("nan")}
    ad = np.abs(d)
    return {"n": int(d.size), "mean": float(ad.mean()), "p99": float(np.quantile(ad, 0.99)), "max": float(ad.max()),
            "bias": float(d.mean())}


# ------------------------------------------------------------------------------------------------ training forward
def train_core(model):
    import arms

    return arms.core(model)


def forward_logps(model, items, dev, micro_tokens: int, pack: bool, norm: float | None = None, chunk: int = 256):
    """Training-forward log p of each item's sampled completion tokens; with `norm`, also the backward of
        sum_i sum_t a_i * (-log p(y_t)) / norm
    over those tokens (forced tokens are context only). items: (ids, plen, mask, a) with mask over ids[plen:].
    pack=True: packed rows (broad/packing.py: positions, attention, GDN state and conv reset per item), pack=False:
    right-padded rows as broad/train_grpo.policy_loss. The vocabulary projection and CE run in `chunk`-token pieces
    (a 1,024-token piece holds 1 GiB of fp32 logits for the 248k vocabulary: the workstation smoke OOMed on it).
    Returns (loss_sum, [per-item log p lists])."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    import packing

    grad = norm is not None
    dec, head = train_core(model)
    order = sorted(range(len(items)), key=lambda k: -len(items[k][0]))
    rows = []
    if pack:
        exs = [types.SimpleNamespace(ids=items[k][0], labels=items[k][0], k=k) for k in order]
        rows = [[e.k for e in g] for g in packing.groups(exs, micro_tokens)] if exs else []
    else:
        i = 0
        while i < len(order):
            n = max(1, micro_tokens // len(items[order[i]][0]))
            rows.append(order[i : i + n])
            i += n
    out = [[] for _ in items]
    total = 0.0
    use_ac = dev.type == "cuda"
    for row in rows:
        pos, tgt, wts, own = [], [], [], []
        if pack:
            exs = [types.SimpleNamespace(ids=items[k][0], labels=items[k][0]) for k in row]
            b = packing.batch(exs, dev, conv_k=packing._conv_k(model))
            off = 0
            for k in row:
                ids, plen, mask, a = items[k]
                for j, m in enumerate(mask):
                    if m:
                        pos.append(off + plen + j - 1)  # position p predicts token p + 1
                        tgt.append(ids[plen + j])
                        wts.append(a)
                        own.append(k)
                off += len(ids)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_ac):
                h = packing.hidden(model, b)
            hs = h[torch.tensor(pos, device=dev, dtype=torch.long)]
        else:
            L = len(items[row[0]][0])
            ids_t = torch.zeros((len(row), L), dtype=torch.long)
            att = torch.zeros((len(row), L), dtype=torch.long)
            for r_i, k in enumerate(row):
                ids, plen, mask, a = items[k]
                ids_t[r_i, : len(ids)] = torch.tensor(ids)
                att[r_i, : len(ids)] = 1
                for j, m in enumerate(mask):
                    if m:
                        pos.append(r_i * L + plen + j - 1)
                        tgt.append(ids[plen + j])
                        wts.append(a)
                        own.append(k)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_ac):
                h = dec(input_ids=ids_t.to(dev), attention_mask=att.to(dev)).last_hidden_state
            hs = h.reshape(-1, h.shape[-1])[torch.tensor(pos, device=dev, dtype=torch.long)]
        ys = torch.tensor(tgt, device=dev, dtype=torch.long)
        ws = torch.tensor(wts, device=dev, dtype=torch.float32)

        def head_ce(x, t):
            return F.cross_entropy(head(x).float(), t, reduction="none")

        loss = 0.0
        lps = []
        for s in range(0, hs.shape[0], chunk):
            if grad:
                ce = checkpoint(head_ce, hs[s : s + chunk], ys[s : s + chunk], use_reentrant=False)
            else:
                ce = head_ce(hs[s : s + chunk], ys[s : s + chunk])
            loss = loss + (ce * ws[s : s + chunk]).sum()
            lps.append(-ce.detach().float().cpu())
        if grad and torch.is_tensor(loss) and loss.requires_grad:
            (loss / norm).backward()
        total += float(loss)
        for k, x in zip(own, torch.cat(lps).tolist() if lps else []):
            out[k].append(x)
    return total, out


def trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def state_of(model) -> dict:
    return {n: p.detach().float().cpu().clone() for n, p in trainable(model)}


def load_state(model, state: dict) -> None:
    import torch

    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(device=params[k].device, dtype=params[k].dtype))


def init_lora_from(model, path) -> dict:
    """--init_from: copy a warm-up run's LoRA A/B (its ckpt_step{N}.pt) into `model`. Refused (SystemExit) unless the
    checkpoint's key set is exactly the model's LoRA tensors (gate tensors excluded: the warm-up has none) with the same
    shapes and finite values. Gate tensors, optimizer, schedule and RNG are not touched."""
    import torch

    st = torch.load(path, map_location="cpu")
    if not isinstance(st, dict) or not all(torch.is_tensor(v) for v in st.values()):
        raise SystemExit(f"--init_from {path}: not a ckpt_step*.pt state dict")
    lora = {n: p for n, p in trainable(model) if "gate_w" not in n}
    missing, extra = sorted(set(lora) - set(st)), sorted(set(st) - set(lora))
    if missing or extra:
        raise SystemExit(f"--init_from {path}: LoRA key set does not match this model ({len(missing)} missing, "
                         f"{len(extra)} extra; e.g. missing {missing[:2]}, extra {extra[:2]}): refusing")
    bad = [(k, tuple(st[k].shape), tuple(p.shape)) for k, p in lora.items() if tuple(st[k].shape) != tuple(p.shape)]
    if bad:
        raise SystemExit(f"--init_from {path}: {len(bad)} LoRA shapes differ (ckpt vs model, e.g. {bad[:2]}): refusing")
    nonfinite = [k for k in lora if not bool(torch.isfinite(st[k]).all())]
    if nonfinite:
        raise SystemExit(f"--init_from {path}: non-finite values in {nonfinite[:2]}: refusing")
    load_state(model, {k: st[k] for k in lora})
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "applied": True, "tensors": len(lora),
            "numel": int(sum(st[k].numel() for k in lora)),
            "norm": float(sum(float(st[k].float().norm()) ** 2 for k in lora) ** 0.5)}


# ------------------------------------------------------------------------------------------------ checkpoints
def latest_complete(out: Path) -> int:
    steps = []
    for p in out.glob("opt_step*.pt"):
        s = int(p.stem[len("opt_step"):])
        if (out / f"ckpt_step{s}.pt").exists():
            steps.append(s)
    return max(steps) if steps else 0


def atomic_save(obj, path: Path) -> None:
    import torch

    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def truncate_log(path: Path, start: int, key: str = "step") -> int:
    """Drop lines whose `key` >= start (a run killed between checkpoints logged steps it will redo)."""
    if not path.exists():
        return 0
    keep, dropped = [], 0
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:  # a line cut by the kill
            dropped += 1
            continue
        if rec.get(key, -1) >= start:
            dropped += 1
        else:
            keep.append(ln)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(x + "\n" for x in keep))
    os.replace(tmp, path)
    return dropped


# ------------------------------------------------------------------------------------------------ vLLM engine
class VEngine:
    """The colocated rollout engine: vLLM with the gate plugin installed with lora_rank (unmerged LoRA on every GDN
    layer; stock kernels for ctrl, the gate for wide / wide1), weights by gate_vllm.set_lora (an in-place copy).
    llm: wrap an engine built elsewhere (readout_vllm.py) instead of building one."""

    def __init__(self, a, model_dir: str, arm: str, max_model_len: int, stops, llm=None):
        import gate_vllm as NV

        self.NV, self.gated, self.stops = NV, arm != "ctrl", list(stops)
        self.kw, self.engine_s = None, 0.0
        if llm is None:
            from vllm import LLM

            self.kw = NV.engine_kwargs(self.gated, gpu_memory_utilization=a.gpu_mem, max_model_len=max_model_len,
                                       max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_batched_tokens,
                                       seed=a.seed)
            t0 = time.time()
            llm = LLM(model=model_dir, **self.kw)
            self.engine_s = time.time() - t0
        self.llm = llm
        self.model = NV.get_model(self.llm)
        self.layout = NV.check_layout(self.model, NV.BaseWeights(model_dir, sorted(NV.gdn_layers(self.model))))

    def info(self) -> dict:
        import torch

        d = {"engine_kwargs": dict(self.kw or {}), "engine_s": self.engine_s, "layout": self.layout,
             "installed": self.NV._STATE["installed"], "gate_calls": dict(self.NV.CALLS),
             "torch_allocated_after_engine_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
        try:
            cc = self.llm.llm_engine.vllm_config.cache_config
            d["num_gpu_blocks"] = cc.num_gpu_blocks
            d["block_size"] = cc.block_size
        except Exception as e:  # noqa: BLE001
            d["num_gpu_blocks"] = f"unavailable: {e}"
        return d

    def sync(self, state: dict | None, gate: bool | None = None, lora: bool = True) -> dict:
        t0 = time.time()
        rec = self.NV.set_lora(self.model, state, gate=self.gated if gate is None else gate, lora=lora)
        rec["sync_s"] = time.time() - t0
        return rec

    def sample(self, contexts, max_tokens: int, seeds):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        S = TRAIN_SAMPLING
        sps = [SamplingParams(temperature=S["temperature"], top_p=S["top_p"], top_k=S["top_k"],
                              presence_penalty=S["presence_penalty"], max_tokens=max_tokens,
                              stop_token_ids=self.stops, seed=s, logprobs=0, skip_special_tokens=False,
                              detokenize=False) for s in seeds]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=c) for c in contexts], sps, use_tqdm=False)
        g, lp = [], []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            g.append(ids)
            lp.append([d[t].logprob for d, t in zip(c.logprobs, ids)])
        return g, lp

    def score(self, items) -> list:
        """Prefill-path log p of each item's sampled completion tokens (items as forward_logps takes them)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        outs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids, _, _, _ in items],
                                 SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0), use_tqdm=False)
        res = []
        for (ids, plen, mask, _), o in zip(items, outs):
            pl = o.prompt_logprobs
            res.append([pl[plen + j][ids[plen + j]].logprob for j, m in enumerate(mask) if m])
        return res


def rollouts(eng, prompts, cap, answer_max, seeds1, seeds2, th, decode):
    g1, lp1 = eng.sample(prompts, cap, seeds1)
    rolls, need = assemble(prompts, g1, lp1, th)
    g2, lp2 = [], []
    if need:
        g2, lp2 = eng.sample([rolls[i].prompt + rolls[i].completion for i in need], answer_max, [seeds2[i] for i in need])
    finish(rolls, need, g2, lp2, decode)
    return rolls, sum(len(x) for x in g1) + sum(len(x) for x in g2)


# ------------------------------------------------------------------------------------------------ startup checks
def effect_agreement(tr_a, tr_b, en_a, en_b) -> dict:
    """The engine's log p shift between two states against the trainer's on the same tokens:
    tr = tr_b - tr_a, en = en_b - en_a per token; slope (least squares through 0), correlation, mean |en - tr|,
    effect size mean |tr|."""
    import numpy as np

    tr = np.array([y - x for a, b in zip(tr_a, tr_b) for x, y in zip(a, b)], dtype=np.float64)
    en = np.array([y - x for a, b in zip(en_a, en_b) for x, y in zip(a, b)], dtype=np.float64)
    den = float((tr ** 2).sum())
    corr = float(np.corrcoef(tr, en)[0, 1]) if tr.std() > 0 and en.std() > 0 else 0.0
    return {"n": int(tr.size), "slope": float((tr * en).sum() / den) if den > 0 else 0.0, "corr": corr,
            "err_mean": float(np.abs(en - tr).mean()), "effect_mean": float(np.abs(tr).mean()),
            "engine_effect_mean": float(np.abs(en).mean())}


def agrees(e: dict, a) -> bool:
    return bool(e["corr"] >= a.effect_corr and abs(e["slope"] - 1.0) <= a.effect_slope_tol)


def calibrate_probe(model, zero: dict, arm: str, items, t0, a, dev) -> tuple:
    """A probe trainable state whose effect on the trainer's log p of `items` is about --probe_effect nats per token
    (mean |delta log p|): LoRA A as in `zero`, LoRA B along a fixed random direction, scaled; then, for gated arms, a
    random gate W scaled the same way on top of the LoRA."""
    import torch

    import patch

    g = torch.Generator().manual_seed(a.seed + 7)
    st = {k: v.clone() for k, v in zero.items()}
    dirs = {k: torch.randn(v.shape, generator=g) for k, v in st.items() if "lora_B" in k or "gate_w" in k}
    info = {"target": a.probe_effect}

    def score(state):
        load_state(model, state)
        with torch.no_grad():
            return forward_logps(model, items, dev, a.micro_tokens, a.pack)[1]

    def eff(x, y):
        d = [abs(u - v) for p, q in zip(x, y) for u, v in zip(p, q)]
        return sum(d) / max(len(d), 1)

    for part, ref in (("lora_B", t0), ("gate_w", None)):
        keys = [k for k in dirs if part in k]
        if not keys:
            continue
        ref = ref if ref is not None else score(st)
        s, e = 1e-3, 0.0
        for _ in range(6):
            for k in keys:
                st[k] = dirs[k] * s
            e = eff(score(st), ref)
            if abs(e - a.probe_effect) < 0.15 * a.probe_effect or e == 0.0:
                break
            s *= min(10.0, max(0.1, a.probe_effect / e))
        info[part] = {"scale": s, "effect": e}
    if any("gate_w" in k for k in dirs):
        load_state(model, st)
        patch.enable_stats(model, True)
        with torch.no_grad():
            forward_logps(model, items[:2], dev, a.micro_tokens, a.pack)
        stt = patch.collect_stats(model)
        patch.enable_stats(model, False)
        info["abs_t_mean"] = sum(stt["abs_t_per_layer"]) / max(len(stt["abs_t_per_layer"]), 1)
        fr = stt["beta_gt1_frac_per_layer"]
        info["beta_gt1_frac_mean" if arm == "wide" else "beta_at_ceiling_frac_mean"] = sum(fr) / max(len(fr), 1)
    return st, info


def parity_check(eng, model, arm, rows, th, a, dev, current: dict) -> dict:
    """Engine vs trainer (module docstring). Leaves both sides at `current` and checks that state too."""
    import torch

    gated = arm != "ctrl"
    prompts = [th.prompt(r["problem"]) for r in rows for _ in range(a.parity_samples)]
    zero = {k: (torch.zeros_like(v) if ("lora_B" in k or "gate_w" in k) else v.clone()) for k, v in current.items()}

    def seeds(tag):
        return [seed_of(a.seed, "parity", tag, k) for k in range(len(prompts))]

    def items_of(g):
        return [(p + x, len(p), [True] * len(x), 0.0) for p, x in zip(prompts, g)]

    def score(items, state):
        load_state(model, state)
        with torch.no_grad():
            return forward_logps(model, items, dev, a.micro_tokens, a.pack)[1]

    t0 = time.time()
    res = {"k": a.parity_k, "floor_abs": a.parity_floor, "prompts": len(rows), "samples": a.parity_samples,
           "tokens": a.parity_tokens, "effect_corr_min": a.effect_corr, "effect_slope_tol": a.effect_slope_tol,
           "effect_visible_x": a.effect_visible}
    # floors: untouched model on both sides, decode path (sampled tokens) and prefill path
    eng.sync(zero, gate=gated)
    g0, lp0 = eng.sample(prompts, a.parity_tokens, seeds("base"))
    it0 = items_of(g0)
    t_base0 = score(it0, zero)
    res["floor_decode"] = diff_stats(t_base0, lp0)
    res["floor_prefill"] = diff_stats(t_base0, eng.score(it0))
    probe, res["probe"] = calibrate_probe(model, zero, arm, it0, t_base0, a, dev)
    nogate = {k: (torch.zeros_like(v) if "gate_w" in k else v) for k, v in probe.items()}
    # the probe state: sampled tokens and their decode logprobs
    eng.sync(probe, gate=gated)
    g1, lp1 = eng.sample(prompts, a.parity_tokens, seeds("probe"))
    it1 = items_of(g1)
    t_full = score(it1, probe)
    res["probe_decode"] = diff_stats(t_full, lp1)
    e_full = eng.score(it1)
    e_lora = None
    if gated:
        eng.sync(probe, gate=False)  # also the gate red arm: an engine the gate never reached
        e_lora = eng.score(it1)
    eng.sync(probe, gate=False, lora=False)  # red arm: A and B zeroed in the engine
    e_red = eng.score(it1)
    eng.sync(zero, gate=gated)
    e_base = eng.score(it1)
    t_base = score(it1, zero)
    t_lora = score(it1, nogate) if gated else t_full
    e_lora = e_lora if gated else e_full
    res["prefill_floor_probe_tokens"] = diff_stats(t_base, e_base)
    res["prefill_lora"] = diff_stats(t_lora, e_lora)
    res["lora_effect"] = effect_agreement(t_base, t_lora, e_base, e_lora)
    res["red_lora_zeroed"] = effect_agreement(t_base, t_lora, e_base, e_red)
    if gated:
        res["prefill_full"] = diff_stats(t_full, e_full)
        res["gate_effect"] = effect_agreement(t_lora, t_full, e_lora, e_full)
        res["red_gate_not_synced"] = effect_agreement(t_lora, t_full, e_lora, e_lora)
    bound_dec = a.parity_k * max(res["floor_decode"]["mean"], a.parity_floor)
    bound_pre = a.parity_k * max(res["floor_prefill"]["mean"], res["prefill_floor_probe_tokens"]["mean"], a.parity_floor)
    visible = a.effect_visible * max(res["floor_prefill"]["mean"], a.parity_floor)
    checks = {"probe_decode": res["probe_decode"]["mean"] <= bound_dec,
              "prefill_lora": res["prefill_lora"]["mean"] <= bound_pre,
              "lora_visible": res["lora_effect"]["effect_mean"] >= visible,
              "lora_effect_agrees": agrees(res["lora_effect"], a),
              "red_lora_fails": not agrees(res["red_lora_zeroed"], a)}
    if gated:
        checks.update({"prefill_full": res["prefill_full"]["mean"] <= bound_pre,
                       "gate_visible": res["gate_effect"]["effect_mean"] >= visible,
                       "gate_effect_agrees": agrees(res["gate_effect"], a),
                       "red_gate_fails": not agrees(res["red_gate_not_synced"], a)})
    res.update(bound_decode=bound_dec, bound_prefill=bound_pre, visible_min=visible, checks=checks,
               **{"pass": all(checks.values())})
    # back to the run's real state; its decode pair, and its own effect when it is large enough to see
    eng.sync(current, gate=gated)
    gc, lpc = eng.sample(prompts, a.parity_tokens, seeds("start"))
    itc = items_of(gc)
    t_cur = score(itc, current)
    res["start"] = diff_stats(t_cur, lpc)
    e_cur = eng.score(itc)
    eng.sync(zero, gate=gated)
    e_cur0 = eng.score(itc)
    eng.sync(current, gate=gated)
    st_eff = effect_agreement(score(itc, zero), t_cur, e_cur0, e_cur)
    st_eff["visible"] = st_eff["effect_mean"] >= visible
    res["start_effect"] = st_eff
    res["start_pass"] = bool(res["start"]["mean"] <= bound_dec and (not st_eff["visible"] or agrees(st_eff, a)))
    res["seconds"] = round(time.time() - t0, 1)
    load_state(model, current)
    return res


def mem_probe(model, dev, micro_tokens: int, seq_len: int, pack: bool) -> dict:
    """One forward+backward of a full micro-row of seq_len-token sequences (random ids), grads dropped after."""
    import torch

    n = max(1, micro_tokens // seq_len)
    g = torch.Generator().manual_seed(0)
    plen = min(64, seq_len // 2)
    items = [(torch.randint(1000, 100000, (seq_len,), generator=g).tolist(), plen, [True] * (seq_len - plen), 1.0)
             for _ in range(n)]
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rec = {"seqs": n, "seq_len": seq_len, "micro_tokens": micro_tokens, "oom": False}
    try:
        forward_logps(model, items, dev, micro_tokens, pack, norm=float(n * (seq_len - plen)))
    except torch.OutOfMemoryError as e:
        rec.update(oom=True, error=str(e)[:300])
    for _, p in trainable(model):
        p.grad = None
    rec["fwd_bwd_s"] = round(time.time() - t0, 2)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        free, tot = torch.cuda.mem_get_info()
        rec.update(max_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   max_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                   device_free_gib=round(free / 2**30, 2), device_total_gib=round(tot / 2**30, 2))
    return rec


# ------------------------------------------------------------------------------------------------ run
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True, help="the frozen rl_pool.jsonl (hash-checked)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--arm", choices=("wide", "wide1", "ctrl"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, required=True, help="thinking cap, set by the step 9 validation job")
    ap.add_argument("--gpu_mem", type=float, required=True, help="vLLM gpu_memory_utilization")
    ap.add_argument("--micro_tokens", type=int, required=True, help="tokens per training micro-row")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--answer_max", type=int, default=384)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--length_on", choices=("completion", "reasoning"), required=True,
                    help="lengths for the length penalty: whole completion, or the sampled reasoning tokens only")
    ap.add_argument("--init_from", default="", help="a warm-up run's ckpt_step{N}.pt: initial LoRA A/B (gate, "
                    "optimizer, schedule fresh); ignored by --resume once the run has its own checkpoints")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--w_lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--no_pack", dest="pack", action="store_false")
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--max_batched_tokens", type=int, default=8192)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--stop_after", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--val_data", default="", help="frozen val.jsonl for the report-only periodic probe")
    ap.add_argument("--val_every", type=int, default=0)
    ap.add_argument("--val_n", type=int, default=128)
    ap.add_argument("--parity_prompts", type=int, default=8)
    ap.add_argument("--parity_samples", type=int, default=2)
    ap.add_argument("--parity_tokens", type=int, default=256)
    ap.add_argument("--parity_k", type=float, default=2.0)
    ap.add_argument("--parity_floor", type=float, default=1e-3)
    ap.add_argument("--probe_effect", type=float, default=0.2, help="probe size: mean |delta log p| per token")
    ap.add_argument("--effect_corr", type=float, default=0.9)
    ap.add_argument("--effect_slope_tol", type=float, default=0.1)
    ap.add_argument("--effect_visible", type=float, default=5.0)
    ap.add_argument("--mem_probe_len", type=int, default=0, help="0 = prompt 256 + cap + answer_max")
    ap.add_argument("--mem_probe_tokens", type=int, default=0, help="micro-row size to probe (0 = --micro_tokens)")
    ap.add_argument("--mem_probe_soft", action="store_true", help="smoke only: record a probe OOM, halve the row until one fits, continue")
    ap.add_argument("--bench_pack", action="store_true", help="time packed vs padded on the first step's batch")
    ap.add_argument("--device", default="cuda")
    return ap


def env_receipt(a, R, NV) -> dict:
    import subprocess
    from importlib.metadata import version

    import torch
    import transformers

    d = {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
         "host": socket.gethostname(), "pylib": os.environ.get("WG_ML7_PYLIB"),
         "code_sha256": {Path(f).name: sha256_file(f)[:16] for f in (__file__, R.__file__, NV.__file__,
                                                                    NEXT / "broad" / "arms.py", NEXT / "broad" / "packing.py")},
         "model_revision": (Path(a.model) / "REVISION").read_text().strip() if (Path(a.model) / "REVISION").exists() else None}
    for pkg in ("vllm", "peft", "fla-core", "math-verify", "triton"):
        try:
            d[pkg] = version(pkg)
        except Exception as e:  # noqa: BLE001
            d[pkg] = f"absent: {type(e).__name__}"
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
    try:
        d["git_describe"] = subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty", "--tags"],
                                           capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        d["git_describe"] = f"unavailable: {e}"
    return d


def run(a, make_engine=None) -> dict:
    """The trainer. make_engine(a, max_model_len, th) builds the rollout engine (default: VEngine); tests pass a
    CPU engine with the same interface (sync, sample, info)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data_rec = {"pool": verify_frozen(a.pool, "rl_pool.jsonl", a.manifest)}
    if a.val_every:
        if not a.val_data:
            raise SystemExit("--val_every needs --val_data (the frozen val.jsonl)")
        data_rec["val"] = verify_frozen(a.val_data, "val.jsonl", a.manifest)
    import gate_vllm as NV

    if make_engine is None:
        NV.install("stock" if a.arm == "ctrl" else a.arm, lora_rank=a.rank)  # sets VLLM_CACHE_ROOT before vllm import
    import numpy as np
    import torch

    import arms
    import packing
    import rl_common as R
    from transformers import AutoTokenizer

    mv = require_grader()
    dev = torch.device("cuda:0" if a.device == "cuda" else a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    th = R.Thinker(tok)
    decode = lambda t: tok.decode(t, skip_special_tokens=True)  # noqa: E731
    pool = load_rows(a.pool)
    order = np.random.default_rng(a.seed).permutation(len(pool))
    if a.steps * a.prompts > len(order) - a.parity_prompts:
        raise SystemExit("pool smaller than the run plus the parity prompts")
    parity_rows = [pool[j] for j in order[len(order) - a.parity_prompts :]]  # rows the run never draws
    val = load_rows(a.val_data)[: a.val_n] if a.val_every else []
    max_prompt = max(len(th.prompt(r["problem"])) for r in pool + val)
    max_model_len = max_prompt + a.cap + len(th.force) + a.answer_max + 16

    t_start = time.time()
    # the training model is built BEFORE the engine: constructing vLLM's ModelConfig changes how transformers resolves
    # this checkpoint's config afterwards (AutoModelForCausalLM then gets the full Qwen3_5Config and fails on
    # vocab_size; reproduced on CPU, 2026-10-08). vLLM's memory budget is measured from its own baseline, so the
    # model allocated first does not shrink the engine's gpu_memory_utilization share.
    torch.manual_seed(a.seed)
    dmap = {"": 0} if dev.type == "cuda" else {"": "cpu"}
    model, _ = arms.load_model(a.model, a.arm, device_map=dmap, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = arms.add_lora(model, a.rank)
    resume_step = latest_complete(out) if a.resume else 0
    init_rec = None
    if a.init_from and resume_step:  # the run's own checkpoint carries the init plus its updates
        init_rec = {"path": a.init_from, "applied": False,
                    "reason": f"resumed from the run's ckpt_step{resume_step}.pt",
                    "sha256": sha256_file(a.init_from) if Path(a.init_from).is_file() else None}
        if (out / "meta.json").exists():
            init_rec["original"] = json.loads((out / "meta.json").read_text()).get("init_from")
    elif a.init_from:
        init_rec = init_lora_from(model, a.init_from)
    packing.install(model)
    packing.CONV_BACKEND = "torch"
    gdn_train = {"kernel": "fla" if packing.fla_active() else "torch_reference", "dtype": "bf16" if dev.type == "cuda" else "fp32"}
    if dev.type == "cuda" and gdn_train["kernel"] != "fla":
        raise SystemExit("training forward without fla on GPU: set WG_ML7_PYLIB to a dir holding fla-core 0.5.2")
    named = trainable(model)
    lora_p = [p for n, p in named if "gate_w" not in n]
    w_p = [p for n, p in named if "gate_w" in n]
    assert (len(w_p) > 0) == (a.arm != "ctrl")
    groups = [{"params": lora_p, "lr": a.lr}] + ([{"params": w_p, "lr": a.w_lr}] if w_p else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    warm = max(1, int(a.steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    params = lora_p + w_p
    eng = (make_engine or (lambda a_, mml, th_: VEngine(a_, a_.model, a_.arm, mml, th_.stops)))(a, max_model_len, th)

    start = 0
    dropped = {}
    if a.resume:
        start = resume_step
        if start:
            load_state(model, torch.load(out / f"ckpt_step{start}.pt", map_location="cpu"))
            o = torch.load(out / f"opt_step{start}.pt", map_location="cpu")
            assert o["step"] == start, (o["step"], start)
            opt.load_state_dict(o["opt"])
            sched.load_state_dict(o["sched"])
        for f in ("train_log.jsonl", "rollouts.jsonl"):
            dropped[f] = truncate_log(out / f, start)
        dropped["val_log.jsonl"] = truncate_log(out / "val_log.jsonl", start + 1, key="after_step")
    elif (out / "train_log.jsonl").exists():
        raise SystemExit(f"{out} already holds a run; pass --resume or pick a new --out")

    current = state_of(model)
    opt_at_start = {"state_entries": len(opt.state), "sched_last_epoch": sched.last_epoch, "start_step": start,
                    "lr": sched.get_last_lr()[0]}
    parity = parity_check(eng, model, a.arm, parity_rows, th, a, dev, current)
    meta = {"args": vars(a), "data": data_rec, "math_verify": mv, "train_sampling": TRAIN_SAMPLING,
            "length_on": a.length_on, "init_from": init_rec, "opt_at_start": opt_at_start,
            "resumed_from": start, "dropped_log_lines": dropped, "engine": eng.info(), "gdn_train": gdn_train,
            "parity": parity, "n_lora": sum(p.numel() for p in lora_p),
            "n_w": sum(p.numel() for p in w_p), "pool_rows": len(pool), "max_model_len": max_model_len,
            "env": env_receipt(a, R, NV), "startup_s": None}
    mname = "meta.json" if not start else f"meta_resume{start}.json"
    if not (parity["pass"] and parity["start_pass"]):
        (out / mname).write_text(json.dumps(meta, indent=1, default=str))
        brief = {"checks": parity["checks"], "start_pass": parity["start_pass"], "lora_effect": parity["lora_effect"],
                 "gate_effect": parity.get("gate_effect")}
        raise SystemExit(f"engine/trainer parity FAILED: {json.dumps(brief)}")
    (out / mname).write_text(json.dumps(meta, indent=1, default=str))
    print("PARITY", json.dumps(parity, default=str), flush=True)
    seq_len = a.mem_probe_len or (256 + a.cap + a.answer_max)
    m_tok = a.mem_probe_tokens or a.micro_tokens
    ladder = [mem_probe(model, dev, m_tok, min(seq_len, m_tok), a.pack)]
    while a.mem_probe_soft and ladder[-1]["oom"] and m_tok // 2 >= min(seq_len, a.micro_tokens):
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        m_tok //= 2  # smoke only: halve until a micro-row fits, so the receipt names the largest that does
        ladder.append(mem_probe(model, dev, m_tok, min(seq_len, m_tok), a.pack))
    meta["mem_probe"] = ladder[0]
    meta["mem_probe_ladder"] = ladder
    meta["startup_s"] = round(time.time() - t_start, 1)
    (out / mname).write_text(json.dumps(meta, indent=1, default=str))
    if meta["mem_probe"]["oom"]:
        if not a.mem_probe_soft:
            raise SystemExit(f"memory probe OOM: a full micro-row of {a.micro_tokens} tokens does not fit beside the "
                             f"engine (lower --micro_tokens or --gpu_mem): {meta['mem_probe']}")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({k: meta[k] for k in ("parity", "mem_probe", "engine", "gdn_train")}, default=str), flush=True)

    def val_probe(after_step):
        t0 = time.time()
        prompts = [th.prompt(r["problem"]) for r in val]
        s1 = [seed_of(a.seed, "val", r["id"], 1) for r in val]
        s2 = [seed_of(a.seed, "val", r["id"], 2) for r in val]
        rolls, ntok = rollouts(eng, prompts, a.cap, a.answer_max, s1, s2, th, decode)
        corr = [bool(R.correct(r.text, v["gold"])) for r, v in zip(rolls, val)]
        rec = {"after_step": after_step, "n": len(val), "acc": sum(corr) / len(corr),
               "reasoning_mean": float(np.mean([r.reasoning_tokens for r in rolls])),
               "forced_frac": float(np.mean([r.forced for r in rolls])), "gen_tokens": ntok,
               "s": round(time.time() - t0, 1)}
        with open(out / "val_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("VAL", json.dumps(rec), flush=True)

    if a.val_every and start == 0:
        val_probe(0)
    last = a.stop_after or a.steps
    for step in range(start, last):
        t0 = time.time()
        rows = [pool[j] for j in order[step * a.prompts : (step + 1) * a.prompts]]
        prompts = [th.prompt(r["problem"]) for r in rows for _ in range(a.group)]
        s1 = [seed_of(a.seed, step, k, 1) for k in range(len(prompts))]
        s2 = [seed_of(a.seed, step, k, 2) for k in range(len(prompts))]
        rolls, ntok = rollouts(eng, prompts, a.cap, a.answer_max, s1, s2, th, decode)
        t_gen = time.time() - t0
        corr = [bool(R.correct(r.text, rows[i // a.group]["gold"])) for i, r in enumerate(rolls)]
        lens = [len(r.completion) for r in rolls]
        ans = [answer_tokens(r) for r in rolls]
        rlen = reward_lengths(rolls, a.length_on)
        rew, adv = group_rewards(corr, rlen, a.group, a.alpha)
        norm = float(sum(sum(r.sampled_mask) for r in rolls))
        idx = [i for i in range(len(rolls)) if adv[i] != 0.0]
        items = [(rolls[i].prompt + rolls[i].completion, len(rolls[i].prompt), rolls[i].sampled_mask, adv[i]) for i in idx]
        model.train()
        bench = None
        if a.bench_pack and step == start and items:
            bench, grads = {}, {}
            for pk in (False, True, False, True):  # first pair warms the kernels (autotune), second is timed
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                tb = time.time()
                forward_logps(model, items, dev, a.micro_tokens, pk, norm=norm)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                bench["packed_s" if pk else "padded_s"] = round(time.time() - tb, 3)
                grads[pk] = [p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
                             for p in params]
                for p in params:
                    p.grad = None
            num = sum(float((x - y).norm() ** 2) for x, y in zip(grads[True], grads[False])) ** 0.5
            den = sum(float(y.norm() ** 2) for y in grads[False]) ** 0.5
            bench["grad_rel_diff_packed_vs_padded"] = num / max(den, 1e-30)
            del grads
        t1 = time.time()
        loss_sum, lps = forward_logps(model, items, dev, a.micro_tokens, a.pack, norm=norm) if items else (0.0, [])
        per_seq = []
        for i, lp in zip(idx, lps):
            rl = sampled_lp(rolls[i])
            assert len(rl) == len(lp), (len(rl), len(lp))
            per_seq.append([x - y for x, y in zip(lp, rl)])
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t1
        sync_rec = eng.sync(state_of(model))
        grp_acc = [sum(corr[g * a.group : (g + 1) * a.group]) / a.group for g in range(a.prompts)]
        rec = {"step": step, "acc": sum(corr) / len(corr),
               "reasoning_mean": float(np.mean([r.reasoning_tokens for r in rolls])),
               "answer_mean": float(np.mean(ans)), "length_mean": float(np.mean(rlen)),
               "completion_mean": float(np.mean(lens)), "forced_frac": float(np.mean([r.forced for r in rolls])),
               "reward_mean": float(np.mean(rew)), "groups_all_correct": sum(x == 1 for x in grp_acc),
               "groups_all_wrong": sum(x == 0 for x in grp_acc), "nonzero_adv": len(idx), "loss_sum": loss_sum,
               "grad_norm": float(gn), "sampled_tokens": norm, "gen_tokens": ntok, "gen_s": round(t_gen, 2),
               "train_s": round(t_train, 2), "sync_s": round(sync_rec["sync_s"], 2),
               "step_s": round(time.time() - t0, 2), "gen_tok_per_s": round(ntok / max(t_gen, 1e-9), 1),
               "lr": sched.get_last_lr()[0], **ratio_stats(per_seq)}
        if bench:
            rec["bench_pack"] = bench
        if dev.type == "cuda":
            rec["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if w_p:
            rec["w_norm"] = sum(p.float().norm().item() ** 2 for p in w_p) ** 0.5
        with open(out / "train_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(out / "rollouts.jsonl", "a") as f:
            for i, r in enumerate(rolls):
                d = {"step": step, "row": rows[i // a.group]["id"], "k": i % a.group, "correct": corr[i],
                     "reasoning": r.reasoning_tokens, "answer": ans[i], "completion": lens[i], "forced": r.forced,
                     "reward": rew[i], "adv": adv[i]}
                if i < 2:
                    d["text"] = decode(r.completion)[-1500:]
                f.write(json.dumps(d) + "\n")
        print(json.dumps(rec), flush=True)
        if (step + 1) % a.ckpt_every == 0 or step + 1 == a.steps or step + 1 == last:
            atomic_save(state_of(model), out / f"ckpt_step{step + 1}.pt")
            atomic_save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1},
                        out / f"opt_step{step + 1}.pt")
            for p in out.glob("opt_step*.pt"):
                if int(p.stem[len("opt_step"):]) < step + 1:
                    p.unlink()
        if a.val_every and ((step + 1) % a.val_every == 0 or step + 1 == a.steps):
            val_probe(step + 1)
    summary = {"steps": a.steps, "ran": [start, last], "wall_s": round(time.time() - t_start, 1)}
    if last == a.steps:
        (out / "DONE").write_text(json.dumps(summary))
    print("END", json.dumps(summary), flush=True)
    return {"meta": meta, "summary": summary, "start_state": current}


def main():
    a = build_parser().parse_args()
    if a.stop_after and a.stop_after > a.steps:
        raise SystemExit("--stop_after beyond --steps")
    run(a)


if __name__ == "__main__":
    main()
