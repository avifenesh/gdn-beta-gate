#!/usr/bin/env python3
"""Train one arm of the negative-eigenvalue retrofit and evaluate it.

Arms (identical except for the widened gate):
  ctrl  LoRA on every GDN layer's in_proj_qkv, in_proj_a, in_proj_b, out_proj
  wide  the same LoRA plus the zero-init widened-beta gate (patch.py), beta in (0, 2)
  wide1 the same LoRA and gate parameters as wide with the ceiling at 1, beta in (0, 1) (patch._wg_beta1)
Every arm sees the same task and replay batches for a given seed; wide and wide1 also start from the same LoRA draw
(their gate W is created the same way, so the RNG stream matches).

--no_grad_ckpt turns off activation recomputation (the decoder's gradient checkpointing). Recomputation reruns the
same forward, so the math is unchanged; it only trades memory for time. Off by default (the banked runs used it).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import patch
import tasks

LORA_TARGETS = r".*layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_a|in_proj_b|out_proj)"


def load(model_path: str, arm: str, qlora: bool):
    kw = dict(dtype=torch.bfloat16, device_map={"": 0})
    if qlora:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    new = patch.install_for_arm(model, arm)
    return model, new


def add_lora(model, new_params, rank: int):
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():
        if "gate_w" in n:
            p.requires_grad_(True)
    return model


def core(model):
    """Return (decoder, lm_head) under peft wrapping."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model, m.lm_head


def pad(batch_ids, pad_id):
    L = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, : len(x)] = torch.tensor(x)
        att[i, : len(x)] = 1
    return ids, att


def task_loss(model, exs, pad_id):
    dec, head = core(model)
    ids, att = pad([e.ids for e in exs], pad_id)
    h = dec(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
    rows, cols, labs = [], [], []
    for i, e in enumerate(exs):
        rows += [i] * len(e.label_pos)
        cols += e.label_pos
        labs += e.labels
    hs = h[torch.tensor(rows).cuda(), torch.tensor(cols).cuda()]
    lab_t = torch.tensor(labs).cuda()

    def head_ce(x, y):
        return F.cross_entropy(head(x).float(), y, reduction="none")

    # full-vocab CE in 1024-row chunks, logits recomputed in backward: same objective, bounded memory
    per = torch.cat([checkpoint(head_ce, hs[s : s + 1024], lab_t[s : s + 1024], use_reentrant=False)
                     for s in range(0, hs.shape[0], 1024)])
    owner = torch.tensor(rows).cuda()
    by_task = {}
    for i, e in enumerate(exs):
        by_task.setdefault(e.task, []).append(per[owner == i].mean())
    return per.mean(), {t: torch.stack(v).mean().item() for t, v in by_task.items()}


def lm_loss(model, chunk_ids):
    dec, head = core(model)
    ids = chunk_ids.cuda()
    h = dec(input_ids=ids).last_hidden_state[:, :-1]
    tgt = ids[:, 1:]
    total, n = 0.0, 0
    hf, tf = h.reshape(-1, h.shape[-1]), tgt.reshape(-1)
    for s in range(0, hf.shape[0], 1024):
        lg = head(hf[s : s + 1024]).float()
        total = total + F.cross_entropy(lg, tf[s : s + 1024], reduction="sum")
        n += lg.shape[0]
    return total / n


@torch.no_grad()
def eval_tasks(model, V, lengths, n_seq, batch, pad_id, seed=10_000):
    model.eval()
    dec, head = core(model)
    label_sets = {
        "parity": V.letter[:2], "z3": V.letter[:3], "swap": V.letter, "s5full": V.letter, "codeswap": V.letter,
    }
    out = {}
    for task in tasks.ALL_TASKS:
        cand = torch.tensor(label_sets[task]).cuda()
        wcand = head.weight[cand] if not hasattr(head, "base_layer") else head.base_layer.weight[cand]
        for L in lengths:
            rng = random.Random(seed + 7919 * L + 104729 * tasks.ALL_TASKS.index(task))
            exs = [tasks.make(task, L, rng, V) for _ in range(n_seq)]
            correct = torch.zeros(n_seq, L, dtype=torch.bool)
            for s in range(0, n_seq, batch):
                chunk = exs[s : s + batch]
                ids, att = pad([e.ids for e in chunk], pad_id)
                h = dec(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
                for i, e in enumerate(chunk):
                    hs = h[i, torch.tensor(e.label_pos).cuda()].to(wcand.dtype)
                    pred = (hs @ wcand.T).argmax(-1)
                    gold = torch.tensor([label_sets[task].index(l) for l in e.labels]).cuda()
                    correct[s + i] = (pred == gold).cpu()
            out[f"{task}@{L}"] = correct.float().mean(0).tolist()  # per-step accuracy over sequences
            out[f"{task}@{L}/seq"] = correct.float().mean(1).tolist()  # per-sequence accuracy
            wins = [(0, 64), (64, 256), (256, 1024)]
            out[f"{task}@{L}/win"] = [  # per-sequence accuracy in steps 1-64, 65-256, 257-1024
                [correct[i, a:min(b, L)].float().mean().item() if a < L else None for a, b in wins]
                for i in range(n_seq)
            ]
    model.train()
    return out


@torch.no_grad()
def eval_val(model, V, pad_id, n_seq=32, L=256, seed=20_000):
    """Checkpoint eval on a validation set (different seed from the final test set): per-task
    accuracy in the 1x (steps 1-64) and 4x (65-256) windows, plus beta usage of the widened gate."""
    patch.enable_stats(model, True)
    res = eval_tasks(model, V, [L], n_seq, 8, pad_id, seed=seed)
    stats = patch.collect_stats(model)
    patch.enable_stats(model, False)
    out = {}
    for t in tasks.ALL_TASKS:
        w = torch.tensor([r[:2] for r in res[f"{t}@{L}/win"]])
        out[t] = {"w1": w[:, 0].mean().item(), "w4": w[:, 1].mean().item(),
                  "first8": [round(a, 3) for a in res[f"{t}@{L}"][:8]]}  # plateau signature
    out["tier1"] = {k: sum(out[t][k] for t in tasks.TIER1) / 3 for k in ("w1", "w4")}
    if stats["beta_gt1_frac_per_layer"]:
        out["beta_gt1_frac_mean"] = sum(stats["beta_gt1_frac_per_layer"]) / len(stats["beta_gt1_frac_per_layer"])
        out["beta_gt1_frac_max"] = max(stats["beta_gt1_frac_per_layer"])
    return out


@torch.no_grad()
def eval_nll(model, chunks, batch):
    model.eval()
    res = []
    for s in range(0, chunks.shape[0], batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res += [lm_loss(model, chunks[i : i + 1]).item() for i in range(s, min(s + batch, chunks.shape[0]))]
    model.train()
    return res


def replay_chunks(path, split, n, L, seed):
    obj = torch.load(path)
    x = obj[split]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:n]
    return x[idx, :L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=sorted(patch.ARMS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--train_len", type=int, default=64)
    ap.add_argument("--replay", required=True, help="torch file with {'train','test'} uint32 chunks")
    ap.add_argument("--replay_batch", type=int, default=4)
    ap.add_argument("--replay_every", type=int, default=2)
    ap.add_argument("--replay_micro", type=int, default=4, help="replay chunks per micro-batch")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w_lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--eval_lengths", default="64,256,1024")
    ap.add_argument("--eval_n", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--nll_n", type=int, default=256)
    ap.add_argument("--max_steps_debug", type=int, default=0)
    ap.add_argument("--init_from", default="", help="trainable-state file (trainable.pt or ckpt) to continue from")
    ap.add_argument("--task_counts", default="",
                    help="e.g. parity:16,swap:64,codeswap:64: fixed sequences per task per step (overrides --batch)")
    ap.add_argument("--micro_tokens", type=int, default=8192,
                    help="with --task_counts: max padded tokens per micro-batch (single-task micro-batches)")
    ap.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--sched", choices=["cosine", "constant"], default="cosine",
                    help="LR after warmup; constant for plateau-then-switch diagnostics")
    ap.add_argument("--train_tasks", default=",".join(tasks.ALL_TASKS),
                    help="comma list of tasks sampled for training; eval always covers all tasks")
    ap.add_argument("--eval_every", type=int, default=100, help="validation eval + checkpoint every N steps")
    ap.add_argument("--no_final_eval", action="store_true", help="sweep runs: skip the test-set eval")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="no decoder activation recomputation (same math, more memory, faster)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    V = tasks.Vocab(tok)
    tasks.reference_check(V)  # independent label recompute; raises on any mismatch
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model, new = load(args.model, args.arm, args.qlora)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:  # what gradient checkpointing forces in training: no KV / GDN cache objects built in the forward
        core(model)[0].config.use_cache = False
    model = add_lora(model, new, args.rank)
    if args.init_from:  # continue from a saved trainable state (fresh optimizer and schedule)
        state = torch.load(args.init_from)
        params = dict(model.named_parameters())
        missing = [k for k in state if k not in params]
        assert not missing, missing[:5]
        with torch.no_grad():
            for k, v in state.items():
                params[k].copy_(v.to(params[k].dtype))
        print(json.dumps({"init_from": args.init_from, "tensors": len(state)}), flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "gate_w" not in n]
    w_p = [p for n, p in model.named_parameters() if p.requires_grad and "gate_w" in n]
    groups = [{"params": lora_p, "lr": args.lr}]
    if w_p:
        groups.append({"params": w_p, "lr": args.w_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    total = args.max_steps_debug or args.steps
    warm = max(1, total // 20)
    decay = (lambda s: 1.0) if args.sched == "constant" else (
        lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * decay(s))
    meta = {
        "args": vars(args), "n_trainable": sum(p.numel() for p in trainable),
        "n_lora": sum(p.numel() for p in lora_p), "n_w": sum(p.numel() for p in w_p),
        "torch": torch.__version__,
    }
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}), flush=True)

    train_chunks = replay_chunks(args.replay, "train", total * args.replay_batch, 512, args.seed)
    train_tasks = args.train_tasks.split(",")
    assert all(t in tasks.ALL_TASKS for t in train_tasks), train_tasks
    task_counts = [(t, int(n)) for t, n in (x.split(":") for x in args.task_counts.split(","))] if args.task_counts else []
    assert all(t in tasks.ALL_TASKS and n > 0 for t, n in task_counts), task_counts
    # task stream: identical across arms for a seed
    rng = random.Random(args.seed)
    log = []
    t0 = time.time()
    tok_count = 0
    model.train()
    acc_loss = {}
    if args.eval_every:  # step-0 validation eval: where each task starts
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v0 = eval_val(model, V, pad_id)
        model.train()
        v0["step"] = 0
        with open(out / "val_log.jsonl", "a") as f:
            f.write(json.dumps(v0) + "\n")
        print("VAL", json.dumps({"step": 0, "tier1": v0["tier1"]}), flush=True)
    for step in range(total):
        if task_counts:
            # fixed sequences per task per step; micro-batches hold one task each (no cross-task padding)
            exs, groups = [], []
            for t, n in task_counts:
                g = [tasks.make(t, args.train_len, rng, V) for _ in range(n)]
                per_mb = max(1, args.micro_tokens // len(g[0].ids)) if args.micro_tokens else n
                exs += g
                groups += [g[i : i + per_mb] for i in range(0, n, per_mb)]
        else:
            # --batch task sequences per optimizer step, in --grad_accum micro-batches (same stream when 1)
            exs = [tasks.make(rng.choice(train_tasks), args.train_len, rng, V) for _ in range(args.batch)]
            mb = -(-len(exs) // args.grad_accum)
            groups = [exs[a : a + mb] for a in range(0, len(exs), mb)]
        per_task, loss_sum = {}, 0.0
        for g in groups:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_mb, pt = task_loss(model, g, pad_id)
            (l_mb * len(g) / len(exs)).backward()
            loss_sum += l_mb.item() * len(g) / len(exs)
            for t, v in pt.items():
                per_task.setdefault(t, []).append(v)
        per_task = {t: sum(v) / len(v) for t, v in per_task.items()}
        loss = torch.tensor(loss_sum)
        rl = torch.zeros(())
        replay_on = bool(args.replay_every and step % args.replay_every == 0)
        if replay_on:
            # replay chunks in micro-batches of --replay_micro (one micro-batch at the defaults: same math as before)
            ch = train_chunks[(step // args.replay_every) * args.replay_batch:][: args.replay_batch]
            rl_sum = 0.0
            for a in range(0, len(ch), args.replay_micro):
                sub = ch[a : a + args.replay_micro]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l_r = lm_loss(model, sub.long())
                (l_r * len(sub) / len(ch)).backward()
                rl_sum += l_r.item() * len(sub) / len(ch)
            rl = torch.tensor(rl_sum)
            tok_count += ch.numel()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        tok_count += sum(len(e.ids) for e in exs)
        for t, v in per_task.items():
            acc_loss.setdefault(t, []).append(v)
        if replay_on:
            acc_loss.setdefault("replay", []).append(rl.item())
        if step % 25 == 0 or step == total - 1:
            rec = {"step": step, "task_loss": loss.item(), "replay_loss": rl.item(),
                   "loss_by_task_25": {t: round(sum(v) / len(v), 4) for t, v in acc_loss.items()},
                   "elapsed_s": round(time.time() - t0, 1), "tok_per_s": round(tok_count / (time.time() - t0), 1),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            if w_p:
                rec["w_norm"] = sum(p.float().norm().item() ** 2 for p in w_p) ** 0.5
            log.append(rec)
            print(json.dumps(rec), flush=True)
            acc_loss = {}
        if args.eval_every and ((step + 1) % args.eval_every == 0 or step == total - 1):
            model.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = eval_val(model, V, pad_id)
            model.train()
            v["step"] = step + 1
            with open(out / "val_log.jsonl", "a") as f:
                f.write(json.dumps(v) + "\n")
            print("VAL", json.dumps({"step": step + 1, "tier1": v["tier1"],
                                     **{t: v[t] for t in tasks.ALL_TASKS},
                                     "beta_gt1": v.get("beta_gt1_frac_mean")}), flush=True)
            torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                       out / f"ckpt_step{step + 1}.pt")
    meta["train_seconds"] = time.time() - t0
    json.dump(log, open(out / "train_log.json", "w"))

    # save trainable state
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, out / "trainable.pt")

    if args.no_final_eval:
        json.dump({"meta": meta}, open(out / "result_sweep.json", "w"))
        print("DONE", out, flush=True)
        return
    lengths = [int(x) for x in args.eval_lengths.split(",")]
    patch.enable_stats(model, True)
    t1 = time.time()
    res = eval_tasks(model, V, lengths, args.eval_n, args.eval_batch, pad_id)
    meta["eval_seconds"] = time.time() - t1
    stats = patch.collect_stats(model)
    patch.enable_stats(model, False)
    test_chunks = replay_chunks(args.replay, "test", args.nll_n, 512, 1234)
    nll = eval_nll(model, test_chunks.long(), 8)
    json.dump({"meta": meta, "tasks": res, "nll": nll, "wg_stats": stats}, open(out / "result.json", "w"))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
