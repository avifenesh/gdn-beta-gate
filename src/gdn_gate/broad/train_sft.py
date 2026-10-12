#!/usr/bin/env python3
"""Step 6 (reduced): broad SFT retrofit of Qwen3.5-4B-Base, one arm and seed per run.

Arms (arms.py): wide (beta in (0,2)), wide1 (beta in (0,1), same added parameters), ctrl (LoRA only).
LoRA r16 on every GDN layer's in_proj_qkv, in_proj_a, in_proj_b, out_proj (the lane's targets), alpha 32.
Recipe kept from the banked 4B main runs where it applies: AdamW (0.9, 0.95), wd 0, lr 1e-4 for LoRA and
gate, 5% warmup then constant, grad clip 1.0, bf16 autocast, gradient checkpointing, 1,200 steps of about
33,472 tokens (the tier-1 run size, about 40.2M tokens). Data: sft_data.py (Tulu 3 minus No Robots).
For a seed, every arm sees the identical example stream and micro-batch split.

--pack (default on CUDA): each micro-batch is one packed row of whole examples (packing.py) instead of a padded
length-sorted group: no padding tokens, position ids, attention, GDN state and conv reset at every example, the
same (position, target) pairs and the same per-step token-mean loss. check_pack.py proves boundary isolation. The
padded path (--no-pack, and the val loss either way) is unchanged.

Writes to --out: ckpt_step{N}.pt (trainable tensors) and opt_step{N}.pt (optimizer, for --resume) at
--ckpt_steps, val_log.jsonl (held-out SFT loss and gate statistics at step 0 and each checkpoint),
train_log.jsonl (every 25 steps), meta.json, and DONE when the last step is saved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import arms
import packing
import sft_data
import patch


PACK_DEFAULT_CUDA = True  # 2026-10-08: GB10 A/B at the step 6 config 941 vs 569 tok/s (x1.654), losses within 7.6e-4 rel over 30 steps; leak checks: red arms 2,170-25,000x above bound (sftspeed/v4/SUMMARY.txt)


def pad(exs, pad_id):
    L = max(len(e.ids) for e in exs)
    ids = torch.full((len(exs), L), pad_id, dtype=torch.long)
    lab = torch.full((len(exs), L), -100, dtype=torch.long)
    att = torch.zeros((len(exs), L), dtype=torch.long)
    for i, e in enumerate(exs):
        ids[i, : len(e.ids)] = torch.tensor(e.ids)
        lab[i, : len(e.labels)] = torch.tensor(e.labels)
        att[i, : len(e.ids)] = 1
    return ids, lab, att


def sft_loss_sum(model, exs, pad_id, dev):
    """Sum of next-token CE over label positions, and their count. Full-vocab CE in 1,024-row chunks with the
    logits recomputed in backward (bounded memory, same objective)."""
    dec, head = arms.core(model)
    ids, lab, att = pad(exs, pad_id)
    h = dec(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
    tgt = lab[:, 1:].to(dev)
    sel = tgt != -100
    hs = h[:, :-1][sel]
    y = tgt[sel]

    def head_ce(x, t):
        return F.cross_entropy(head(x).float(), t, reduction="sum")

    total = sum(checkpoint(head_ce, hs[s : s + 1024], y[s : s + 1024], use_reentrant=False)
                for s in range(0, hs.shape[0], 1024))
    return total, int(y.numel())


@torch.no_grad()
def val_eval(model, val, pad_id, dev, micro_tokens):
    model.eval()
    patch.enable_stats(model, True)
    tot, n = 0.0, 0
    for g in sft_data.micro_batches(val, micro_tokens):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            s, c = sft_loss_sum(model, g, pad_id, dev)
        tot += float(s)
        n += c
    st = patch.collect_stats(model)
    patch.enable_stats(model, False)
    model.train()
    out = {"val_loss": tot / max(n, 1), "val_tokens": n}
    fr = st["beta_gt1_frac_per_layer"]
    if fr:  # wide: share of beta > 1; wide1: share at the ceiling (beta >= 0.999)
        out["gate_stat_mean"] = sum(fr) / len(fr)
        out["gate_stat_max"] = max(fr)
        out["abs_t_mean"] = sum(st["abs_t_per_layer"]) / len(st["abs_t_per_layer"])
    return out


def env_receipt(model_path):
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"err {e}"

    import peft
    import transformers
    try:
        import fla
        fla_v = getattr(fla, "__version__", "?")
    except Exception:  # noqa: BLE001
        fla_v = None
    try:
        import triton
        tri_v = triton.__version__
    except Exception:  # noqa: BLE001
        tri_v = None
    here = Path(__file__).resolve().parent
    return {
        "host": platform.node(), "machine": platform.machine(), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__, "fla": fla_v, "triton": tri_v,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_describe": sh(f"git -C {here} describe --always --dirty"),
        "model_path": model_path,
        "code_sha256": {f: sh(f"sha256sum {here / f} | cut -c1-16")
                        for f in ("arms.py", "sft_data.py", "train_sft.py", "packing.py")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="parquet written by sft_data.py")
    ap.add_argument("--arm", choices=arms.ARMS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--step_tokens", type=int, default=33472)
    ap.add_argument("--micro_tokens", type=int, default=24576)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--ckpt_steps", default="300,700,1200")
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--stop_after", type=int, default=0, help="pilot: stop after this many steps (schedule unchanged)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pack", action=argparse.BooleanOptionalAction, default=None,
                    help="packed micro-batches (default: on for CUDA, off for CPU, where the reference GDN is per-row)")
    ap.add_argument("--conv_backend", default="torch", choices=["torch", "fla", "cc1d"],
                    help="packed short conv (packing.py); cc1d is opt-in only")
    args = ap.parse_args()
    if args.pack is None:
        args.pack = PACK_DEFAULT_CUDA and args.device != "cpu"

    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if args.device != "cuda" else "cuda:0")
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    R = sft_data.Renderer(tok)
    mix = sft_data.Mixture(args.data)
    val = sft_data.val_set(mix, R, args.val_n, args.max_len)

    dmap = {"": 0} if dev.type == "cuda" else {"": "cpu"}
    model, new = arms.load_model(args.model, args.arm, device_map=dmap,
                                 dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)  # fp32 only for CPU tests
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = arms.add_lora(model, args.rank)
    if args.pack:
        packing.install(model)
        packing.CONV_BACKEND = args.conv_backend
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "gate_w" not in n]
    w_p = [p for n, p in model.named_parameters() if p.requires_grad and "gate_w" in n]
    assert (len(w_p) > 0) == (args.arm != "ctrl"), (args.arm, len(w_p))
    groups = [{"params": lora_p, "lr": args.lr}] + ([{"params": w_p, "lr": args.w_lr}] if w_p else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    warm = max(1, int(args.steps * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    trainable = lora_p + w_p
    ckpts = sorted(int(x) for x in args.ckpt_steps.split(",") if x)

    stream = sft_data.Stream(mix, R, args.seed, args.max_len)
    start = 0
    if args.resume:
        done = [s for s in ckpts if (out / f"ckpt_step{s}.pt").exists() and (out / f"opt_step{s}.pt").exists()]
        if done:
            start = max(done)
            state = torch.load(out / f"ckpt_step{start}.pt")
            params = dict(model.named_parameters())
            with torch.no_grad():
                for k, v in state.items():
                    params[k].copy_(v.to(params[k].dtype))
            o = torch.load(out / f"opt_step{start}.pt")
            opt.load_state_dict(o["opt"])
            sched.load_state_dict(o["sched"])
            for _ in range(start):  # replay the stream to the same position (deterministic)
                stream.step(args.step_tokens)
            assert stream.pos == o["stream_pos"], (stream.pos, o["stream_pos"])

    meta = {"args": vars(args), "env": env_receipt(args.model), "n_lora": sum(p.numel() for p in lora_p),
            "n_w": sum(p.numel() for p in w_p), "n_val": len(val), "train_rows": int(len(mix.train_rows)),
            "val_rows": int(len(mix.val_rows)), "resumed_from": start}
    (out / ("meta.json" if not start else f"meta_resume{start}.json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}), flush=True)

    if start == 0:
        v0 = val_eval(model, val, pad_id, dev, args.micro_tokens)
        v0["step"] = 0
        with open(out / "val_log.jsonl", "a") as f:
            f.write(json.dumps(v0) + "\n")
        print("VAL", json.dumps(v0), flush=True)

    model.train()
    t0 = time.time()
    tok_count = lab_count = 0
    win_loss, win_n = 0.0, 0
    last = args.stop_after or args.steps
    for step in range(start, last):
        exs = stream.step(args.step_tokens)
        mbs = packing.groups(exs, args.micro_tokens) if args.pack else sft_data.micro_batches(exs, args.micro_tokens)
        n_lab = sum(sum(1 for l in e.labels[1:] if l != -100) for e in exs)
        loss_sum = 0.0
        for g in mbs:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                s, c = packing.loss_sum(model, g, dev) if args.pack else sft_loss_sum(model, g, pad_id, dev)
            (s / n_lab).backward()  # token-level mean over the whole step
            loss_sum += float(s)
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        tok_count += sum(len(e.ids) for e in exs)
        lab_count += n_lab
        win_loss += loss_sum
        win_n += n_lab
        if step % 25 == 0 or step == last - 1:
            el = time.time() - t0
            rec = {"step": step, "loss_25": round(win_loss / max(win_n, 1), 4), "grad_norm": round(float(gn), 4),
                   "elapsed_s": round(el, 1), "tok_per_s": round(tok_count / el, 1), "step_tokens": sum(len(e.ids) for e in exs),
                   "n_ex": len(exs), "n_micro": len(mbs), "label_frac": round(lab_count / max(tok_count, 1), 3),
                   "stream_pos": stream.pos, "skipped": stream.skipped, "lr": sched.get_last_lr()[0]}
            if dev.type == "cuda":
                rec["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                rec["max_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            if w_p:
                rec["w_norm"] = round(sum(p.float().norm().item() ** 2 for p in w_p) ** 0.5, 5)
            with open(out / "train_log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            win_loss, win_n = 0.0, 0
        if step + 1 in ckpts:
            v = val_eval(model, val, pad_id, dev, args.micro_tokens)
            v["step"] = step + 1
            with open(out / "val_log.jsonl", "a") as f:
                f.write(json.dumps(v) + "\n")
            print("VAL", json.dumps(v), flush=True)
            torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                       out / f"ckpt_step{step + 1}.pt")
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "stream_pos": stream.pos},
                       out / f"opt_step{step + 1}.pt")
            prev = [s for s in ckpts if s < step + 1]
            for s in prev:  # keep only the newest optimizer state
                p = out / f"opt_step{s}.pt"
                if p.exists():
                    p.unlink()
    summary = {"steps_run": last - start, "train_seconds": time.time() - t0, "tokens": tok_count,
               "label_tokens": lab_count, "tok_per_s": tok_count / max(time.time() - t0, 1e-9),
               "stream_pos": stream.pos, "skipped": stream.skipped}
    (out / f"summary_{start}_{last}.json").write_text(json.dumps(summary, indent=1))
    if last == args.steps:
        (out / "DONE").write_text(json.dumps(summary))
    print("END", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
