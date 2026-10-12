#!/usr/bin/env python3
"""Step 6 trainer on a fast stack (stacks.py): train_sft.py's run, recipe, data stream, checkpoints and results
layout, with the model built and the loss computed by the chosen stack.

Same as train_sft.py --pack: arms, LoRA r16 alpha 32 on the GDN projections, AdamW (0.9, 0.95) wd 0, lr 1e-4 for LoRA
and gate, 5% warmup then constant, grad clip 1.0, 1,200 steps of about 33,472 tokens in packed rows of at most 24,576,
the seed's example stream, val loss (padded path, train_sft.val_eval) at step 0 and every checkpoint.
Outputs in --out, as train_sft.py: ckpt_step{N}.pt (trainable tensors under train_sft.py's names, so arms.load_trained
and the battery read them unchanged), opt_step{N}.pt (for --resume), val_log.jsonl, train_log.jsonl, meta.json
(+ the stack's build info), DONE.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--stack", required=True, choices=["unsloth", "ours"])
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--arm", choices=["wide", "wide1", "ctrl"], required=True)
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
ap.add_argument("--stop_after", type=int, default=0)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--gc", default="unsloth")
ap.add_argument("--loss", default="model", choices=["model", "ours"])
ap.add_argument("--init_from", default="", help="initial trainable tensors (dump_init.py: train_sft.py's draw for this arm and seed)")
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stacks  # noqa: E402

stacks.early(args.stack)

import torch  # noqa: E402

import packing  # noqa: E402
import patch  # noqa: E402
import sft_data  # noqa: E402
import train_sft  # noqa: E402

OURS_PREFIX = "base_model.model.model."  # train_sft.py's trainable-tensor names = prefix + stacks.canon(name)


def main():
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    R = sft_data.Renderer(tok)
    mix = sft_data.Mixture(args.data)
    val = sft_data.val_set(mix, R, args.val_n, args.max_len)
    model, info = stacks.build(args.stack, args.model, args.arm, rank=args.rank, gc=args.gc, seed=args.seed)
    named = stacks.trainable_named(model)
    lora_p = [p for n, p in named if "gate_w" not in n]
    w_p = [p for n, p in named if "gate_w" in n]
    groups = [{"params": lora_p, "lr": args.lr}] + ([{"params": w_p, "lr": args.w_lr}] if w_p else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    warm = max(1, int(args.steps * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    trainable = lora_p + w_p
    ckpts = sorted(int(x) for x in args.ckpt_steps.split(",") if x)

    def state_dict_ours():
        return {OURS_PREFIX + n: p.detach().cpu() for n, p in named}

    def load_ours(state):
        src = {stacks.canon(k): v for k, v in state.items()}
        with torch.no_grad():
            for n, p in named:
                p.copy_(src[n].to(p.dtype))

    if args.init_from:  # replaced by the checkpoint below when resuming
        info["init_from"] = args.init_from
        info["init_loaded"] = stacks.load_trainable(model, torch.load(args.init_from, map_location="cpu"))
    stream = sft_data.Stream(mix, R, args.seed, args.max_len)
    start = 0
    if args.resume:
        done = [s for s in ckpts if (out / f"ckpt_step{s}.pt").exists() and (out / f"opt_step{s}.pt").exists()]
        if done:
            start = max(done)
            load_ours(torch.load(out / f"ckpt_step{start}.pt"))
            o = torch.load(out / f"opt_step{start}.pt")
            opt.load_state_dict(o["opt"])
            sched.load_state_dict(o["sched"])
            for _ in range(start):
                stream.step(args.step_tokens)
            assert stream.pos == o["stream_pos"], (stream.pos, o["stream_pos"])

    env = train_sft.env_receipt(args.model)
    env["code_sha256_fast"] = {f: train_sft.subprocess.run(f"sha256sum {HERE / f} | cut -c1-16", shell=True,
                                                           capture_output=True, text=True).stdout.strip()
                               for f in ("stacks.py", "train_sft_fast.py")}
    meta = {"args": vars(args), "env": env, "stack_info": info, "n_lora": sum(p.numel() for p in lora_p),
            "n_w": sum(p.numel() for p in w_p), "n_val": len(val), "train_rows": int(len(mix.train_rows)),
            "val_rows": int(len(mix.val_rows)), "resumed_from": start}
    (out / ("meta.json" if not start else f"meta_resume{start}.json")).write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}, default=str), flush=True)

    def val_eval():
        # the padded path, as train_sft.val_eval (packing wrappers delegate unchanged without packing kwargs)
        model.eval()
        patch.enable_stats(model, True)
        tot, n = 0.0, 0
        with torch.no_grad():
            for g in sft_data.micro_batches(val, args.micro_tokens):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dec, head = stacks.core(model)
                    ids, lab, att = train_sft.pad(g, pad_id)
                    h = dec(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
                    tgt = lab[:, 1:].to(dev)
                    sel = tgt != -100
                    hs, y = h[:, :-1][sel], tgt[sel]
                    s = sum(torch.nn.functional.cross_entropy(head(hs[i:i + 1024]).float(), y[i:i + 1024], reduction="sum")
                            for i in range(0, hs.shape[0], 1024))
                tot += float(s)
                n += int(y.numel())
        st = patch.collect_stats(model)
        patch.enable_stats(model, False)
        model.train()
        r = {"val_loss": tot / max(n, 1), "val_tokens": n}
        fr = st["beta_gt1_frac_per_layer"]
        if fr:
            r["gate_stat_mean"] = sum(fr) / len(fr)
            r["gate_stat_max"] = max(fr)
            r["abs_t_mean"] = sum(st["abs_t_per_layer"]) / len(st["abs_t_per_layer"])
        return r

    if start == 0:
        v0 = val_eval()
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
        mbs = packing.groups(exs, args.micro_tokens)
        n_lab = sum(sum(1 for l in e.labels[1:] if l != -100) for e in exs)
        loss_sum = 0.0
        for g in mbs:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.stack == "unsloth" and args.loss == "model":
                    l, _ = stacks.loss_model(model, g, dev, n_lab)
                    l.backward()
                    loss_sum += float(l.detach()) * n_lab
                else:
                    s, _ = stacks.loss_ours_row(model, g, dev)
                    (s / n_lab).backward()
                    loss_sum += float(s.detach())
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
                   "elapsed_s": round(el, 1), "tok_per_s": round(tok_count / el, 1),
                   "step_tokens": sum(len(e.ids) for e in exs), "n_ex": len(exs), "n_micro": len(mbs),
                   "label_frac": round(lab_count / max(tok_count, 1), 3), "stream_pos": stream.pos,
                   "skipped": stream.skipped, "lr": sched.get_last_lr()[0],
                   "max_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
            if w_p:
                rec["w_norm"] = round(sum(p.float().norm().item() ** 2 for p in w_p) ** 0.5, 5)
            with open(out / "train_log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            win_loss, win_n = 0.0, 0
        if step + 1 in ckpts:
            v = val_eval()
            v["step"] = step + 1
            with open(out / "val_log.jsonl", "a") as f:
                f.write(json.dumps(v) + "\n")
            print("VAL", json.dumps(v), flush=True)
            torch.save(state_dict_ours(), out / f"ckpt_step{step + 1}.pt")
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "stream_pos": stream.pos},
                       out / f"opt_step{step + 1}.pt")
            for s in [s for s in ckpts if s < step + 1]:
                p = out / f"opt_step{s}.pt"
                if p.exists():
                    p.unlink()
    summary = {"steps_run": last - start, "train_seconds": time.time() - t0, "tokens": tok_count,
               "label_tokens": lab_count, "tok_per_s": tok_count / max(time.time() - t0, 1e-9),
               "stream_pos": stream.pos, "skipped": stream.skipped, "stack": args.stack}
    (out / f"summary_{start}_{last}.json").write_text(json.dumps(summary, indent=1))
    if last == args.steps:
        (out / "DONE").write_text(json.dumps(summary))
    print("END", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
