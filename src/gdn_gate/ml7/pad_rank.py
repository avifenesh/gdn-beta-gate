#!/usr/bin/env python3
"""pad_rank.py: widen a LoRA trainable state (a warm-up run's ckpt_step{N}.pt) from rank r to rank R > r, for the
parameter-matched control arm ctrlpm of the Qwen3.5-4B record (owner approval 2026-10-11 ~23:55 Israel).

Per LoRA pair (A: r x in, B: out x r, PEFT layout, scale alpha / r = 2 with alpha = 2 r at every rank):
  A' = [A; A_new]   A_new: (R - r) x in, PEFT's own LoRA A init (kaiming_uniform with a = sqrt(5), i.e.
                    U(-1/sqrt(in), 1/sqrt(in))), drawn from a generator seeded by (seed, key), so each arm seed
                    gets its own fresh components and the draw does not depend on key order
  B' = [B, 0]       the new columns are exactly zero
so B' A' x = B A x: at init the widened LoRA computes what the rank-r state computes (the forward of a ctrlpm run
at step 0 is the warm-up's), and the new components train like a fresh LoRA (B gets a gradient, A follows). The
trainer is unchanged: train_grpo_vllm.py --arm ctrl --rank R --init_from OUT loads OUT through init_lora_from, whose
key, shape and finiteness checks apply as for any warm-up.

Writes OUT (the widened state, same keys) and OUT.pad.json (source path and sha256, ranks, seed, sha256 of OUT,
per-key checks: first r components bit-equal to the source, new B columns zero, new A within the init bound).
Refuses a state with gate tensors, a non-LoRA tensor, a half pair, mixed ranks, or R <= r.

  pad_rank.py --src WARM/ckpt_step40.pt --rank 20 --seed S --out DIR/ctrlpm_sS_r20_init.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

KEY = re.compile(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])\.(?P<ad>[^.]+)\.weight$")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gen_seed(seed: int, key: str) -> int:
    return int(hashlib.sha256(f"ctrlpm_pad:{seed}:{key}".encode()).hexdigest()[:15], 16)


def pad_state(st: dict, rank: int, seed: int) -> tuple[dict, dict]:
    """Return (widened state, per-key record). Pure function of (st, rank, seed)."""
    import torch

    bad = [k for k in st if "gate_w" in k]
    if bad:
        raise SystemExit(f"pad_rank: gate tensors in the source ({bad[:2]}): a warm-up state has none")
    pairs: dict = {}
    for k, v in st.items():
        m = KEY.match(k)
        if not m or not torch.is_tensor(v) or v.dim() != 2:
            raise SystemExit(f"pad_rank: {k} is not a 2-D LoRA A/B tensor")
        pairs.setdefault((m["mod"], m["ad"]), {})[m["ab"]] = k
    half = [p for p, d in pairs.items() if set(d) != {"A", "B"}]
    if half:
        raise SystemExit(f"pad_rank: half LoRA pairs {half[:2]}")
    ranks = {st[d["A"]].shape[0] for d in pairs.values()} | {st[d["B"]].shape[1] for d in pairs.values()}
    if len(ranks) != 1:
        raise SystemExit(f"pad_rank: mixed ranks {sorted(ranks)}")
    r = ranks.pop()
    if rank <= r:
        raise SystemExit(f"pad_rank: target rank {rank} must exceed the source rank {r}")
    out, rec = {}, {}
    for (mod, ad), d in sorted(pairs.items()):
        A, B = st[d["A"]], st[d["B"]]
        fan_in = A.shape[1]
        bound = 1.0 / math.sqrt(fan_in)  # kaiming_uniform_(a=sqrt(5)): gain sqrt(1/3), bound = sqrt(3) gain / sqrt(fan_in)
        g = torch.Generator().manual_seed(gen_seed(seed, d["A"]))
        A_new = torch.empty(rank - r, fan_in, dtype=torch.float32).uniform_(-bound, bound, generator=g).to(A.dtype)
        out[d["A"]] = torch.cat([A, A_new], 0).contiguous()
        out[d["B"]] = torch.cat([B, torch.zeros(B.shape[0], rank - r, dtype=B.dtype)], 1).contiguous()
        rec[d["A"]] = {"shape": list(out[d["A"]].shape), "new_abs_max": float(A_new.abs().max()), "bound": bound}
        rec[d["B"]] = {"shape": list(out[d["B"]].shape)}
    return out, {"rank_from": r, "rank_to": rank, "pairs": len(pairs), "per_key": rec}


def verify(src: dict, out: dict, r: int, rank: int) -> list[str]:
    """Every way OUT is not a faithful widening of SRC; empty = OK."""
    import torch

    bad = []
    if set(src) != set(out):
        bad.append("key sets differ")
        return bad
    for k, v in src.items():
        w = out[k]
        if ".lora_A." in k:
            if tuple(w.shape) != (rank, v.shape[1]) or not torch.equal(w[:r], v):
                bad.append(f"{k}: first {r} rows not bit-equal to the source or wrong shape {tuple(w.shape)}")
            elif not bool(torch.isfinite(w).all()) or float(w[r:].abs().max()) > 1.0 / math.sqrt(v.shape[1]) + 1e-12:
                bad.append(f"{k}: new rows non-finite or above the init bound")
            elif float(w[r:].abs().max()) == 0.0:
                bad.append(f"{k}: new rows all zero (dead capacity: neither A nor B would get a gradient)")
        else:
            if tuple(w.shape) != (v.shape[0], rank) or not torch.equal(w[:, :r], v):
                bad.append(f"{k}: first {r} columns not bit-equal to the source or wrong shape {tuple(w.shape)}")
            elif bool((w[:, r:] != 0).any()):
                bad.append(f"{k}: new columns not exactly zero")
    return bad


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = torch.load(a.src, map_location="cpu")
    out, rec = pad_state(src, a.rank, a.seed)
    bad = verify(src, out, rec["rank_from"], a.rank)
    if bad:
        raise SystemExit("pad_rank: verification failed: " + "; ".join(bad[:3]))
    o = Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    tmp = o.with_name(o.name + ".tmp")
    torch.save(out, tmp)
    tmp.replace(o)
    n_src = sum(v.numel() for v in src.values())
    n_out = sum(v.numel() for v in out.values())
    receipt = {"what": "LoRA rank widening for ctrlpm (pad_rank.py)", "source": str(Path(a.src).resolve()),
               "source_sha256": sha256_file(a.src), "out": str(o.resolve()), "sha256": sha256_file(o),
               "seed": a.seed, "rank_from": rec["rank_from"], "rank_to": a.rank, "pairs": rec["pairs"],
               "numel_source": n_src, "numel_out": n_out,
               "init": "new A rows U(-1/sqrt(in), 1/sqrt(in)) (PEFT kaiming_uniform a=sqrt(5)), generator seed "
                       "sha256('ctrlpm_pad:{seed}:{key}')[:15]; new B columns zero", "verified": True,
               "argv": sys.argv if argv is None else ["pad_rank.py"] + list(argv)}
    Path(str(o) + ".pad.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("sha256", "source_sha256", "rank_from", "rank_to", "numel_out")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
