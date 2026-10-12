#!/usr/bin/env python3
"""hybrid_ckpts.py OUT_DIR: G1 "gate only" inputs. For wide1 and wide x seeds 0-2 (Qwen3.5-4B, step 150) write a
checkpoint that holds the shared warm-up's LoRA tensors (ctrl_s0_cap512_fmt_box/ckpt_step40.pt) and the trained gate
tensors of the arm run, as OUT_DIR/<arm>_s<k>_warmLoRA_gate150.pt, plus OUT_DIR/receipt.json (sources, output sha256).

Reconstructed for the release from the lab's inline build; on the lab checkpoints its outputs match the sha256 values
of the lab receipt (results/mech/hybrid_receipt.json) byte for byte.
Usage: python hybrid_ckpts.py OUT_DIR [--grpo DIR]   (DIR default $WG_RUNS/ml7_4b/grpo)"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--grpo", default=os.path.join(os.environ.get("WG_RUNS", "runs"), "ml7_4b/grpo"))
    a = ap.parse_args()
    g, out = Path(a.grpo), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    warm = torch.load(g / "ctrl_s0_cap512_fmt_box/ckpt_step40.pt", map_location="cpu", weights_only=True)
    rec = {}
    for arm in ("wide1", "wide"):
        for s in (0, 1, 2):
            src = g / f"{arm}_s{s}_cap512_r_box/ckpt_step150.pt"
            st = torch.load(src, map_location="cpu", weights_only=True)
            gate = {k: v for k, v in st.items() if "lora_" not in k}
            hy = {**warm, **gate}
            name = f"{arm}_s{s}_warmLoRA_gate150.pt"
            torch.save(hy, out / name)
            rec[f"hybrid/{name}"] = {"lora_from": "ctrl_s0_cap512_fmt_box/ckpt_step40.pt", "gate_from": str(src),
                                     "sha256": sha(out / name)}
            print(name, rec[f"hybrid/{name}"]["sha256"][:16])
    (out / "receipt.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
