#!/usr/bin/env python3
"""Export the paper's trained adapters to safetensors for the Hugging Face Hub, and convert them back for this code.

The training code saves each run's trainable state (PEFT LoRA tensors, plus the gate matrix W of the gated arms) as a
flat torch state dict, `ckpt_step<N>.pt` or `trainable.pt`. The lab's files name the gate tensor `<internal>_w`; this
release names it `gate_w`. `export` renames the key, writes `adapter.safetensors` plus `adapter_meta.json` per run
in the Hub layout below, and a SHA256SUMS file. `to-pt` turns a downloaded `adapter.safetensors` back into the
`.pt` file that `--ckpt` / `--init_from` / `--job` of the released scripts load.

Hub layout (one repository per training recipe and base model; one folder per arm and seed):
  <record>/<arm>/seed<k>/adapter.safetensors, adapter_meta.json
  records: qwen3.5-2b-grpo, qwen3.5-4b-grpo, qwen3.5-4b-base-chat-sft, qwen3.5-4b-base-forgetting
  arms (paper names): lora (receipt ctrl), lora-r20 (ctrlpm), gate01 (wide1), gate02 (wide), warmup

Usage:
  python scripts/adapters.py list   --runs DIR [--retrofit_runs DIR]          # what would be exported, with sizes
  python scripts/adapters.py export --runs DIR [--retrofit_runs DIR] --out DIR [--record NAME]
  python scripts/adapters.py to-pt  adapter.safetensors OUT.pt
Needs: torch, safetensors (the train environment, env/train.txt). CPU only.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PAPER_ARM = {"ctrl": "lora", "ctrlpm": "lora-r20", "wide1": "gate01", "wide": "gate02"}
GATE_CEIL = {"ctrl": None, "ctrlpm": None, "wide1": 1.0, "wide": 2.0}
RANK = {"ctrl": 16, "ctrlpm": 20, "wide1": 16, "wide": 16}
GATE_KEY = re.compile(r"(\.linear_attn\.)[a-z]+_w(\.weight)$")  # the gate matrix W; the lab used another prefix
BASE = {
    "qwen3.5-2b-grpo": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "qwen3.5-4b-grpo": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen3.5-4b-base-chat-sft": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
    "qwen3.5-4b-base-forgetting": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
}
RECIPE = {
    "qwen3.5-2b-grpo": "GRPO, correctness-gated length penalty alpha 0.2, 150 steps from the shared warm-up (paper Section 4)",
    "qwen3.5-4b-grpo": "GRPO, correctness-gated length penalty alpha 0.2, 150 steps from the shared warm-up (paper Section 4)",
    "qwen3.5-4b-base-chat-sft": "chat SFT on Tulu 3 (permissive subsets), 1,200 steps, lr 1e-4 (paper Section 7)",
    "qwen3.5-4b-base-forgetting": "LoRA on synthetic state tracking with WikiText replay, 1,200 steps, lr 1e-4 (paper Section 7)",
}


def catalog(runs: Path, retro: Path | None):
    """(record, arm, seed, step, path) for every adapter the paper reads, in the lab's run layout."""
    out = []
    for arm in ("ctrl", "wide1", "wide"):
        for s in range(6):
            tag = "gb10" if s < 3 else "box"
            out.append(("qwen3.5-2b-grpo", arm, s, 150, runs / f"ml7/grpo/{arm}_s{s}_cap512_r_{tag}/ckpt_step150.pt"))
    out.append(("qwen3.5-2b-grpo", "warmup", 0, 40, runs / "ml7/grpo/ctrl_s0_cap512_fmt_gb10/ckpt_step40.pt"))
    for arm in ("ctrl", "ctrlpm", "wide1", "wide"):
        for s in range(3):
            out.append(("qwen3.5-4b-grpo", arm, s, 150, runs / f"ml7_4b/grpo/{arm}_s{s}_cap512_r_box/ckpt_step150.pt"))
    out.append(("qwen3.5-4b-grpo", "warmup", 0, 40, runs / "ml7_4b/grpo/ctrl_s0_cap512_fmt_box/ckpt_step40.pt"))
    for arm in ("ctrl", "wide1", "wide"):
        for s in range(6):
            out.append(("qwen3.5-4b-base-chat-sft", arm, s, 1200, runs / f"broad/step6/{arm}_s{s}/ckpt_step1200.pt"))
    if retro is not None:
        for arm in ("ctrl", "wide"):
            for s in range(3):
                out.append(("qwen3.5-4b-base-forgetting", arm, s, 1200, retro / f"box/main_{arm}_s{s}/trainable.pt"))
    for s in range(3):
        out.append(("qwen3.5-4b-base-forgetting", "wide1", s, 1200, runs / f"w1main/main_wide1_s{s}_box2/trainable.pt"))
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_list(a):
    tot, miss = 0, 0
    for rec, arm, s, step, p in catalog(Path(a.runs), Path(a.retrofit_runs) if a.retrofit_runs else None):
        ok = p.exists()
        n = p.stat().st_size if ok else 0
        tot += n
        miss += not ok
        print(f"{rec:28s} {PAPER_ARM.get(arm, arm):9s} seed{s} step{step:<5d} {n / 1e6:8.1f} MB {'' if ok else 'MISSING ' + str(p)}")
    print(f"total {tot / 1e6:.1f} MB, missing {miss}")


def cmd_export(a):
    import torch
    from safetensors.torch import save_file

    out = Path(a.out)
    sums = {}
    for rec, arm, s, step, p in catalog(Path(a.runs), Path(a.retrofit_runs) if a.retrofit_runs else None):
        if a.record and rec != a.record:
            continue
        if not p.exists():
            print("skip (missing)", p, file=sys.stderr)
            continue
        state = torch.load(p, map_location="cpu", weights_only=True)
        renamed = {GATE_KEY.sub(r"\1gate_w\2", k): v.contiguous() for k, v in state.items()}
        train_arm = "ctrl" if arm == "warmup" else arm
        meta = {"record": rec, "arm": PAPER_ARM.get(arm, arm), "receipt_arm": arm, "seed": s, "step": step,
                "base_model": BASE[rec][0], "base_revision": BASE[rec][1], "recipe": RECIPE[rec],
                "lora_rank": RANK[train_arm], "lora_alpha": 2 * RANK[train_arm],
                "lora_targets": ["in_proj_qkv", "in_proj_a", "in_proj_b", "out_proj"],
                "gate": None if GATE_CEIL[train_arm] is None else
                {"tensor": "gate_w", "init": "zeros", "dtype": "float32", "beta_ceiling": GATE_CEIL[train_arm]},
                "n_tensors": len(renamed), "n_params": int(sum(v.numel() for v in renamed.values())),
                "source_sha256": sha256(p)}
        d = out / rec / PAPER_ARM.get(arm, arm) / f"seed{s}"
        d.mkdir(parents=True, exist_ok=True)
        save_file(renamed, str(d / "adapter.safetensors"), metadata={"record": rec, "arm": meta["arm"]})
        (d / "adapter_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        for f in ("adapter.safetensors", "adapter_meta.json"):
            sums.setdefault(rec, []).append(f"{sha256(d / f)}  {(d / f).relative_to(out / rec)}")
        print("wrote", d)
    for rec, lines in sums.items():
        (out / rec / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def cmd_to_pt(a):
    import torch
    from safetensors.torch import load_file

    torch.save(load_file(a.src), a.dst)
    print("wrote", a.dst)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "export"):
        p = sub.add_parser(name)
        p.add_argument("--runs", required=True, help="lab run root (WG_RUNS)")
        p.add_argument("--retrofit_runs", default=None, help="run root of the 4B forgetting runs (ctrl, wide)")
        if name == "export":
            p.add_argument("--out", required=True)
            p.add_argument("--record", default=None, choices=sorted(BASE))
    p = sub.add_parser("to-pt")
    p.add_argument("src")
    p.add_argument("dst")
    a = ap.parse_args()
    {"list": cmd_list, "export": cmd_export, "to-pt": cmd_to_pt}[a.cmd](a)


if __name__ == "__main__":
    main()
