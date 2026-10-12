#!/usr/bin/env python3
"""dose_ckpts.py OUT_DIR: G2 gate dose-response inputs. For wide1 seeds 0-2 (Qwen3.5-4B, step 150) write copies of the
checkpoint with every gate tensor (keys containing gate_w) multiplied by k in {0, 0.5, 1.5, 2}; the LoRA tensors are
copied unchanged. Writes OUT_DIR/receipt.json (source sha256, k, output sha256, per-file check that only gate tensors
changed and by exactly k)."""
import hashlib, json, sys
import os
from pathlib import Path

import torch

G = os.path.join(os.environ.get("WG_RUNS", "runs"), "ml7_4b/grpo")
KS = (0.0, 0.5, 1.5, 2.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rec = {}
    for s in (0, 1, 2):
        src = f"{G}/wide1_s{s}_cap512_r_box/ckpt_step150.pt"
        st = torch.load(src, map_location="cpu")
        for k in KS:
            new = {n: (v * k if "gate_w" in n else v.clone()) for n, v in st.items()}
            ng = sum("gate_w" in n for n in new)
            assert ng == 24, ng
            for n in st:
                if "gate_w" in n:
                    assert torch.equal(new[n], st[n] * k)
                else:
                    assert torch.equal(new[n], st[n])
            name = f"wide1_s{s}_x{k:g}.pt"
            torch.save(new, out / name)
            rec[name] = {"source": src, "source_sha256": sha(src), "k": k, "gate_tensors": ng, "sha256": sha(out / name)}
    (out / "receipt.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({n: v["sha256"][:12] for n, v in rec.items()}))


if __name__ == "__main__":
    main()
