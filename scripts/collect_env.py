#!/usr/bin/env python3
"""Collect the env block of every released readout (results/**/readout_*.json.gz) into results/ENVIRONMENT.json.

Each readout records the engine and library versions, the GPU model, the model revision and the sha256 of the code
and data it ran with. This script lists them per readout and summarizes the distinct values per record, so the
versions and hardware the paper states can be checked without opening each file. CPU only, stdlib only.
Usage: python scripts/collect_env.py [--check]   (--check: exit 1 if the file on disk differs from a fresh build)
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ENVIRONMENT.json"
KEYS = ("vllm", "torch", "transformers", "math_verify", "python", "gpu", "model_revision")


def build() -> dict:
    per, summary = {}, {}
    for p in sorted(ROOT.glob("results/**/readout_*.json.gz")):
        env = json.loads(gzip.open(p).read())["env"]
        rel = str(p.relative_to(ROOT))
        per[rel] = {k: env.get(k) for k in KEYS} | {"code_sha256": env.get("sha256", {})}
        rec = str(p.parent.relative_to(ROOT))
        s = summary.setdefault(rec, {k: [] for k in KEYS})
        for k in KEYS:
            if env.get(k) not in s[k]:
                s[k].append(env.get(k))
    return {"note": "env blocks of the released readouts; code_sha256 values refer to the lab files before the "
                    "publication scrub (see results/README.md)", "by_record": summary, "readouts": per}


def main():
    new = json.dumps(build(), indent=1) + "\n"
    if "--check" in sys.argv:
        ok = OUT.exists() and OUT.read_text() == new
        print("ENVIRONMENT.json up to date" if ok else "ENVIRONMENT.json differs from a fresh build")
        sys.exit(0 if ok else 1)
    OUT.write_text(new)
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
