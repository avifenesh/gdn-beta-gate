"""Decode the paired example traces of the implementation appendix and render the prompt template (CPU only).

GSM8K test item 0 (the first item, chosen by position), sample 0, Qwen3.5-4B seed 0, LoRA (ctrl) and Gate(0,1)
(wide1), from the step-14 test readouts. Token ids come from the readout trace files; the tokenizer and chat template
are the pinned Qwen3.5-4B revision 851bf6e8. Writes paper/data/example_traces.txt (plain text, so ledger text rows can
quote it). Run: uv run --with tokenizers,jinja2 python3 paper/figs/example_traces.py
The trace files (*.traces.jsonl.gz) are not released, so the script runs on the lab tree only; its output is released.
"""
import gzip
import os
import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
SNAP = Path(os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))) / "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
READ = ROOT / "results/ml7_4b/step14"
DATA = Path(os.environ.get("WG_RUNS", "runs")) / "ml7/data/test_gsm8k.jsonl"
OUT = ROOT / "paper/data/example_traces.txt"
ITEM, SAMPLE = "gsm8k_test:0", 0
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."  # rl_common.SUFFIX


def render(content: str) -> str:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tpl = env.from_string((SNAP / "chat_template.jinja").read_text())
    return tpl.render(messages=[{"role": "user", "content": content}], add_generation_prompt=True,
                      enable_thinking=True, bos_token="", eos_token="<|im_end|>")


def main():
    tok = Tokenizer.from_file(str(SNAP / "tokenizer.json"))
    item = next(json.loads(x) for x in open(DATA) if json.loads(x)["id"] == ITEM)
    out = ["# prompt template (rendered, {problem} = the problem text)", render("{problem}" + SUFFIX), "",
           f"# item {ITEM}, sample {SAMPLE}", "problem: " + item["problem"], "gold: " + item["answer"], ""]
    for arm in ("ctrl", "wide1"):
        rows = json.load(open(READ / f"readout_gsm8k_{arm}_0.json"))["rows"]
        row = next(r for r in rows if r["id"] == ITEM and r["sample"] == SAMPLE)
        tr = next(json.loads(x) for x in gzip.open(READ / f"readout_gsm8k_{arm}_0.json.traces.jsonl.gz", "rt")
                  if json.loads(x)["id"] == ITEM and json.loads(x)["sample"] == SAMPLE)
        assert len(tr["think"]) == row["think_len"]
        out += [f"# arm {arm} seed 0: reasoning tokens {row['think_len']}, closed {row['closed']}, "
                f"correct at 8192 {row['caps']['8192'][0]}", "## thinking", tok.decode(tr["think"]),
                "## answer at the 8192 budget", tr["answers"]["8192"], ""]
    OUT.write_text("\n".join(out))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
