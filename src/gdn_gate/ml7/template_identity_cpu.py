"""CPU check for the 4B record (ml_pipeline_4b, Gate A): Qwen3.5-2B vs 4B vs 9B prompts as rl_common.Thinker.prompt
renders them (chat template, enable_thinking=True, the boxed-answer suffix), on every item of the frozen ml7 data
(rl_pool, val, test_gsm8k, test_math500): token ids identical, and the ids Thinker uses (</think>, <think>,
<|im_end|>, <|endoftext|>, eos, the forced close) identical. Red arm: rendered without enable_thinking the 2B (non-
thinking by default) and the 4B (thinking by default) differ, so the check can see a template difference.
Tokenizer files only (no weights). usage: python template_identity_cpu.py [--out F]"""
import json, os, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
RUNS = os.environ.get("WG_RUNS", str(HERE.parents[2] / "runs"))
HF = os.environ.get("WG_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
sys.path.insert(0, str(HERE.parent / "broad"))
from transformers import AutoTokenizer
SUFFIX = None
src = open(HERE.parent / "broad/rl_common.py").read()
# pull SUFFIX without importing torch-heavy module
for line in src.splitlines():
    if line.startswith("SUFFIX"):
        exec(line); break
M = {"2B": f"{HF}/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc",
     "4B": f"{HF}/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
     "9B": f"{HF}/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
D = f"{RUNS}/ml7/data"
rows = [json.loads(x) for f in ("rl_pool", "val", "test_gsm8k", "test_math500") for x in open(f"{D}/{f}.jsonl") if x.strip()]
T = {k: AutoTokenizer.from_pretrained(v) for k, v in M.items()}
def render(tok, p, **kw):
    return tok.apply_chat_template([{"role": "user", "content": p + SUFFIX}], add_generation_prompt=True, tokenize=True, return_dict=False, **kw)
out = {"n_items": len(rows), "suffix_sha": hashlib.sha256(SUFFIX.encode()).hexdigest()[:16]}
ref = [render(T["2B"], r["problem"], enable_thinking=True) for r in rows]
for k in ("4B", "9B"):
    got = [render(T[k], r["problem"], enable_thinking=True) for r in rows]
    out[f"{k}_identical_thinking_prompts"] = sum(a == b for a, b in zip(ref, got))
    tk, t2 = T[k], T["2B"]
    out[f"{k}_special_ids_equal"] = all(tk.convert_tokens_to_ids(s) == t2.convert_tokens_to_ids(s) for s in ("</think>", "<think>", "<|im_end|>", "<|endoftext|>")) and tk.eos_token_id == t2.eos_token_id
    out[f"{k}_force_ids_equal"] = tk.encode("\n</think>\n\n", add_special_tokens=False) == t2.encode("\n</think>\n\n", add_special_tokens=False)
# red arm: default (no enable_thinking) differs between the 2B (non-thinking default) and the 4B (thinking default)
r0 = rows[0]["problem"]
out["red_default_2B_vs_4B_differ"] = render(T["2B"], r0) != render(T["4B"], r0)
out["red_default_4B_equals_thinking"] = render(T["4B"], r0) == render(T["4B"], r0, enable_thinking=True)
out["pass"] = (out["4B_identical_thinking_prompts"] == len(rows) and out["9B_identical_thinking_prompts"] == len(rows)
               and out["4B_special_ids_equal"] and out["9B_special_ids_equal"] and out["4B_force_ids_equal"]
               and out["9B_force_ids_equal"] and out["red_default_2B_vs_4B_differ"])
out["models"] = M
print(json.dumps(out, indent=1))
if "--out" in sys.argv:
    from pathlib import Path
    o = Path(sys.argv[sys.argv.index("--out") + 1]); o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1))
sys.exit(0 if out["pass"] else 1)
