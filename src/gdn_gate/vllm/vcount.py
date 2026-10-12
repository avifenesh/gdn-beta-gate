#!/usr/bin/env python3
"""Counting probe for the Qwen3.5 retrofit checkpoints (claim C3) on vLLM: lm-eval RULER ruler_cwe (common-word
frequency) and ruler_fwe (frequent coded words), the two running-tally tasks. On Olmo-Hybrid-7B, clamping beta at 1
cost CWE 8k 0.568 -> 0.476 (p 0.0008) while the lm battery did not move (FINDINGS 2026-10-09 10:40); this asks whether
a range retrofitted onto a model pretrained without it (Qwen3.5, arms wide / wide1 vs ctrl and the base) shows the
same counting signal.

One vLLM engine per arm type, as vscore_4b.py: the process installs the gate plugin for its engine
(gate_vllm.install), builds one engine through lm-eval's VLLM class, and scores each checkpoint of that type in turn
by writing its merged LoRA and gates in place (gate_vllm.set_checkpoint; bf16 merge, delta_kept recorded and
gated). Engines: wide (wide arm), wide1 (wide1 arm), stock (ctrl arm, strip twins with gates zeroed, the base).

Tasks are synthetic and offline: lm-eval builds them from the tokenizer (metadata tokenizer + max_seq_lengths), 500
items per length; --limit takes the first N (default 500, all). Greedy: the tasks ask for do_sample false and
temperature 0, and greedy_fix() (vbattery_olmo._greedy_fix's arithmetic, plus a counter) forces temperature 0 for any
request that did not ask for sampling, against the lm-eval 0.4.13 vLLM default of 1.0; the receipt carries the count
of requests that still asked for sampling (must be 0). Raw prompts, no chat template (as the Olmo probe).

Layout (what olmo/ruler_paired.py reads, with --base for the reference job):
  <out>/<job>/samples_ruler<L>/<task>.jsonl   doc_id, target, filtered_resps, resps, doc_hash, prompt_hash and the
                                               per-length scores under str(L) (lm-eval's process_results keys)
  <out>/<job>/lmeval_ruler<L>.json            results, n-samples, versions, plus vcount {tasks, limit, digest}
  <out>/<job>/receipt.json                    versions, argv, host, gpu, checkpoint sha256, set_checkpoint receipt,
                                               layout check, data digests, timing per length
A (job, length) whose lmeval file holds every asked task at the same limit is skipped (resumable).

Jobs: step 6 names resolved under --root (<root>/<arm>_s<seed>/ckpt_step<step>.pt):
  <arm>_s<seed>[_strip][@<step>]   arm in wide | wide1 | ctrl; _strip = a gated arm's LoRA with its gates zeroed
                                   (stock engine); @step overrides --step; job dir <name> with @S as _stepS
  base                             the untouched model (stock engine)
or explicit --job NAME=CKPT[:strip|:base] (vscore_4b.py's form, e.g. the retrofit box adapters). The engine of each
job is inferred from its name (ctrl / strip / base -> stock, wide1 -> wide1, wide -> wide) and must equal --engine;
a name with no arm token runs on --engine as given. --plan prints "ENGINE name,name" lines for a runner.

Usage:
  vcount.py --engine wide --root R --names wide_s0,wide_s1 --model M --out O [--lengths 4096,8192,16384] [--limit 500]
  vcount.py --engine stock --job main_ctrl_s0=/path/trainable.pt --model M --out O
  vcount.py --plan --root R --names wide_s0,wide1_s0,ctrl_s0,wide_s0_strip,base
  vcount.py --data_only --model M --out O --lengths 4096   (CPU: build the task data, write its digests, no engine)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_GDN_DECODE_KERNEL", "triton")
HERE = Path(__file__).resolve().parent

TASKS = ("ruler_cwe", "ruler_fwe")
ENGINES = ("wide", "wide1", "stock")
ARM_ENGINE = {"wide": "wide", "wide1": "wide1", "ctrl": "stock"}
STEP6_NAME = re.compile(r"^(?P<arm>wide1|wide|ctrl)_s(?P<seed>\d+)(?P<strip>_strip)?(?:@(?P<step>\d+))?$")
DELTA_KEPT_MIN = 0.999  # gate_9b.py's bar for the bf16 merge
SAMPLE_KEYS = ("doc_id", "target", "filtered_resps", "resps", "doc_hash", "prompt_hash")


class VcountError(RuntimeError):
    pass


@dataclass
class Job:
    name: str          # output directory name
    ckpt: str | None   # None for the base model
    mode: str          # trained | strip | base
    engine: str        # wide | wide1 | stock
    arm: str | None = None
    seed: int | None = None
    step: int | None = None


# ------------------------------------------------------------------------------------------------ job resolution
def infer_engine(name: str) -> str | None:
    """Engine implied by a job name's tokens; None when the name carries no arm token."""
    toks = set(re.split(r"[_@]", name))
    if "strip" in toks or "ctrl" in toks or name == "base":
        return "stock"
    if "wide1" in toks:
        return "wide1"
    if "wide" in toks:
        return "wide"
    return None


def resolve_name(name: str, root: str | None, step: int) -> Job:
    if name == "base":
        return Job("base", None, "base", "stock")
    m = STEP6_NAME.match(name)
    if not m:
        raise VcountError(f"cannot resolve {name!r}: expected <wide|wide1|ctrl>_s<seed>[_strip][@step] or base")
    if root is None:
        raise VcountError(f"{name}: --root is required for step 6 names")
    arm, seed, strip = m.group("arm"), int(m.group("seed")), bool(m.group("strip"))
    st = int(m.group("step")) if m.group("step") else step
    if strip and arm == "ctrl":
        raise VcountError(f"{name}: ctrl has no gates to strip")
    ckpt = Path(root) / f"{arm}_s{seed}" / f"ckpt_step{st}.pt"
    if not ckpt.is_file():
        raise VcountError(f"{name}: no checkpoint {ckpt}")
    out = name.replace("@", "_step")
    return Job(out, str(ckpt), "strip" if strip else "trained", "stock" if strip else ARM_ENGINE[arm], arm, seed, st)


def parse_job(spec: str, engine: str | None) -> Job:
    """vscore_4b.py's NAME=CKPT[:strip|:base]."""
    if "=" not in spec:
        raise VcountError(f"--job {spec!r}: expected NAME=CKPT[:strip|:base]")
    name, rest = spec.split("=", 1)
    ckpt, _, mode = rest.partition(":")
    if mode not in ("", "strip", "base"):
        raise VcountError(f"--job {spec!r}: unknown mode {mode!r}")
    mode = mode or "trained"
    if mode != "base" and not Path(ckpt).is_file():
        raise VcountError(f"--job {spec!r}: no checkpoint {ckpt}")
    inferred = "stock" if mode in ("strip", "base") else infer_engine(name)
    eng = inferred or engine
    if eng is None:
        raise VcountError(f"--job {spec!r}: the name implies no engine; pass --engine")
    return Job(name, None if mode == "base" else ckpt, mode, eng)


def resolve_jobs(a) -> list[Job]:
    jobs = [resolve_name(n, a.root, a.step) for n in (a.names.split(",") if a.names else []) if n]
    jobs += [parse_job(s, a.engine) for s in a.job or []]
    if not jobs:
        raise VcountError("no jobs: pass --names and/or --job")
    seen = set()
    for j in jobs:
        if j.name in seen:
            raise VcountError(f"duplicate job name {j.name}")
        seen.add(j.name)
    return jobs


def plan(jobs: list[Job]) -> dict:
    out: dict = {}
    for j in jobs:
        out.setdefault(j.engine, []).append(j.name)
    return {e: out[e] for e in ENGINES if e in out}


def check_state_keys(keys, job: Job, decay: bool = False) -> dict:
    """The checkpoint's tensors must fit the job: gates present for a trained gated arm, absent (or stripped) on the
    stock engine, and LoRA tensors present for every non-base job. Catches a wide checkpoint scored as ctrl. With
    --decay (wideg / ctrlg) only the LoRA and the decay gate are checked here; set_checkpoint checks the rest."""
    keys = list(keys)
    n_gate = sum(1 for k in keys if re.search(r"linear_attn\.gate_w(?:\.weight)?$", k))
    n_decay = sum(1 for k in keys if re.search(r"linear_attn\.wg_wa(?:\.weight)?$", k))
    n_lora = sum(1 for k in keys if ".lora_A" in k or ".lora_B" in k)
    if n_lora == 0:
        raise VcountError(f"{job.name}: checkpoint has no LoRA tensors")
    if decay:
        if not n_decay:
            raise VcountError(f"{job.name}: --decay but the checkpoint has no wg_wa")
        return {"gate_tensors": n_gate, "decay_tensors": n_decay, "lora_tensors": n_lora}
    if job.mode == "trained" and job.engine == "stock" and n_gate:
        raise VcountError(f"{job.name}: checkpoint carries {n_gate} gates but the job is a stock-engine trained run "
                          "(name it _strip to score the LoRA alone)")
    if job.mode == "trained" and job.engine != "stock" and not n_gate:
        raise VcountError(f"{job.name}: gated engine {job.engine} but the checkpoint has no gate_w")
    if job.mode == "strip" and not n_gate:
        raise VcountError(f"{job.name}: strip asked but the checkpoint has no gates")
    return {"gate_tensors": n_gate, "decay_tensors": n_decay, "lora_tensors": n_lora}


# ------------------------------------------------------------------------------------------------ samples and files
def sample_row(s: dict) -> dict:
    """olmo/battery.py's per-sample fields plus doc_hash and prompt_hash (cross-job item identity)."""
    row = {k: s.get(k) for k in SAMPLE_KEYS if k in s}
    row.update({m: s[m] for m in s if m.isdigit()})
    return row


def write_samples(sdir: Path, samples: dict) -> dict:
    """Write <sdir>/<task>.jsonl; return {task: digest} over (doc_id, doc_hash)."""
    sdir.mkdir(parents=True, exist_ok=True)
    dig = {}
    for t, rows in samples.items():
        rows = [sample_row(s) for s in rows]
        with open(sdir / f"{t}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        dig[t] = digest_rows(rows)
    return dig


def digest_rows(rows) -> str:
    h = hashlib.sha256()
    for doc_id, dh in sorted((int(r["doc_id"]), str(r.get("doc_hash"))) for r in rows):
        h.update(f"{doc_id}:{dh}\n".encode())
    return h.hexdigest()[:16]


def length_done(jdir: Path, L: int, tasks, limit: int) -> bool:
    p = jdir / f"lmeval_ruler{L}.json"
    if not p.is_file():
        return False
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        return False
    v = d.get("vcount", {})
    if v.get("limit") != limit or not set(tasks) <= set(d.get("results", {})):
        return False
    return all((jdir / f"samples_ruler{L}" / f"{t}.jsonl").is_file() for t in tasks)


def sha256_file(p, n=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(n):
            h.update(b)
    return h.hexdigest()


def model_files_sha(model: str) -> dict:
    out = {}
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        p = Path(model) / f
        if p.is_file():
            out[f] = sha256_file(p)[:16]
    return out


def save_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ lm-eval
GREEDY = {"requests": 0, "sampled": 0}
CWE_MODULE = "lm_eval.tasks.ruler.cwe_utils"


def reset_ruler_state() -> bool:
    """lm-eval 0.4.13's cwe_utils keeps a module-level RNG = random.Random(42) (shuffles WORDS at import, then every
    word list in get_example) and the yaml loader caches the module in sys.modules for the life of the process, so
    a second ruler_cwe build in one process gets other contexts (same targets: those come from the reseeded global
    random) under the same doc_ids. Measured 2026-10-09 on the 4B tokenizer: the 8k digest built after 4k in one
    process differs from 8k built first. Every build here must equal a fresh-process build, so the RNG is put back to
    its import-time state (Random(42) after the WORDS shuffle; the rebuilt WORDS must equal the module's). Returns
    False when the module is not loaded yet (the next load is fresh)."""
    import random

    m = sys.modules.get(CWE_MODULE)
    if m is None:
        return False
    rng = random.Random(42)
    words = sorted({w for c in ["noun", "adjective", "verb"] for w in m.r._categories[c]})
    rng.shuffle(words)
    if words != m.WORDS:
        raise VcountError("cwe_utils.WORDS differs from a fresh import: cannot reset its RNG")
    m.RNG = rng
    return True


def check_digests(out: Path, L: int, digest: dict) -> None:
    """Every job under <out> must have been scored on the same items: compare with the digests already written."""
    for p in out.glob(f"*/lmeval_ruler{L}.json"):
        try:
            other = json.load(open(p)).get("vcount", {})
        except (OSError, ValueError):
            continue
        for t, d in digest.items():
            od = other.get("digest", {}).get(t)
            if od is not None and other.get("limit") is not None and od != d:
                raise VcountError(f"{t}@{L}: items differ from {p} ({d} vs {od}); pairing would be invalid")


def greedy_fix(VLLM) -> None:
    """vbattery_olmo._greedy_fix's arithmetic: lm-eval 0.4.13's VLLM drops do_sample=False without a temperature and
    vLLM's default temperature is 1.0, so restore temperature 0 whenever the request did not ask for sampling. Counts
    requests and the ones that asked for sampling (the receipt requires 0 for RULER)."""
    if getattr(VLLM.modify_gen_kwargs, "_vcount", False):
        return
    orig = VLLM.modify_gen_kwargs

    def modify(gen_kwargs, eos=None, default_max_gen_toks=256):
        sample = bool(gen_kwargs.get("do_sample", False)) or float(gen_kwargs.get("temperature", 0.0) or 0.0) > 0.0
        kw, until, mx = orig(gen_kwargs, eos=eos, default_max_gen_toks=default_max_gen_toks)
        GREEDY["requests"] += 1
        if sample:
            GREEDY["sampled"] += 1
        else:
            kw["temperature"] = 0.0
        return kw, until, mx

    modify._vcount = True
    VLLM.modify_gen_kwargs = staticmethod(modify)


def run_ruler(lm, tasks, jdir: Path, L: int, limit: int, tokenizer: str) -> dict:
    from lm_eval import simple_evaluate

    reset_ruler_state()
    res = simple_evaluate(model=lm, tasks=list(tasks), limit=limit or None, log_samples=True,
                          metadata={"tokenizer": tokenizer, "max_seq_lengths": [L]}, random_seed=0,
                          numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)
    dig = write_samples(jdir / f"samples_ruler{L}", res.get("samples", {}))
    keep = {"results": res["results"], "n-samples": res.get("n-samples"), "configs_versions": res.get("versions"),
            "vcount": {"tasks": list(tasks), "limit": limit, "length": L, "digest": dig}}
    save_json(keep, jdir / f"lmeval_ruler{L}.json")
    return keep


def data_only(a) -> int:
    """Build the task data exactly as simple_evaluate does (same seeds, TaskManager metadata) and write per-task
    digests over the first --limit docs, with lm-eval's own doc_hash; no engine, no GPU."""
    import random

    import numpy as np
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable, hash_string

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"model": a.model, "model_files_sha": model_files_sha(a.model), "argv": sys.argv, "host": platform.node(),
           "versions": versions(), "lengths": {}}
    for L in a.lengths_list:
        t0 = time.time()
        reset_ruler_state()
        random.seed(0)
        np.random.seed(1234)
        loaded = TaskManager(metadata={"tokenizer": a.model, "max_seq_lengths": [L]}).load(list(a.tasks_list))
        per = {}
        for t in a.tasks_list:
            docs = list(loaded["tasks"][t].eval_docs)[: a.limit or None]
            rows = [{"doc_id": i, "doc_hash": hash_string(json.dumps(d, indent=2, default=handle_non_serializable,
                                                                     ensure_ascii=False))} for i, d in enumerate(docs)]
            per[t] = {"n": len(rows), "digest": digest_rows(rows)}
        rec["lengths"][str(L)] = {"tasks": per, "seconds": round(time.time() - t0, 1)}
        save_json(rec, out / "data_only.json")
        print("DATA", L, json.dumps(rec["lengths"][str(L)]), flush=True)
    return 0


def versions() -> dict:
    from importlib import metadata as md

    v = {}
    for p in ("vllm", "torch", "lm_eval", "transformers", "triton", "wonderwords", "numpy"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    return v


def code_sha() -> dict:
    out = {}
    for f in ("vcount.py", "gate_vllm.py"):
        p = HERE / f
        if p.is_file():
            out[f] = sha256_file(p)[:16]
    return out


# ------------------------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", choices=ENGINES)
    ap.add_argument("--decay", action="store_true", help="retrofit decay-gate arms (wideg / ctrlg), as vscore_4b.py")
    ap.add_argument("--root", help="step 6 run root (<root>/<arm>_s<seed>/ckpt_step<step>.pt)")
    ap.add_argument("--names", default="", help="comma list: <wide|wide1|ctrl>_s<seed>[_strip][@step] | base")
    ap.add_argument("--step", type=int, default=1200, help="checkpoint step for --names without @step")
    ap.add_argument("--job", action="append", help="NAME=CKPT[:strip|:base] (vscore_4b.py form)")
    ap.add_argument("--model", help="base model dir (tokenizer for the task data, weights for the engine)")
    ap.add_argument("--out")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--lengths", default="4096,8192,16384")
    ap.add_argument("--limit", type=int, default=500, help="first N items per task and length (500 = all)")
    ap.add_argument("--gpu_mem", type=float, default=0.80)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--max_num_batched_tokens", type=int, default=8192)
    ap.add_argument("--max_model_len", type=int, default=0, help="0 = max(lengths) + 1024")
    ap.add_argument("--min_delta_kept", type=float, default=DELTA_KEPT_MIN)
    ap.add_argument("--plan", action="store_true", help="print 'ENGINE name,name' per engine and exit")
    ap.add_argument("--data_only", action="store_true", help="CPU: build the task data and write digests only")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    a.tasks_list = [t for t in a.tasks.split(",") if t]
    a.lengths_list = [int(x) for x in a.lengths.split(",") if x]
    if not a.tasks_list or not a.lengths_list:
        ap.error("--tasks and --lengths must be non-empty")
    if a.limit < 0:
        ap.error("--limit must be >= 0")
    a.max_model_len = a.max_model_len or max(a.lengths_list) + 1024
    if a.max_model_len < max(a.lengths_list):
        ap.error("--max_model_len below the longest length")
    if a.plan:
        return a
    if not a.model or not a.out:
        ap.error("--model and --out are required")
    if not a.data_only and not a.engine:
        ap.error("--engine is required (one engine per process; see --plan)")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.data_only:
        return data_only(a)
    jobs = resolve_jobs(a)
    if a.plan:
        for e, names in plan(jobs).items():
            print(e, ",".join(names))
        return 0
    wrong = [j.name for j in jobs if j.engine != a.engine]
    if wrong:
        raise VcountError(f"jobs {wrong} need another engine than {a.engine} (see --plan)")

    sys.path.insert(0, str(HERE))
    import gate_vllm as NV  # imports torch; the engine host only

    NV.install(a.engine, decay=a.decay)
    import torch
    from lm_eval.models.vllm_causallms import VLLM

    greedy_fix(VLLM)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gated = a.engine != "stock" or a.decay
    ek = NV.engine_kwargs(gated, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs,
                          max_num_batched_tokens=a.max_num_batched_tokens)
    dtype, seed = ek.pop("dtype"), ek.pop("seed")
    ek.pop("max_model_len", None)
    t0 = time.time()
    lm = VLLM(pretrained=a.model, dtype=dtype, seed=seed, max_model_len=a.max_model_len, batch_size="auto", **ek)
    t_engine = round(time.time() - t0, 1)
    llm = lm.model
    model = NV.get_model(llm)
    base = NV.BaseWeights(a.model, sorted(NV.gdn_layers(model)))
    layout = NV.check_layout(model, base)
    common = {"engine": a.engine, "decay": a.decay, "model": a.model, "model_files_sha": model_files_sha(a.model),
              "tasks": a.tasks_list, "limit": a.limit, "max_model_len": a.max_model_len, "engine_kwargs": ek,
              "host": platform.node(), "gpu": torch.cuda.get_device_name(0), "versions": versions(),
              "code_sha256": code_sha(), "argv": sys.argv, "layout": layout,
              "git_describe": subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty"],
                                             capture_output=True, text=True).stdout.strip()}
    for job in jobs:
        jdir = out / job.name
        jdir.mkdir(parents=True, exist_ok=True)
        rp = jdir / "receipt.json"
        rec = json.load(open(rp)) if rp.exists() else {}
        todo = [L for L in a.lengths_list if not length_done(jdir, L, a.tasks_list, a.limit)]
        if not todo:
            print("SKIP", job.name, "all lengths done", flush=True)
            continue
        t1 = time.time()
        state = None
        if job.mode != "base":
            state = torch.load(job.ckpt, map_location="cpu", weights_only=False)
            keys = check_state_keys(NV._flat(state).keys(), job, a.decay)
        else:
            keys = {}
        sc = NV.set_checkpoint(model, base, state, gate=(job.mode == "trained" and a.engine != "stock"),
                               decay=a.decay)
        if job.mode != "base":
            if sc["lora_tensors_merged"] == 0:
                raise VcountError(f"{job.name}: no LoRA tensor was merged")
            if sc["delta_kept"] < a.min_delta_kept:
                raise VcountError(f"{job.name}: bf16 merge kept {sc['delta_kept']:.5f} of the delta (< {a.min_delta_kept})")
        if job.mode == "trained" and a.engine != "stock" and sc["gate_layers"] != layout["gdn_layers"]:
            raise VcountError(f"{job.name}: {sc['gate_layers']} gate layers written, engine has {layout['gdn_layers']}")
        llm.reset_prefix_cache()
        del state
        rec.update(common)
        rec.update({"job": asdict(job), "ckpt_sha256": sha256_file(job.ckpt) if job.ckpt else None,
                    "ckpt_keys": keys, "set_checkpoint": sc})
        tm = rec.setdefault("timing_s", {})
        tm["engine_build_shared"] = t_engine
        tm["set_checkpoint"] = round(time.time() - t1, 1)
        dig = rec.setdefault("data_digest", {})
        for L in todo:
            t2 = time.time()
            g0 = dict(GREEDY)
            r = run_ruler(lm, a.tasks_list, jdir, L, a.limit, a.model)
            try:
                check_digests(out, L, r["vcount"]["digest"])
            except VcountError:
                bad = jdir / f"lmeval_ruler{L}.json"
                os.replace(bad, bad.with_suffix(".digest_mismatch"))  # never counted as done
                raise
            tm[f"ruler{L}"] = round(time.time() - t2, 1)
            for t, d in r["vcount"]["digest"].items():
                dig[f"{t}@{L}"] = d
            rec.setdefault("greedy", {})[f"ruler{L}"] = {"requests": GREEDY["requests"] - g0["requests"],
                                                          "sampled": GREEDY["sampled"] - g0["sampled"]}
            save_json(rec, rp)
            scores = {t: r["results"].get(t, {}).get(f"{L},none") for t in a.tasks_list}
            print("RULER", job.name, L, json.dumps(scores), tm[f"ruler{L}"], "s", flush=True)
            if GREEDY["sampled"] != g0["sampled"]:
                raise VcountError(f"{job.name} L={L}: {GREEDY['sampled'] - g0['sampled']} requests asked for sampling")
        print("DONE", job.name, round(time.time() - t1), "s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VcountError as e:
        print("vcount:", e, file=sys.stderr)
        sys.exit(2)
