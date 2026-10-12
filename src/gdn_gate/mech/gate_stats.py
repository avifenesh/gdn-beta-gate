"""Per-layer, per-head write-strength (beta) statistics of 4B GRPO checkpoints under teacher forcing (HF forward,
the trainer's own model build: arms.load_model + arms.add_lora, transformers 5.17 GDN forward patched by patch.py).

For every GDN layer and value head, over the reasoning tokens of the input traces (prompt tokens excluded):
  s = sigmoid(b)            the write strength the (LoRA-adapted) projections produce, i.e. beta without the gate
  t = tanh(W x)             the gate (0 for ctrl/ctrlpm/warm-up: they are loaded on the wide1 forward with W = 0,
                            which is the library forward exactly, patch.py G0)
  beta                      the write strength the delta rule uses (= s when W = 0)
  g                         log-decay of the state, -exp(A_log) * softplus(a + dt_bias), from the in_proj_a output
Accumulates means, histograms of s and beta, beta - s by s-bin and by relative position in the trace, and beta - s
summed per token id. Writes one JSON per (checkpoint, input set).

Usage: gate_stats.py --model DIR --inputs inputs.json --out_dir D --job name:arm:rank:ckpt:set[,set...] [--job ...]
  arm = the arm whose forward to install (wide | wide1; ctrl-type checkpoints use wide1 with W = 0)."""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "broad"))
import torch
import torch.nn.functional as F

import arms  # noqa: E402
import patch  # noqa: E402

NB_HIST, NB_S, NB_POS = 40, 20, 10


class Acc:
    def __init__(self, L, H, V, dev):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)  # noqa: E731
        self.cnt = z(L, H); self.s = z(L, H); self.beta = z(L, H); self.t = z(L, H); self.tabs = z(L, H)
        self.tpos = z(L, H); self.bgt1 = z(L, H); self.d = z(L, H); self.g = z(L, H); self.dec = z(L, H)
        self.s_lo = z(L, H)  # share of tokens with s < 0.1
        self.hist_s = z(L, NB_HIST); self.hist_b = z(L, NB_HIST)
        self.d_sbin = z(L, NB_S); self.n_sbin = z(L, NB_S)
        self.d_pos = z(L, NB_POS); self.n_pos = z(L, NB_POS)
        self.tok_d = z(L, V); self.tok_n = z(V)


def run_job(model_dir, inputs, job, out_dir, dev, limit=0, maxt=0, dtype=torch.bfloat16):
    name, arm, rank, ckpt, sets = job
    t0 = time.time()
    torch.manual_seed(0)
    model, _ = arms.load_model(model_dir, arm, device_map={"": dev.index if dev.type == "cuda" else "cpu"}, dtype=dtype)
    model = arms.add_lora(model, int(rank)).eval()
    state = torch.load(ckpt, map_location="cpu") if ckpt != "none" else {}
    canon = lambda k: k[k.find("layers."):] if "layers." in k else k  # noqa: E731
    by = {canon(k): v for k, v in state.items()}
    params = dict(model.named_parameters())
    loaded, gate_loaded = 0, 0
    with torch.no_grad():
        for n, p in params.items():
            if not p.requires_grad:
                continue
            c = canon(n)
            if c in by:
                p.copy_(by[c].to(p.dtype)); loaded += 1
                gate_loaded += "gate_w" in n
            elif "gate_w" in n:
                p.zero_()  # ctrl-type checkpoint on the gated forward: W = 0 gives the library forward
            else:
                raise SystemExit(f"{name}: trainable {n} absent from {ckpt}")
    unused = [k for k in by if not any(canon(n) == k for n in params)]
    assert not unused, unused[:3]
    layers = patch.gdn_layers(model)
    L, H = len(layers), layers[0].num_v_heads
    V = model.get_input_embeddings().weight.shape[0]
    cur = {"P": 0, "pos": None, "ids": None, "acc": None}

    def make_beta(li, layer, orig):
        def f(b, x):
            beta = orig(b, x)
            A = cur["acc"]
            if A is None:
                return beta
            P = cur["P"]
            with torch.no_grad():
                s = b[0, P:].float().sigmoid()
                t = torch.tanh(layer.gate_w(x[0, P:].to(layer.gate_w.weight.dtype))).float()
                bt = beta[0, P:].float()
                d = bt - s
                A.cnt[li] += s.shape[0]; A.s[li] += s.sum(0); A.beta[li] += bt.sum(0); A.t[li] += t.sum(0)
                A.tabs[li] += t.abs().sum(0); A.tpos[li] += (t > 0).double().sum(0); A.bgt1[li] += (bt > 1).double().sum(0)
                A.d[li] += d.sum(0); A.s_lo[li] += (s < 0.1).double().sum(0)
                A.hist_s[li] += torch.histc(s, NB_HIST, 0, 2).double(); A.hist_b[li] += torch.histc(bt, NB_HIST, 0, 2).double()
                sb = (s * NB_S).long().clamp(0, NB_S - 1).flatten()
                A.d_sbin[li].index_add_(0, sb, d.flatten().double()); A.n_sbin[li].index_add_(0, sb, torch.ones_like(sb, dtype=torch.float64))
                pb = cur["pos"]
                dm = d.mean(1).double()
                A.d_pos[li].index_add_(0, pb, dm); A.n_pos[li].index_add_(0, pb, torch.ones_like(dm))
                A.tok_d[li].index_add_(0, cur["ids"], dm)
            return beta
        return f

    def make_a_hook(li, layer):
        def h(mod, inp, out):
            A = cur["acc"]
            if A is None:
                return
            P = cur["P"]
            with torch.no_grad():
                a = out[0, P:].float()
                g = -layer.A_log.float().exp() * F.softplus(a + layer.dt_bias.float())
                A.g[li] += g.sum(0).double(); A.dec[li] += g.exp().sum(0).double()
        return h

    hooks = []
    for li, layer in enumerate(layers):
        layer._wg_beta = make_beta(li, layer, layer._wg_beta)
        hooks.append(layer.in_proj_a.register_forward_hook(make_a_hook(li, layer)))
    dec, _ = arms.core(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    for set_name in sets.split("+"):
        A = Acc(L, H, V, dev)
        cur["acc"] = A
        n_tok = 0
        exs = inputs["sets"][set_name][: limit or None]
        for ex in exs:
            think = ex["think"][: maxt or None]
            ids = torch.tensor([ex["prompt"] + think], device=dev)
            P = len(ex["prompt"]); T = len(think)
            cur["P"] = P
            cur["pos"] = (torch.arange(T, device=dev) * NB_POS // max(1, T)).clamp(0, NB_POS - 1)
            cur["ids"] = ids[0, P:]
            A.tok_n.index_add_(0, cur["ids"], torch.ones(T, dtype=torch.float64, device=dev))
            with torch.no_grad():
                dec(input_ids=ids)
            n_tok += T
        cur["acc"] = None
        c = A.cnt.clamp(min=1)
        top = {}
        tn = A.tok_n.clamp(min=1)
        for li in range(L):
            md = A.tok_d[li] / tn
            ok = A.tok_n >= 20
            vals = torch.where(ok, md, torch.zeros_like(md))
            hi = torch.topk(vals, 15).indices.tolist(); lo = torch.topk(-vals, 15).indices.tolist()
            top[li] = {"up": [(i, float(md[i]), int(A.tok_n[i])) for i in hi], "down": [(i, float(md[i]), int(A.tok_n[i])) for i in lo]}
        res = {"name": name, "arm_forward": arm, "rank": int(rank), "ckpt": ckpt, "set": set_name, "n_seq": len(exs), "maxt": maxt, "dtype": str(dtype), "device": str(dev),
               "n_tokens": n_tok, "L": L, "H": H, "loaded_tensors": loaded, "gate_tensors_loaded": gate_loaded,
               "mean": {k: (getattr(A, k) / c).tolist() for k in ("s", "beta", "t", "tabs", "tpos", "bgt1", "d", "g", "dec", "s_lo")},
               "hist_s": A.hist_s.tolist(), "hist_beta": A.hist_b.tolist(),
               "d_by_sbin": (A.d_sbin / A.n_sbin.clamp(min=1)).tolist(), "n_by_sbin": A.n_sbin.tolist(),
               "d_by_relpos": (A.d_pos / A.n_pos.clamp(min=1)).tolist(),
               "token_delta_top": top, "wall_s": time.time() - t0}
        (out_dir / f"{name}__{set_name}.json").write_text(json.dumps(res))
        print(f"{name} {set_name}: {n_tok} tokens, mean beta-s {float((A.d.sum() / A.cnt.sum())):+.4f}, "
              f"|t| {float(A.tabs.sum() / A.cnt.sum()):.4f}, beta>1 {float(A.bgt1.sum() / A.cnt.sum()):.4f}, {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--job", action="append", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maxt", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    inputs = json.load(open(a.inputs))
    out = Path(a.out_dir)
    for j in a.job:
        parts = j.split(":")
        if (out / f"{parts[0]}__{parts[4].split('+')[-1]}.json").exists():
            print("skip", parts[0]); continue
        run_job(a.model, inputs, parts, out, torch.device(a.device), a.limit, a.maxt, torch.float32 if a.fp32 else torch.bfloat16)


if __name__ == "__main__":
    main()
