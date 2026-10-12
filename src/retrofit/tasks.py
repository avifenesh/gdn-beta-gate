"""State-tracking tasks for the negative-eigenvalue retrofit.

Every task is a token stream plus labels read from the LM head at chosen
positions. Labels never enter the stream, so the model cannot copy a previous
answer; the only way to be right at step t is to carry the state from the
inputs. Steps are counted in task steps (one bit, one swap, one statement).

Tier 1 (one reflection per step suffices, per Grazzi et al. Thm 3):
  parity    Z2 running parity of a bit stream.
  swap      one ball under five cups; each token is one transposition.
  codeswap  the same group written as Python tuple swaps, several tokens per step;
            label is the current value of variable a.
Tier 2 (needs a rotation or several reflections per step; expected to fail):
  s5full    one ball under five cups; each token is an arbitrary permutation.
  z3        running sum mod 3.
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass

TIER1 = ("parity", "swap", "codeswap")
TIER2 = ("s5full", "z3")
ALL_TASKS = TIER1 + TIER2

CUPS = "abcde"
PAIRS = [a + b for a, b in itertools.combinations(CUPS, 2)]  # 10 transpositions
PERMS = list(itertools.permutations(range(5)))  # 120


@dataclass
class Example:
    task: str
    ids: list[int]
    label_pos: list[int]  # index into ids whose logits predict the label
    labels: list[int]  # token ids
    n_steps: int


class Vocab:
    """Single-token ids for every symbol, resolved once from the tokenizer."""

    def __init__(self, tok):
        def one(s):
            ids = tok.encode(s, add_special_tokens=False)
            assert len(ids) == 1, (s, ids)
            return ids[0]

        self.tok = tok
        self.bit = [one("0"), one("1")]
        self.tri = [one("0"), one("1"), one("2")]
        self.letter = [one(" " + c) for c in "ABCDE"]
        self.pair = [one(" " + p) for p in PAIRS]
        self.nl = one("\n")
        vocab = tok.get_vocab()
        words = sorted(i for w, i in vocab.items() if re.fullmatch("Ġ[a-z]{4,6}", w))
        self.perm_word = words[:120]
        assert len(set(self.perm_word)) == 120
        self.header = {t: tok.encode(f"Task: {t}\n", add_special_tokens=False) for t in ALL_TASKS}
        self.code_init = tok.encode("".join(f"{c}={c.upper()}\n" for c in CUPS), add_special_tokens=False)
        self.code_stmt = {}
        for a, b in itertools.combinations(CUPS, 2):
            for x, y in ((a, b), (b, a)):
                self.code_stmt[(x, y)] = tok.encode(f"{x},{y}={y},{x}\n", add_special_tokens=False)
                assert self.code_stmt[(x, y)][-1] == self.nl


def make(task: str, n_steps: int, rng: random.Random, V: Vocab) -> Example:
    ids = list(V.header[task])
    pos, lab = [], []
    if task == "parity":
        p = 0
        for _ in range(n_steps):
            b = rng.randrange(2)
            p ^= b
            ids.append(V.bit[b])
            pos.append(len(ids) - 1)
            lab.append(V.letter[p])
    elif task == "z3":
        s = 0
        for _ in range(n_steps):
            d = rng.randrange(3)
            s = (s + d) % 3
            ids.append(V.tri[d])
            pos.append(len(ids) - 1)
            lab.append(V.letter[s])
    elif task == "swap":
        ball = rng.randrange(5)
        ids += [V.letter[ball], V.nl]
        for _ in range(n_steps):
            k = rng.randrange(10)
            i, j = CUPS.index(PAIRS[k][0]), CUPS.index(PAIRS[k][1])
            if ball == i:
                ball = j
            elif ball == j:
                ball = i
            ids.append(V.pair[k])
            pos.append(len(ids) - 1)
            lab.append(V.letter[ball])
    elif task == "s5full":
        ball = rng.randrange(5)
        ids += [V.letter[ball], V.nl]
        for _ in range(n_steps):
            k = rng.randrange(120)
            ball = PERMS[k][ball]
            ids.append(V.perm_word[k])
            pos.append(len(ids) - 1)
            lab.append(V.letter[ball])
    elif task == "codeswap":
        val = list(range(5))  # val[var] = value index held by variable var
        ids += V.code_init
        for _ in range(n_steps):
            i, j = rng.sample(range(5), 2)
            val[i], val[j] = val[j], val[i]
            ids += V.code_stmt[(CUPS[i], CUPS[j])]
            pos.append(len(ids) - 1)
            lab.append(V.letter[val[0]])
    else:
        raise ValueError(task)
    return Example(task, ids, pos, lab, n_steps)


def reference_check(V: Vocab) -> None:
    """Independent recomputation of the labels from the token stream (red arm for generator bugs)."""
    rng = random.Random(123)
    inv_pair = {t: k for k, t in enumerate(V.pair)}
    inv_perm = {t: k for k, t in enumerate(V.perm_word)}
    inv_letter = {t: k for k, t in enumerate(V.letter)}
    for task in ALL_TASKS:
        for _ in range(50):
            ex = make(task, rng.randrange(1, 40), rng, V)
            body = ex.ids[len(V.header[task]):]
            if task in ("parity", "z3"):
                m = 2 if task == "parity" else 3
                syms = V.bit if task == "parity" else V.tri
                s, want = 0, []
                for t in body:
                    s = (s + syms.index(t)) % m
                    want.append(s)
            elif task in ("swap", "s5full"):
                ball = inv_letter[body[0]]
                want = []
                for t in body[2:]:
                    if task == "swap":
                        a, b = PAIRS[inv_pair[t]]
                        a, b = CUPS.index(a), CUPS.index(b)
                        ball = b if ball == a else a if ball == b else ball
                    else:
                        ball = PERMS[inv_perm[t]][ball]
                    want.append(ball)
            else:
                text = V.tok.decode(body)
                env = {c: c.upper() for c in CUPS}
                want = []
                for line in text.split("\n")[5:]:
                    if not line:
                        continue
                    lhs, rhs = line.split("=")
                    x, y = lhs.split(",")
                    env[x], env[y] = env[y], env[x]
                    want.append("ABCDE".index(env["a"]))
            got = [inv_letter[t] for t in ex.labels]
            assert got == want, (task, got[:10], want[:10])
