"""Broad SFT data for step 6: the Tulu 3 SFT mixture minus its non-commercial subset, rendered as ChatML.

Choice (receipt: results/broad/staging_queue.json): allenai/tulu-3-sft-mixture is a public, documented mix of general chat, instruction
following, knowledge (FLAN v2), math (Persona MATH/GSM/Algebra, NuminaMath-TIR), code (Persona Python, Evol
CodeAlpaca), science (SciRIFF), tables, multilingual (Aya) and safety data, licensed ODC-BY-1.0 as a
collection. Its card lists one non-commercial subset, No Robots (CC-BY-NC-4.0); prepare() drops it, so every
remaining subset is ODC-BY-1.0, Apache-2.0, MIT or CC-BY-4.0 by the card.

Rendering: <|im_start|>{role}\\n{content}<|im_end|>\\n per message. Loss covers each assistant message's
content and its <|im_end|>; the role header, the trailing newline and every non-assistant message are context.
Segments are tokenized separately and concatenated, so the label mask is exact by construction.

Stream: a run's examples are a seed-dependent permutation of the train rows (identical for every arm at a
seed); rows longer than max_len tokens or with no assistant tokens are skipped deterministically. A fixed 1%
of rows, chosen by a hash of the row id, is held out for validation loss and never trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DROP_SOURCES = ("no_robots",)  # CC-BY-NC-4.0 per the Tulu 3 card; matched as a substring of the source field
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def is_holdout(row_id: str, frac_permille: int = 10) -> bool:
    return int(hashlib.sha256(f"tulu3:{row_id}".encode()).hexdigest()[:8], 16) % 1000 < frac_permille


def prepare(src_files: list[str], out: str) -> dict:
    """Write the filtered mixture to one parquet (id, source, messages as JSON text). Returns counts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("id", pa.large_string()), ("source", pa.large_string()), ("messages", pa.large_string())])
    rows_in = rows_out = 0
    by_src: dict = {}
    with pq.ParquetWriter(out, schema) as w:
        for f in src_files:  # one file at a time: the whole mixture overflows 32-bit offsets when concatenated
            t = pq.read_table(f, columns=["id", "messages", "source"])
            src = t.column("source").to_pylist()
            ids = t.column("id").to_pylist()
            msgs = t.column("messages").to_pylist()
            keep = [i for i, s in enumerate(src) if not any(d in (s or "") for d in DROP_SOURCES)]
            rows_in += len(src)
            rows_out += len(keep)
            for i in keep:
                by_src[src[i]] = by_src.get(src[i], 0) + 1
            w.write_table(pa.table({"id": pa.array([str(ids[i]) for i in keep], pa.large_string()),
                                    "source": pa.array([src[i] for i in keep], pa.large_string()),
                                    "messages": pa.array([json.dumps(msgs[i], ensure_ascii=False) for i in keep],
                                                         pa.large_string())}, schema=schema))
    return {"rows_in": rows_in, "rows_out": rows_out, "dropped": rows_in - rows_out, "by_source": by_src}


@dataclass
class Example:
    row: int
    ids: list[int]
    labels: list[int]  # -100 where no loss


class Renderer:
    def __init__(self, tok):
        self.tok = tok
        enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
        self.im_start, self.im_end = enc(IM_START), enc(IM_END)
        assert len(self.im_start) == 1 and len(self.im_end) == 1, (self.im_start, self.im_end)
        self.nl = enc("\n")
        self.enc = enc

    def render(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        ids, labels = [], []
        for m in messages:
            role, content = m["role"], m["content"] or ""
            head = self.im_start + self.enc(f"{role}\n")
            body = self.enc(content) + self.im_end
            ids += head + body + self.nl
            if role == "assistant":
                labels += [-100] * len(head) + body + [-100] * len(self.nl)
            else:
                labels += [-100] * (len(head) + len(body) + len(self.nl))
        return ids, labels

    def prompt(self, messages: list[dict]) -> list[int]:
        """Context for generation: the messages, then the open assistant header."""
        ids, _ = self.render(messages)
        return ids + self.im_start + self.enc("assistant\n")


class Mixture:
    def __init__(self, path: str):
        import pyarrow.parquet as pq

        t = pq.read_table(path)
        self.ids = t.column("id").to_pylist()
        self.sources = t.column("source").to_pylist()
        self.messages = t.column("messages").to_pylist()  # JSON text, parsed on use
        self.hold = np.array([is_holdout(str(i)) for i in self.ids])
        self.train_rows = np.flatnonzero(~self.hold)
        self.val_rows = np.flatnonzero(self.hold)

    def msgs(self, row: int) -> list[dict]:
        return json.loads(self.messages[row])


class Stream:
    """Deterministic example stream for one seed. state() / skip_to() make a run resumable."""

    def __init__(self, mix: Mixture, renderer: Renderer, seed: int, max_len: int):
        self.mix, self.r, self.max_len = mix, renderer, max_len
        self.order = np.random.default_rng(seed).permutation(mix.train_rows)
        self.pos = 0
        self.skipped = 0

    def next(self) -> Example:
        while True:
            if self.pos >= len(self.order):
                raise StopIteration("mixture exhausted")
            row = int(self.order[self.pos])
            self.pos += 1
            ids, labels = self.r.render(self.mix.msgs(row))
            if len(ids) > self.max_len or all(l == -100 for l in labels):
                self.skipped += 1
                continue
            return Example(row, ids, labels)

    def step(self, step_tokens: int) -> list[Example]:
        """Examples for one optimizer step: drawn until the real token count reaches step_tokens."""
        exs, n = [], 0
        while n < step_tokens:
            e = self.next()
            exs.append(e)
            n += len(e.ids)
        return exs


def micro_batches(exs: list[Example], micro_tokens: int) -> list[list[Example]]:
    """Length-sorted groups whose padded size (rows x longest) stays within micro_tokens."""
    exs = sorted(exs, key=lambda e: -len(e.ids))
    groups, cur = [], []
    for e in exs:
        longest = len(cur[0].ids) if cur else len(e.ids)
        if cur and (len(cur) + 1) * longest > micro_tokens:
            groups.append(cur)
            cur = []
        cur.append(e)
    if cur:
        groups.append(cur)
    return groups


def val_set(mix: Mixture, renderer: Renderer, n: int, max_len: int) -> list[Example]:
    out = []
    for row in mix.val_rows:
        ids, labels = renderer.render(mix.msgs(int(row)))
        if len(ids) <= max_len and any(l != -100 for l in labels):
            out.append(Example(int(row), ids, labels))
        if len(out) == n:
            break
    return out


if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--src_glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(a.src_glob))
    assert files, a.src_glob
    res = prepare(files, a.out)
    res["src_files"] = files
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_source"}))
