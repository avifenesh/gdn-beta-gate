#!/usr/bin/env python3
"""Step 3 (9B part): the 4B scorer's battery (gdn_gate/rescore/rescore_4b.py, imported unchanged) on a banked
Qwen3.5-9B-Base retrofit checkpoint. Only the base model path and the replay corpora paths differ: they are module
constants in rescore_4b.py, set here from --model and --replay_* before its main() runs. Qwen3.5-9B also has 24 GDN
layers (32 layers, 3:1), so rescore_4b's strip-twin assert (24 gates zeroed) holds unchanged.

Usage: rescore_9b.py --model DIR --replay_wikitext P --replay_clean P -- <rescore_4b.py arguments>
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# rescore_4b.py lives in the same directory
if not os.path.isfile(os.path.join(HERE, "rescore_4b.py")):
    sys.exit("rescore_9b: rescore_4b.py not found next to " + HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--replay_wikitext", required=True)
ap.add_argument("--replay_clean", required=True)
a, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]

import rescore_4b  # noqa: E402

rescore_4b.MODEL = a.model
rescore_4b.REPLAYS = {"wikitext": a.replay_wikitext, "clean": a.replay_clean}
sys.argv = [os.path.join(sys.path[0], "rescore_4b.py")] + rest
rescore_4b.main()
