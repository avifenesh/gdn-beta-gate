# Compute

Rented compute for the paper was about 293 GPU-hours on NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (96 GB):
about 83 GPU-hours for the 4B and 9B forgetting runs, about 173 in the three rentals of the RL records and about 37
for the fresh held-out set and the mechanism readouts (Phase 5), each rental one server. Times are wall-clock from
start to teardown, Israel time, taken from the lab's rental log. Pilots, calibration, follow-up runs and stopped runs
inside these windows are included.

| Rental | GPUs | Wall time | GPU-hours | Work |
|---|---|---|---|---|
| 0 (forgetting) | 8 x RTX PRO 6000, 10.4 h | 2026-09-29 21:09 to 09-30 07:31 | 83 | forgetting recipe: 4B LoRA and Gate(0,2) seeds 0 to 2, all six 9B runs, plus unreported follow-ups (clean replay, decay gate, 128-step sequences, 9B continuations) |
| 1 | 8 x RTX PRO 6000, 9.5 h | 2026-10-11 01:40 to 11:08 | 76 | calibration; 2B test read 1; 2B same-hardware validation set; chat SFT seeds 1 to 5 (15 runs); Gate(0,1) forgetting runs, same-server LoRA and Gate(0,2) seed 0 twins, scorer checks and HumanEval of 9B LoRA seed 1 |
| 2 | 8 x RTX PRO 6000, 10.3 h | 2026-10-11 13:29 to 23:45 | 82 | 4B record: answer cap, cap pilots, warm-up, 9 arm runs, validation, test read 1; 2B seeds 3 to 5 and test read 2 |
| 3 | 4 x RTX PRO 6000, 3.7 h | 2026-10-12 00:07 to 03:47 | 15 | 4B LoRA-r20 arm (3 seeds), its validation and test read 2 |
| 4 (Phase 5) | 4 x RTX PRO 6000, 4.9 h | 2026-10-12 05:04 to about 10:00 | 20 | fresh held-out set readouts (12 of the 14 4B readouts finished here) and mechanism jobs; spot server, evicted |
| 5 (Phase 5) | 8 x RTX PRO 6000, 2.2 h | 2026-10-12 10:08 to 12:19 | 18 | the rest of the fresh readouts (4B and 2B) and mechanism jobs, from the same disk |

Rental 1 ran as a preemptible server from 01:40 to 05:15 and then as an on-demand server from 05:29 to 11:08.
Rental 4 was a spot server and was evicted at about 10:00. The finished readouts had already been copied off and
checked by sha256; four unfinished readouts (two 4B, two 2B) restarted from scratch on rental 5, which started from the
same disk with the code lock and the done markers intact. No finished readout was rerun.

Owned hardware, not counted above: two NVIDIA DGX Spark (GB10) units trained the 2B warm-up, its pilots and the 2B
runs of seeds 0 to 2, the chat SFT seed 0 runs (three arms) and LoRA seeds 4 and 5 of the 4B forgetting recipe, and
scored the 9B forgetting runs (except LoRA seed 1) and those two LoRA seeds. The same two units ran the whole
Qwen3.5-9B RL record (v1.2.0): the warm-up and the six arm runs (71 h of logged training steps; 2.8 h for the warm-up,
10.8 to 12.0 h per 150-step run), the seven validation readouts (9.7 h) and the seven AIME 2025 readouts (18.5 h of
readout time; the read took 11.9 h of wall time on the two units, 2026-10-16 00:58 to 12:51 Israel time), about 99
GB10-hours in all (`paper/data/compute_hours.json`, `r9b_*`). One RTX 5090 Laptop GPU (24 GB) ran the
Olmo-Hybrid-7B lesions, the 2B validation readouts, the learning-rate pilots of the forgetting recipe, the HumanEval
scoring of the 4B LoRA and Gate(0,2) forgetting runs (seeds 0 to 2), and the counting probe and HumanEval battery of
the chat SFT seed 0 runs.

The GPU type of every scored readout is recorded in its env block; `results/ENVIRONMENT.json` lists them
(`python scripts/collect_env.py`).
