#!/usr/bin/env python3
"""BuildScale (10-11): as vlm_floors_20261002.py, but HEIGHT asks the height in metres (free whole number)
instead of the number of stories (user 10-11: try asking height directly and compare with the floor version;
the LiDAR reference is a height in metres, and industrial buildings are tall but single-storey).
Turn 1 keeps the same SIGN / HEIGHT / USE layout so the prompt differs only in the HEIGHT line; turn 2 (SUB) is
skipped (use is not the point of this run). Street view only (satellite could not judge floors, 10-06).
Probabilities: the answer is a free number, so we only store the distribution of the FIRST token after
"HEIGHT:" over the digits 0-9 and X (Qwen tokenises numbers digit by digit) -> p_height_first, pmass_height.
Output jsonl: building_id, modality, raw1, sign, height_m (float or None), height_unseen, use_major, p_use, ...
"""
import argparse
import json
import os
import re
import time

import transformers
from PIL import Image
from transformers import AutoProcessor

from vlm_baseline_infer import wait_for_cuda
from vlm_floors_20261002 import USE_OPTS, USE_TXT, Asker, letter, load_model

STREET_P1 = (
    "This is a street-level photo in Orange County, Florida, USA. The target is the main building in the "
    "center of this photo. Answer about the target building only; ignore other buildings, even taller ones "
    "behind or next to it.\n"
    "SIGN: the business or institution name on a readable sign or logo on or in front of the target "
    "building, or none (street signs, plaza names and house numbers do not count).\n"
    "HEIGHT: its height in meters, from the ground to the highest point of its roof, as a whole number "
    "(your best estimate), or X if the target building is not visible.\n"
    "USE: what it is used for. If a sign names the business or institution, follow the sign. "
    "Choose one letter:\n{u}\n"
    "Answer with exactly three lines and no explanation, for example:\nSIGN: none\nHEIGHT: 6\nUSE: A")
FIRST = list("0123456789") + ["X"]


def run_one(asker, j):
    im = Image.open(j["image"]).convert("RGB")
    assert j["modality"] == "street"
    msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                         {"type": "text", "text": STREET_P1.format(u=USE_TXT)}]}]
    r1, n1, pr = asker.ask(msgs, im, 48, ["HEIGHT", "USE"], {"HEIGHT": FIRST, "USE": list(USE_OPTS)})
    m = re.search(r"HEIGHT:\s*\**\s*(\d+(?:\.\d+)?|X)\b", r1.replace("*", ""))
    s = re.search(r"SIGN:\s*(.*)", r1)
    ul = letter(r1, "USE")
    row = {"building_id": j["building_id"], "modality": j["modality"], "raw1": r1,
           "sign": s.group(1).strip() if s else None,
           "height_m": float(m.group(1)) if m and m.group(1) != "X" else None,
           "height_unseen": bool(m and m.group(1) == "X"),
           "use_major": USE_OPTS.get(ul), "n_gen_tokens": n1}
    p, mass = pr.get("HEIGHT", (None, None))
    row["p_height_first"], row["pmass_height"] = p, mass
    p, mass = pr.get("USE", (None, None))
    row["p_use"] = {USE_OPTS[o]: v for o, v in p.items()} if p else None
    row["pmass_use"] = mass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/Qwen3.6-27B")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs)]
    jobs = [j for j in jobs if j["modality"] == "street" and j["building_id"] % a.nshards == a.shard]
    if a.limit:
        jobs = jobs[:a.limit]
    done = set()
    if os.path.exists(a.out_file):
        done = {d["building_id"] for d in map(json.loads, open(a.out_file))}
    todo = [j for j in jobs if j["building_id"] not in done]
    print(f"jobs {len(jobs)}, done {len(done)}, todo {len(todo)}", flush=True)
    wait_for_cuda()
    t0 = time.time()
    model = load_model(a.model_dir).eval()
    proc = AutoProcessor.from_pretrained(a.model_dir)
    print(f"model loaded in {time.time() - t0:.0f}s, transformers {transformers.__version__}", flush=True)
    asker = Asker(model, proc)
    t0 = time.time()
    with open(a.out_file, "a") as out:
        for i, j in enumerate(todo):
            t1 = time.time()
            row = run_one(asker, j)
            row["gen_time_s"] = round(time.time() - t1, 2)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i < 10 or i % 100 == 0:
                print(f"[{i}/{len(todo)}] {row['building_id']}: H={row['height_m']} unseen={row['height_unseen']} "
                      f"U={row['use_major']} p1={row['p_height_first']} m={row['pmass_height']} "
                      f"{row['gen_time_s']}s | {row['raw1']!r}", flush=True)
    print(f"done {len(todo)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
