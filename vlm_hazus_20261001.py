#!/usr/bin/env python3
"""BuildScale P3 (10-01): Qwen3.6-27B multiple-choice inference in the Hazus scheme, street view or
satellite, with answer-token probabilities (the token-confidence signal).

Questions (single-letter answers, so the softmax over the option letters at the answer position IS
the model's distribution over options):
  turn 1  [street only] SIGN: business / institution name on a readable sign, or none
          HEIGHT: A 1-3 / B 4-7 / C 8+ stories / X target not visible          (Hazus height classes)
          USE:    A RES ... H ACC / X not visible                               (Hazus general occupancy
                                                                                  + accessory)
  turn 2  SUB: Hazus occupancy class within the chosen USE (only for RES / COM / GOV / EDU; AGR -> AGR1,
          REL -> REL1, IND and ACC have none). "Pick a subtype only if a sign or an unmistakable feature
          shows it, else X" (09-23 decision: fine class only when there is direct evidence).
Hazus specifics are spelled out in the options (hotels = RES4, warehouses = COM2, community centres =
REL1) so the model does not fall back on everyday categories.
Thinking mode off, greedy decoding (strict short-answer setting chosen on 10-01: reasoning first was
13x slower and no better on the satellite test).
Probabilities: generate(output_scores=True); for each answer we find the generation step right after
"HEIGHT:" / "USE:" / "SUB:" and take softmax over the full vocabulary, then sum the mass of each option
letter (with and without a leading space). Stored: p_<q> = {option: prob} (renormalised over options)
and pmass_<q> = total probability mass on the option letters (sanity: should be ~1).
Input  --jobs jsonl: building_id, modality ("street" | "sat"), image, [chip_side_m for sat]
Output --out_file jsonl, resumable on (building_id, modality); --shard/--nshards for parallel GPUs.
"""
import argparse
import json
import os
import re
import time

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

from vlm_baseline_infer import wait_for_cuda

HEIGHT_OPTS = {"A": "L", "B": "M", "C": "H", "X": "unseen"}
HEIGHT_TXT = ("  A: 1-3 stories (low-rise)\n  B: 4-7 stories (mid-rise)\n  C: 8 or more stories (high-rise)\n"
              "  X: the target building is not visible")
USE_OPTS = {"A": "RES", "B": "COM", "C": "IND", "D": "AGR", "E": "REL", "F": "GOV", "G": "EDU", "H": "ACC",
            "X": "unseen"}
USE_TXT = ("  A: residential -- houses, mobile homes, apartments, condos, townhouses, and also hotels/motels, "
           "dormitories, nursing homes\n"
           "  B: commercial -- shops, gas stations, warehouses and self-storage, salons and repair shops, offices, "
           "banks, hospitals and clinics, restaurants and bars, entertainment and fitness, theaters, "
           "parking garages\n"
           "  C: industrial -- factories, manufacturing or processing plants\n"
           "  D: agricultural -- barns, greenhouses, farm buildings\n"
           "  E: religious or community -- churches and other places of worship, community centres, "
           "clubhouses of membership organisations\n"
           "  F: government -- city hall, courthouse, post office, government offices, police or fire station\n"
           "  G: education -- schools, libraries, colleges, universities\n"
           "  H: accessory structure -- detached garage, shed, carport, canopy\n"
           "  X: the target building is not visible")
SUB = {  # major -> {letter: (Hazus class, description)}
    "RES": {"A": ("RES1", "single-family house"), "B": ("RES2", "mobile / manufactured home"),
            "C": ("RES3", "multi-family: apartments, condos, townhouses, duplexes"),
            "D": ("RES4", "hotel or motel"), "E": ("RES5", "dormitory"), "F": ("RES6", "nursing home")},
    "COM": {"A": ("COM1", "retail store, supermarket, pharmacy, gas station"),
            "B": ("COM2", "wholesale, warehouse, self-storage"),
            "C": ("COM3", "personal or repair service: salon, laundry, auto repair, car wash"),
            "D": ("COM4", "office"), "E": ("COM5", "bank"), "F": ("COM6", "hospital"),
            "G": ("COM7", "medical or dental clinic"),
            "H": ("COM8", "restaurant, bar, entertainment, fitness, attraction"),
            "I": ("COM9", "theater or cinema"), "J": ("COM10", "parking garage")},
    "GOV": {"A": ("GOV1", "general government services"), "B": ("GOV2", "police, fire or emergency station")},
    "EDU": {"A": ("EDU1", "school or library"), "B": ("EDU2", "college or university")},
}
FIXED_SUB = {"AGR": "AGR1", "REL": "REL1"}
MAJOR_NAME = {"RES": "residential", "COM": "commercial", "GOV": "government", "EDU": "education"}

STREET_P1 = (
    "This is a street-level photo in Orange County, Florida, USA. The target is the main building in the "
    "center of this photo. Answer about the target building only; ignore other buildings, even taller ones "
    "behind or next to it.\n"
    "SIGN: the business or institution name on a readable sign or logo on or in front of the target "
    "building, or none (street signs, plaza names and house numbers do not count).\n"
    "HEIGHT: how many stories it has above ground. Choose one letter:\n{h}\n"
    "USE: what it is used for. If a sign names the business or institution, follow the sign. "
    "Choose one letter:\n{u}\n"
    "Answer with exactly three lines and no explanation, for example:\nSIGN: none\nHEIGHT: A\nUSE: A")
SAT_P1 = (
    "This is a top-down aerial photo (north up) of an area about {side:.0f} m x {side:.0f} m in Orange County, "
    "Florida, USA. The target building's footprint is outlined in red. Answer about the outlined building only.\n"
    "HEIGHT: how many stories it has above ground (use shadow length, visible side walls, roof structure, "
    "size and surroundings). Choose one letter:\n{h}\n"
    "USE: what it is used for. Choose one letter:\n{u}\n"
    "Answer with exactly two lines and no explanation, for example:\nHEIGHT: A\nUSE: A")
P2 = ("Now choose its specific type within {name}. Pick a specific type only if a sign or an unmistakable "
      "feature shows it; otherwise answer X. Choose one letter:\n{opts}\n  X: {name}, cannot tell which type\n"
      "Answer with one line and no explanation, for example:\nSUB: X")


def load_model(model_dir):
    for name in ["AutoModelForMultimodalLM", "AutoModelForImageTextToText"]:
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    raise RuntimeError("no multimodal Auto class in this transformers version")


class Asker:
    def __init__(self, model, proc):
        self.model, self.proc, self.tok = model, proc, proc.tokenizer
        self.letter_ids = {}

    def ids_of(self, letter):
        """Token ids that decode to the bare letter, with or without a leading space."""
        if letter not in self.letter_ids:
            ids = set()
            for s in (letter, " " + letter):
                t = self.tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
            self.letter_ids[letter] = sorted(ids)
        return self.letter_ids[letter]

    def ask(self, msgs, im, max_new, keys, options):
        """Greedy generation; returns (text, {key: (probs over options, option mass)})."""
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inp = self.proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                    output_scores=True, return_dict_in_generate=True)
        gen = g.sequences[0, inp["input_ids"].shape[1]:].tolist()
        out = self.tok.decode(gen, skip_special_tokens=True).strip()
        probs = {}
        for key in keys:
            pat = re.compile(rf"{key}:\s*$")
            for k in range(len(gen)):
                if pat.search(self.tok.decode(gen[:k], skip_special_tokens=True)):
                    nxt = self.tok.decode([gen[k]]).strip()
                    if not nxt:          # a bare-space token before the letter: look one step further
                        continue
                    p = torch.softmax(g.scores[k][0].float(), -1)
                    mass = {o: float(p[self.ids_of(o)].sum()) for o in options[key]}
                    tot = sum(mass.values())
                    probs[key] = ({o: round(v / tot, 5) for o, v in mass.items()} if tot > 0 else None,
                                  round(tot, 5))
                    break
        return out, len(gen), probs


def letter(text, key):
    m = re.search(rf"{key}:\s*\**\s*([A-Z])\b", text.replace("*", ""))
    return m.group(1) if m else None


def run_one(asker, j):
    im = Image.open(j["image"]).convert("RGB")
    street = j["modality"] == "street"
    p1 = (STREET_P1.format(h=HEIGHT_TXT, u=USE_TXT) if street
          else SAT_P1.format(side=j["chip_side_m"], h=HEIGHT_TXT, u=USE_TXT))
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": p1}]}]
    opts = {"HEIGHT": list(HEIGHT_OPTS), "USE": list(USE_OPTS)}
    r1, n1, pr = asker.ask(msgs, im, 48 if street else 16, ["HEIGHT", "USE"], opts)
    hl, ul = letter(r1, "HEIGHT"), letter(r1, "USE")
    s = re.search(r"SIGN:\s*(.*)", r1)
    row = {"building_id": j["building_id"], "modality": j["modality"], "raw1": r1,
           "sign": s.group(1).strip() if s else None,
           "height": HEIGHT_OPTS.get(hl), "use_major": USE_OPTS.get(ul), "use_fine": None, "raw2": None,
           "n_gen_tokens": n1}
    for key, name in [("HEIGHT", "height"), ("USE", "use")]:
        p, mass = pr.get(key, (None, None))
        mp = HEIGHT_OPTS if key == "HEIGHT" else USE_OPTS
        row[f"p_{name}"] = {mp[o]: v for o, v in p.items()} if p else None
        row[f"pmass_{name}"] = mass
    major = row["use_major"]
    if major in FIXED_SUB:
        row["use_fine"] = FIXED_SUB[major]
    elif major in SUB:
        sub = SUB[major]
        p2 = P2.format(name=MAJOR_NAME[major], opts="\n".join(f"  {k}: {d}" for k, (_, d) in sub.items()))
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": r1}]},
                 {"role": "user", "content": [{"type": "text", "text": p2}]}]
        r2, n2, pr2 = asker.ask(msgs, im, 8, ["SUB"], {"SUB": list(sub) + ["X"]})
        sl = letter(r2, "SUB")
        row["raw2"] = r2
        row["n_gen_tokens"] += n2
        row["use_fine"] = sub[sl][0] if sl in sub else ("" if sl == "X" else None)
        p, mass = pr2.get("SUB", (None, None))
        row["p_sub"] = {(sub[o][0] if o in sub else "X"): v for o, v in p.items()} if p else None
        row["pmass_sub"] = mass
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
    jobs = [j for j in jobs if j["building_id"] % a.nshards == a.shard]
    if a.limit:
        jobs = jobs[:a.limit]
    done = set()
    if os.path.exists(a.out_file):
        done = {(d["building_id"], d["modality"]) for d in map(json.loads, open(a.out_file))}
    todo = [j for j in jobs if (j["building_id"], j["modality"]) not in done]
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
            if i < 10 or i % 200 == 0:
                print(f"[{i}/{len(todo)}] {row['building_id']} {row['modality']}: H={row['height']} "
                      f"U={row['use_major']}/{row['use_fine']} p_h={row['p_height']} m={row['pmass_height']} "
                      f"p_u_max={max(row['p_use'].values()) if row['p_use'] else None} "
                      f"{row['gen_time_s']}s | {row['raw1']!r} {row['raw2']!r}", flush=True)
    print(f"done {len(todo)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
