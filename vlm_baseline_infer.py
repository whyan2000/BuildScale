"""Run Qwen2.5-VL over the baseline_crops/ manifest, asking for a floor
count, and compare against the OSM building:levels ground truth.

v1 baseline smoke test -- single image per building, simple prompt, no
prompt engineering/calibration yet. Purpose is to see what the model
actually outputs before investing in a real pipeline.
"""

import argparse
import json
import os
import re
import subprocess
import time

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

PROMPT = (
    "This is a street-level photo of a building in a US suburban/urban area. "
    "How many floors/stories does the main building in the center of this "
    "photo appear to have? Look at window rows, distinct facade levels, and "
    "roofline. Answer in this exact format: FLOORS: <integer>. If you "
    "cannot tell, answer FLOORS: unknown."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--crops_dir", default="baseline_crops")
    p.add_argument("--out_file", default="baseline_results.jsonl")
    return p.parse_args()


def extract_floor_count(text):
    m = re.search(r"FLOORS:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def wait_for_cuda(max_attempts=8, delay_s=15):
    for attempt in range(1, max_attempts + 1):
        if torch.cuda.is_available():
            return
        print(f"CUDA not available yet (attempt {attempt}/{max_attempts}), "
              f"retrying in {delay_s}s...", flush=True)
        if attempt == 1:
            try:
                out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
                print("nvidia-smi stdout:\n" + out.stdout, flush=True)
                print("nvidia-smi stderr:\n" + out.stderr, flush=True)
            except Exception as e:
                print(f"nvidia-smi failed to run: {e}", flush=True)
        time.sleep(delay_s)
        try:
            torch.cuda.init()
        except Exception as e:
            print(f"torch.cuda.init() raised: {e}", flush=True)
    raise RuntimeError(f"CUDA still not available after {max_attempts} attempts")


def main():
    args = parse_args()
    wait_for_cuda()
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    manifest = []
    with open(os.path.join(args.crops_dir, "manifest.jsonl")) as f:
        for line in f:
            manifest.append(json.loads(line))

    done_ids = set()
    if os.path.exists(args.out_file):
        with open(args.out_file) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["building_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"resuming: {len(done_ids)} buildings already done in {args.out_file}", flush=True)

    manifest = [m for m in manifest if m["building_id"] not in done_ids]
    print(f"{len(manifest)} crops to run (after resume filter)", flush=True)
    if not manifest:
        print("nothing to do, exiting", flush=True)
        return

    t0 = time.time()
    model = None
    for attempt in range(1, 6):
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda"
            )
            break
        except RuntimeError as e:
            print(f"model load attempt {attempt}/5 failed: {e}", flush=True)
            if attempt == 5:
                raise
            time.sleep(15)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    results = []
    errors = []
    t_start = time.time()
    with open(args.out_file, "a") as out_f:
        for i, m in enumerate(manifest):
            img_path = os.path.join(args.crops_dir, m["crop_file"])
            image = Image.open(img_path).convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")

            t1 = time.time()
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            gen_time = time.time() - t1

            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            raw_response = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            pred = extract_floor_count(raw_response)

            row = {**m, "vlm_raw": raw_response, "vlm_pred_floors": pred, "gen_time_s": round(gen_time, 2)}
            results.append(row)
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            if pred is None:
                errors.append(m["building_id"])
            if (i + 1) % 500 == 0 or i == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta_min = (len(manifest) - i - 1) / rate / 60 if rate > 0 else float("inf")
                print(f"[{i+1}/{len(manifest)}] building={m['building_id']} "
                      f"pred={pred} raw='{raw_response}' ({gen_time:.1f}s) "
                      f"rate={rate:.2f}/s ETA={eta_min:.1f}min", flush=True)

    parsed = [r for r in results if r["vlm_pred_floors"] is not None]
    have_gt = [r for r in parsed if r.get("levels_gt") is not None]
    print(f"\n=== SUMMARY (this run) ===")
    print(f"total this run: {len(results)}, parsed: {len(parsed)}, unparseable: {len(errors)}")
    if have_gt:
        errs = [abs(r["vlm_pred_floors"] - r["levels_gt"]) for r in have_gt]
        exact = sum(1 for e in errs if e == 0)
        within1 = sum(1 for e in errs if e <= 1)
        mae = sum(errs) / len(errs)
        print(f"(of {len(have_gt)} with ground truth) exact match: {exact}/{len(have_gt)} "
              f"({100*exact/len(have_gt):.1f}%)")
        print(f"within 1 floor: {within1}/{len(have_gt)} ({100*within1/len(have_gt):.1f}%)")
        print(f"MAE: {mae:.2f} floors")
    print(f"appended to {args.out_file}")


if __name__ == "__main__":
    main()
