#!/usr/bin/env python3
"""
Full download of the 78,185 matched panorama images (mapillary_metadata/
matched_image_ids.txt -- the images that survived the 50m + azimuth
ray-casting match, see match_full.py).

Each image needs 2 requests: one to the Graph API for its thumb_2048_url
(the lightweight metadata harvest never fetched image-content URLs, only
id/camera_type/etc), and one to actually download the bytes from that URL.
Concurrent (ThreadPoolExecutor, I/O-bound) with a resume checkpoint file
(mapillary_metadata/panorama_images_done.txt, one id per line, same pattern
as harvest_mapillary_metadata.py's leaf_tiles_done.txt) so a Slurm
time-limit kill or transient failure doesn't require starting over.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TOKEN_FILE = "token_maplliary.txt"
IMAGE_IDS_FILE = "mapillary_metadata/matched_image_ids.txt"
OUT_DIR = "mapillary_metadata/panorama_images"
DONE_FILE = "mapillary_metadata/panorama_images_done.txt"
FAILED_FILE = "mapillary_metadata/panorama_images_failed.txt"
RESOLUTION_FIELD = "thumb_2048_url"
N_WORKERS = 8
MAX_RETRIES = 4

os.makedirs(OUT_DIR, exist_ok=True)

with open(TOKEN_FILE) as f:
    TOKEN = f.read().strip()
HEADERS = {"Authorization": f"OAuth {TOKEN}"}

write_lock = threading.Lock()


def get_with_retry(url, params=None):
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except Exception:
            if attempt >= MAX_RETRIES:
                return None
            time.sleep(2 * attempt)
            continue
        if resp.status_code == 200:
            return resp
        # 429s get a longer backoff than generic errors
        time.sleep((5 if resp.status_code == 429 else 2) * attempt)
        if attempt >= MAX_RETRIES:
            return None


def process(image_id):
    out_path = f"{OUT_DIR}/{image_id}.jpg"
    meta_resp = get_with_retry(f"https://graph.mapillary.com/{image_id}",
                                params={"fields": RESOLUTION_FIELD})
    if meta_resp is None:
        return image_id, False, "metadata_fail"
    url = meta_resp.json().get(RESOLUTION_FIELD)
    if not url:
        return image_id, False, "no_url_field"

    img_resp = get_with_retry(url)
    if img_resp is None:
        return image_id, False, "download_fail"

    with open(out_path, "wb") as f:
        f.write(img_resp.content)
    return image_id, True, len(img_resp.content)


def main():
    with open(IMAGE_IDS_FILE) as f:
        all_ids = [line.strip() for line in f if line.strip()]

    done_ids = set()
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            done_ids = set(line.strip() for line in f)

    todo = [i for i in all_ids if i not in done_ids]
    print(f"total matched images: {len(all_ids)}, already done: {len(done_ids)}, "
          f"remaining: {len(todo)}", flush=True)

    n_ok, n_fail, total_bytes = 0, 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex, \
         open(DONE_FILE, "a") as donef, open(FAILED_FILE, "a") as failf:
        futures = {ex.submit(process, iid): iid for iid in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            image_id = futures[fut]
            try:
                image_id, ok, info = fut.result()
            except Exception as e:
                ok, info = False, f"exception:{e}"
            with write_lock:
                if ok:
                    n_ok += 1
                    total_bytes += info
                    donef.write(image_id + "\n")
                    donef.flush()
                else:
                    n_fail += 1
                    failf.write(f"{image_id}\t{info}\n")
                    failf.flush()
                if i % 500 == 0 or i == len(todo):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    remain_s = (len(todo) - i) / rate if rate > 0 else 0
                    print(f"[{elapsed:.0f}s] {i}/{len(todo)} processed, {n_ok} ok, {n_fail} fail, "
                          f"{total_bytes/1e6:.0f}MB downloaded, rate {rate:.1f}/s, "
                          f"ETA {remain_s/60:.1f} min", flush=True)

    print(f"\n=== DOWNLOAD DONE ===", flush=True)
    print(json.dumps({
        "processed_this_run": len(todo),
        "ok": n_ok, "fail": n_fail,
        "total_bytes_this_run": total_bytes,
        "elapsed_seconds": round(time.time() - t0, 1),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
