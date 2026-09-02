# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

import os
import time
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import product
from vllm import LLM
import argparse
import logging
import multiprocessing

logging.getLogger("vllm").setLevel(logging.WARNING)

def get_config():
    parser = argparse.ArgumentParser(description="Run Heterogeneous Face Recognition with MLLMs.")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing HFR_CARMEN_MCXFACE data.")
    parser.add_argument("--crop_dir", type=str, default=None,
                        help="Directory with cropped face images. Defaults to <base_dir>/CROPS.")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save output CSVs.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model name to load (Hugging Face or local path).")
    parser.add_argument("--enroll_modalities", nargs="+", default=["VIS"])
    parser.add_argument("--probe_modalities", nargs="+", default=["NIR", "SWIR"])
    parser.add_argument("--compare_all", action="store_true")
    parser.add_argument("--positive_samples", type=int, default=30)
    parser.add_argument("--negative_samples", type=int, default=30)
    parser.add_argument("--max_pairs", type=int, default=None)
    args = parser.parse_args()

    if args.crop_dir is None:
        args.crop_dir = os.path.join(args.base_dir, "CROPS")
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def compute_similarity(llm, image1_path, image2_path, modality_1, modality_2):
    conversation = [
        {"role": "system", "content": "You are an AI assistant specialised expert in heterogeneous face recognition."},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": Image.open(image1_path)},
                {"type": "image_pil", "image_pil": Image.open(image2_path)},
                {
                    "type": "text",
                    "text": (
                        f"I give you two face images taken under {modality_1} and {modality_2}. "
                        "On a scale from 0 to 100, how likely (as a single number) are these two faces "
                        "of the same person? Only output a single number (no other text)."
                    ),
                },
            ],
        },
    ]
    try:
        outputs = llm.chat(conversation)
        text = outputs[0].outputs[0].text.strip()
        score_str = ''.join(ch for ch in text if ch.isdigit() or ch == '.')
        score = float(score_str) if score_str else None
        return score
    except Exception as e:
        print(f"⚠️ Error comparing {image1_path} and {image2_path}: {e}")
        return None

def main():
    args = get_config()
    BASE_DIR = args.base_dir
    CROP_DIR = args.crop_dir
    OUTPUT_DIR = args.output_dir
    MODEL_NAME = args.model
    ENROLL_MODALITIES = args.enroll_modalities
    PROBE_MODALITIES = args.probe_modalities
    COMPARE_ALL_COMBINATIONS = args.compare_all
    POSITIVE_SAMPLES = args.positive_samples
    NEGATIVE_SAMPLES = args.negative_samples
    MAX_PAIRS = args.max_pairs

    # Try to set start method to 'fork' if platform permits (optional)
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        # start method already set, or not supported (e.g., Windows)
        pass
    except Exception:
        pass

    print("Loading model...")
    llm = LLM(model=MODEL_NAME, trust_remote_code=True)
    print("Model loaded.")

    enroll_mod="VIS"
    probe_mod="NIR"
    pair_name = f"{enroll_mod}-{probe_mod}"
    enroll_csv = os.path.join(BASE_DIR, f"view2_1-enroll.csv")
    probe_csv = os.path.join(BASE_DIR, f"view2_1-probe.csv")
    if not (os.path.exists(enroll_csv) and os.path.exists(probe_csv)):
        print(f"Missing CSVs for {pair_name}, skipping.")
        return 0

    OUTPUT_CSV = os.path.join(OUTPUT_DIR, pair_name, f"{MODEL_NAME.replace('/', '_')}_{pair_name}_results.csv")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results = []

    print(f"\n🔍 Processing {pair_name}: {enroll_mod} ↔ {probe_mod}")

    df_enroll = pd.read_csv(enroll_csv)
    df_probe = pd.read_csv(probe_csv)

    if COMPARE_ALL_COMBINATIONS:
        pairs = list(product(df_enroll.iterrows(), df_probe.iterrows()))
    else:
        import random
        all_pairs = list(product(df_enroll.iterrows(), df_probe.iterrows()))
        positive_pairs = [pair for pair in all_pairs if pair[0][1]["REFERENCE_ID"] == pair[1][1]["REFERENCE_ID"]]
        negative_pairs = [pair for pair in all_pairs if pair[0][1]["REFERENCE_ID"] != pair[1][1]["REFERENCE_ID"]]
        random.seed(10)
        positive_sampled = random.sample(positive_pairs, min(POSITIVE_SAMPLES, len(positive_pairs)))
        negative_sampled = random.sample(negative_pairs, min(NEGATIVE_SAMPLES, len(negative_pairs)))
        pairs = positive_sampled + negative_sampled

    if MAX_PAIRS is not None:
        pairs = pairs[:MAX_PAIRS]

    for (i_e, enroll_row), (i_p, probe_row) in tqdm(pairs, total=len(pairs), unit="pair"):
        ref_enroll = enroll_row["REFERENCE_ID"]
        ref_probe = probe_row["REFERENCE_ID"]
        same_person = int(ref_enroll == ref_probe)

        img1 = os.path.join(CROP_DIR, str(enroll_row["SPATH"]))
        img2 = os.path.join(CROP_DIR, str(probe_row["SPATH"]))

        if not img1.lower().endswith(".jpg"):
            img1 += ".jpg"
        if not img2.lower().endswith(".jpg"):
            img2 += ".jpg"

        if not (os.path.exists(img1) and os.path.exists(img2)):
            print(f"⚠️ Missing image for pair ({ref_enroll}, {ref_probe})")
            continue

        score = compute_similarity(llm, img1, img2, enroll_mod, probe_mod)
        results.append({
            "PAIR": pair_name,
            "ENROLL_ID": ref_enroll,
            "PROBE_ID": ref_probe,
            "MODALITY_1": enroll_mod,
            "MODALITY_2": probe_mod,
            "ENROLL_IMAGE": enroll_row["SPATH"],
            "PROBE_IMAGE": probe_row["SPATH"],
            "SAME_PERSON": same_person,
            "SCORE": score,
        })

        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Saved {len(df_out)} results to {OUTPUT_CSV}")

    del llm
    time.sleep(1)
    print("🧹 Done.")

if __name__ == "__main__":
    main()