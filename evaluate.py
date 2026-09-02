# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

import pandas as pd
from sklearn.metrics import roc_curve
import numpy as np
import glob
import os

def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    # Find threshold where FNR = FPR
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold

def tar_at_far(y_true, scores, far_target=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Find index where FPR is closest to target FAR
    idx = np.argmin(np.abs(fpr - far_target))
    tar = tpr[idx]
    threshold = thresholds[idx]
    return tar, threshold


RESULTS_DIR = "./results"   # folder containing all CSV files
csv_files = glob.glob(os.path.join(RESULTS_DIR, "*/*.csv"))

with open('accuracy_results.txt', 'w') as f:
    # f.write(f"file, no. retrieved pairs, no. total pairs, retrieved/total pairs, EER, Threshold\n")
    f.write(f"HFR & Model & Accuire Rate & EER & TAR@FAR=1%\n")

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    orig_len = len(df)
    # Remove rows with NaN in SCORE or SAME_PERSON
    df = df.dropna(subset=["SCORE", "SAME_PERSON"])

    # SAME_PERSON column = {0,1}
    y_true = df["SAME_PERSON"].astype(int).values
    scores = df["SCORE"].astype(float).values

    if np.isnan(scores).any():
        import pdb; pdb.set_trace()
        print(f"NaN values found in SCORE column of {csv_path}")
        continue
    if np.isnan(y_true).any():
        print(f"NaN values found in SAME_PERSON column of {csv_path}")
        continue


    try:
        eer, thr = compute_eer(y_true, scores)
        tar_01, thr_01 = tar_at_far(y_true, scores, far_target=0.01)

    except Exception as e:
        print(f"Error while evaluating {csv_path}: {e}")
        continue

    print(f"{len(y_true)} scores out of {orig_len} ({round(len(y_true) / orig_len * 100)}%) in {os.path.basename(csv_path)} \t → EER = {eer*100:.2f}% \t (threshold = {thr:.2f})")
    with open('accuracy_results.txt', 'a') as f:
        f.write(f"{os.path.basename(csv_path).split('_')[2]} & {os.path.basename(csv_path).split('_')[1]} & {round(len(y_true) / orig_len * 100)}% & {eer*100:.2f}% & {tar_01*100:.2f}% \n")
