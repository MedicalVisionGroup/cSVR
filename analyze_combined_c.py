import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
import sys
import re

# --- Configuration ---
CSV_FILE = "/data/vision/polina/users/mfirenze/cSVR/evaluate_metrics/result_history.csv"
OUT_DIR = "/data/vision/polina/users/mfirenze/cSVR/train_outs"

# (json_key, default_value) — add new metrics here only
METRIC_KEYS = [
    ("mean_max",        None),
    ("median_max",      None),
    ("95_per",          None),
    ("ssim",            None),
    ("ncc",             None),
    ("psnr",            None),
    ("slice_ssim_mean", None),
    ("slice_ncc_mean",  None),
    ("slice_ssim_med",  None),
    ("slice_ncc_med",   None),
    ("slice_ncc_med11", 0),
    ("slice_ncc_med19", 0),
    ("slice_ncc_med25", 0),
    ("slice_psnr_mean", 0),
]


def agg(values, fn=np.mean):
    return np.round(fn(np.array(values)), 3)


if len(sys.argv) < 2:
    print(f"Usage: python {os.path.basename(__file__)} <json_file>")
    sys.exit(1)

json_file = sys.argv[1]

# --- Load data ---
df = pd.read_csv(CSV_FILE)

with open(json_file, "r") as f:
    data = json.load(f)

# --- Extract per-subject metrics ---
collected = {key: [] for key, _ in METRIC_KEYS}
for entry in data:
    for subject, metrics in entry.items():
        for key, default in METRIC_KEYS:
            collected[key].append(metrics.get(key, default))

num_entries = len(data)

# --- Parse ROOT_NAME from filename ---
match_root = re.match(r"combined_(.+)\.json", os.path.basename(json_file))
ROOT_NAME = match_root.group(1) if match_root else os.path.splitext(os.path.basename(json_file))[0]

# --- Parse timing info from log files ---
pose_time = []
reg_time_svr = []
recon_time_svr = []
num_slices = []

for i in range(num_entries):
    filepath = os.path.join(OUT_DIR, f"o_{ROOT_NAME}_{i}.out")
    print(filepath)
    if not os.path.exists(filepath):
        print("not exist")
        continue
    with open(filepath, "r") as f:
        text = f.read()

    match_slice_num = re.search(r"SLICE NUM:\s*(\d+)", text)
    if match_slice_num:
        slice_num = int(match_slice_num.group(1))
        print(f"Extracted slice number: {slice_num}")
        num_slices.append(slice_num)

    match_pose = re.findall(r"POSE ESTIMATION:([0-9]+\.[0-9]{3})", text)
    if match_pose:
        print("POSE TIME:", match_pose[-1])
        pose_time.append(float(match_pose[-1]))

    matches_recon = re.findall(r"RECON TIME:([0-9]+\.[0-9]{3})", text)
    if matches_recon:
        print(matches_recon[-1])
        recon_time_svr.append(float(matches_recon[-1]))

    matches_reg = re.findall(r"TIME ALL REGISTRATION:([0-9]+\.[0-9]{3})", text)
    if matches_reg:
        print("FINAL MATCH:", matches_reg[-1])
        reg_time_svr.append(float(matches_reg[-1]))

print("recon times")
print(pose_time)
print(recon_time_svr)
print(reg_time_svr)
print("num slices")
print(num_slices)

m = collected  # shorthand

# --- Build new CSV row ---
new_entry = pd.DataFrame([{
    "date":               pd.to_datetime(datetime.now()),
    "run_name":           os.path.basename(json_file),
    "num_samples":        num_entries,
    "TRE_mean_max":       agg(m["mean_max"]),
    "TRE_mean_med":       agg(m["median_max"]),
    "TRE_mean_std":       agg(m["mean_max"], np.std),
    "TRE_mean_95per":     agg(m["95_per"]),
    "TRE_mean_95per_std": agg(m["95_per"], np.std),
    "SSIM":               agg(m["ssim"]),
    "NCC":                agg(m["ncc"]),
    "SSIM_slice_mean":    agg(m["slice_ssim_mean"]),
    "NCC_slice_mean":     agg(m["slice_ncc_mean"]),
    "SSIM_slice_med":     agg(m["slice_ssim_med"]),
    "NCC_slice_med":      agg(m["slice_ncc_med"]),
    "NCC_slice_med11":    agg(m["slice_ncc_med11"]),
    "NCC_slice_med19":    agg(m["slice_ncc_med19"]),
    "NCC_slice_med25":    agg(m["slice_ncc_med25"]),
    "SSIM_slice_std":     agg(m["slice_ssim_mean"], np.std),
    "NCC_slice_std":      agg(m["slice_ncc_mean"], np.std),
    "PSNR":               agg(m["psnr"]),
    "PSNR_slice_mean":    agg(m["slice_psnr_mean"]),
    "pose time network opt": np.round(np.mean(pose_time), 3) if pose_time else 0,
    "reg time svr ":         np.round(np.mean(reg_time_svr), 3) if reg_time_svr else 0,
    "recon time svr ":       np.round(np.mean(recon_time_svr), 3) if recon_time_svr else 0,
}])

df = pd.concat([df, new_entry], ignore_index=True)
df.to_csv(CSV_FILE, index=False)
