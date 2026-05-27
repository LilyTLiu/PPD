#!/usr/bin/env python
"""Oracle evaluation on Design-Bench tasks."""

import argparse
import importlib
import json
import os
import sys
import traceback

import numpy as np
import design_bench as db

TASK_DATASET_MAP = {
    "DKittyMorphology-Exact-v0": ("design_bench.datasets.continuous.dkitty_morphology_dataset", "DKittyMorphologyDataset"),
    "AntMorphology-Exact-v0": ("design_bench.datasets.continuous.ant_morphology_dataset", "AntMorphologyDataset"),
    "TFBind8-Exact-v0": ("design_bench.datasets.discrete.tf_bind_8_dataset", "TFBind8Dataset"),
    "TFBind10-Exact-v0": ("design_bench.datasets.discrete.tf_bind_10_dataset", "TFBind10Dataset"),
    "Superconductor-RandomForest-v0": ("design_bench.datasets.continuous.superconductor_dataset", "SuperconductorDataset"),
}


def get_oracle_y_range(task_name):
    if task_name in TASK_DATASET_MAP:
        module_path, class_name = TASK_DATASET_MAP[task_name]
        module = importlib.import_module(module_path)
        dataset_cls = getattr(module, class_name)
        dataset = dataset_cls()
        y = dataset.y
    else:
        task = db.make(task_name)
        y = task.y
    return float(y.min()), float(y.max())


def evaluate_multiple_designs(task_name, designs_paths, results_dir, method=None, data_dir=None):
    print(f"Loading Design-Bench Task: {task_name}")
    task = db.make(task_name)

    y_offline = task.y
    offline_best = np.max(y_offline)
    offline_best_normalized = None
    oracle_y_min, oracle_y_max = get_oracle_y_range(task_name)

    gen_100th_raw_list, gen_50th_raw_list = [], []

    for seed, path in enumerate(designs_paths):
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist. Skipping.")
            continue
        if path.endswith("_pred_means.npy") or path.endswith("_pred_stds.npy"):
            continue

        designs = np.load(path)
        y_true = task.predict(designs).ravel()

        oracle_y_min = min(oracle_y_min, float(np.min(y_true)))
        oracle_y_max = max(oracle_y_max, float(np.max(y_true)))

        generated_100th = np.max(y_true)
        generated_50th = np.median(y_true)
        gen_100th_raw_list.append(generated_100th)
        gen_50th_raw_list.append(generated_50th)

        if offline_best_normalized is None:
            offline_best_normalized = (float(offline_best) - oracle_y_min) / (oracle_y_max - oracle_y_min)

        norm_100th = (float(generated_100th) - oracle_y_min) / (oracle_y_max - oracle_y_min)
        norm_50th = (float(generated_50th) - oracle_y_min) / (oracle_y_max - oracle_y_min)

        seed_result = {
            "generated_100th_raw": float(generated_100th),
            "generated_50th_raw": float(generated_50th),
            "generated_100th_normalized": norm_100th,
            "generated_50th_normalized": norm_50th,
        }
        print(f"  Seed {seed}: 100th={generated_100th:.4f} ({norm_100th:.4f}), 50th={generated_50th:.4f} ({norm_50th:.4f})")

    if not gen_100th_raw_list:
        print("No valid designs evaluated.")
        return None

    mean_100th = np.mean(gen_100th_raw_list)
    std_100th = np.std(gen_100th_raw_list)
    mean_50th = np.mean(gen_50th_raw_list)
    std_50th = np.std(gen_50th_raw_list)
    norm_100th_mean = (mean_100th - oracle_y_min) / (oracle_y_max - oracle_y_min)
    norm_50th_mean = (mean_50th - oracle_y_min) / (oracle_y_max - oracle_y_min)

    print(f"\n{'='*40}")
    print(f"100th Percentile (Raw):  {mean_100th:.4f} ± {std_100th:.4f}")
    print(f"100th Percentile (Norm): {norm_100th_mean:.4f}")
    print(f"50th  Percentile (Raw):  {mean_50th:.4f} ± {std_50th:.4f}")
    print(f"50th  Percentile (Norm): {norm_50th_mean:.4f}")
    print(f"Offline Best (Norm):     {offline_best_normalized:.4f}")

    summary = {
        "task": task_name,
        "oracle_y_min": oracle_y_min,
        "oracle_y_max": oracle_y_max,
        "offline_best_raw": float(offline_best),
        "offline_best_normalized": offline_best_normalized,
        "generated_100th_raw_mean": float(mean_100th),
        "generated_100th_raw_std": float(std_100th),
        "generated_50th_raw_mean": float(mean_50th),
        "generated_50th_raw_std": float(std_50th),
        "generated_100th_normalized_mean": float(norm_100th_mean),
        "generated_50th_normalized_mean": float(norm_50th_mean),
    }

    method_tag = method or "PPD"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"{task_name}_{method_tag}_summary_eval.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--designs_paths", type=str, nargs='+', required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="PPD")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    evaluate_multiple_designs(
        task_name=args.task,
        designs_paths=args.designs_paths,
        results_dir=args.results_dir,
        method=args.method,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
