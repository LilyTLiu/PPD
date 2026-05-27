#!/usr/bin/env python
"""PPD: Prior-informed Preference Distillation for Offline MBO.

TabPFN V2.6 teacher → two-stage distillation set → uncertainty-weighted
ListNet student → gradient-based design optimization.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from pipeline.utils import (
    set_seed, str2bool, parse_hidden_dims,
    fit_standardizer, apply_standardizer,
    infer_is_discrete_task, clear_cuda_memory,
)
from pipeline.teacher import TabPFNTeacher
from pipeline.distill_set import build_distill_set
from pipeline.student import SimpleMLP, train_student
from pipeline.search import optimize_candidates


def _prepare_teacher_context(X_off_full, y_off_full, X_off, y_off,
                             teacher_ctx_max, n_synth_hint):
    """Handle large datasets: dedup + top-K by y-value for teacher context."""
    if teacher_ctx_max <= 0:
        return X_off, y_off
    X_src, y_src = X_off_full, y_off_full

    X_int = np.rint(X_src).astype(np.int64)
    key_to_indices = {}
    for i, row in enumerate(X_int):
        key = tuple(row)
        if key not in key_to_indices:
            key_to_indices[key] = []
        key_to_indices[key].append(i)
    unique_X, unique_y = [], []
    for key, indices in key_to_indices.items():
        unique_X.append(X_src[indices[0]])
        unique_y.append(np.mean(y_src[indices]))
    unique_X = np.array(unique_X, dtype=np.float32)
    unique_y = np.array(unique_y, dtype=np.float32)
    n_dedup = len(unique_X)
    sorted_idx = np.argsort(unique_y)

    budget = max(1, teacher_ctx_max - n_synth_hint)
    for _ in range(5):
        actual_ctx = min(budget, n_dedup)
        n_synth_est = n_synth_hint
        new_budget = max(1, teacher_ctx_max - n_synth_est)
        if new_budget == budget:
            break
        budget = new_budget

    if n_dedup <= budget:
        X_ctx, y_ctx = unique_X, unique_y
    else:
        top_idx = sorted_idx[-budget:]
        X_ctx, y_ctx = unique_X[top_idx], unique_y[top_idx]

    print(f"[Teacher Context] Full {len(X_src)} -> dedup {n_dedup} -> "
          f"top-{len(X_ctx)} (budget={budget}, est_synth={n_synth_hint})")
    return X_ctx, y_ctx


def main():
    p = argparse.ArgumentParser(description="PPD: Prior-informed Preference Distillation for MBO")
    # Required
    p.add_argument("--task_name", type=str, required=True)
    p.add_argument("--data_X", type=str, required=True)
    p.add_argument("--data_y", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    # General
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--maximize_flag", type=str, default="true")
    # Preprocessing
    p.add_argument("--normalize_xs", type=str, default="true")
    p.add_argument("--normalize_ys", type=str, default="true")
    p.add_argument("--normalize_logits", type=str, default="true")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Subsample offline data. -1 = use all.")
    p.add_argument("--teacher_context_max", type=int, default=0,
                   help="Max context samples for teacher (0=use all).")
    # Teacher
    p.add_argument("--teacher_synth_ratio", type=float, default=0.005,
                   help="Synthetic sample ratio for teacher context.")
    p.add_argument("--conservative_only", type=str, default="false")
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--teacher_predict_batch_size", type=int, default=4096)
    # Distillation set
    p.add_argument("--global_ratio", type=float, default=0.35)
    p.add_argument("--global_sampling_method", type=str, default="sobol",
                   choices=["uniform", "sobol"])
    p.add_argument("--global_refine_ratio", type=float, default=2.0,
                   help="Extra samples per global anchor (0=off).")
    p.add_argument("--global_refine_scale", type=float, default=0.05)
    p.add_argument("--local_sampling_method", type=str, default="density")
    p.add_argument("--priority_pool_multiplier", type=float, default=4.0)
    p.add_argument("--priority_pool_max", type=int, default=60000)
    p.add_argument("--priority_power", type=float, default=2.5)
    p.add_argument("--priority_select_ratio", type=float, default=0.35)
    p.add_argument("--cluster_noise_scale", type=float, default=0.35)
    p.add_argument("--score_pred_weight", type=float, default=0.7)
    p.add_argument("--score_if_weight", type=float, default=0.3)
    p.add_argument("--if_contamination", type=float, default=0.1)
    p.add_argument("--mutation_rate", type=float, default=0.12)
    p.add_argument("--max_mutations_per_seq", type=int, default=2)
    p.add_argument("--discrete_oversample", type=int, default=3)
    # OOD & Quality filters
    p.add_argument("--structured_ood_quantile", type=float, default=0.95)
    p.add_argument("--structured_ood_scale", type=float, default=1.2)
    p.add_argument("--ood_relax_scale", type=float, default=2.5)
    p.add_argument("--ood_min_keep_ratio", type=float, default=0.8)
    p.add_argument("--ood_hamming_quantile", type=float, default=0.95)
    p.add_argument("--ood_hamming_scale", type=float, default=1.5)
    p.add_argument("--skip_ood_filter", type=str, default="false")
    p.add_argument("--distill_min_pred_quantile", type=float, default=0.0,
                   help="Quality gate: keep only candidates above this quantile of offline y.")
    p.add_argument("--uncertainty_drop_ratio", type=float, default=0.2)
    # Uncertainty weighting
    p.add_argument("--uncertainty_weight_beta", type=float, default=1.0,
                   help="Sample-level uncertainty weighting (0=off, 1.0=on).")
    # MLP
    p.add_argument("--hidden_dims", type=str, default="2048,2048")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Distillation loss weight.")
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--list_loss", type=str, default="listnet",
                   choices=["listnet", "rankcosine"])
    p.add_argument("--list_num_samples", type=int, default=10000)
    p.add_argument("--list_length", type=int, default=1000)
    p.add_argument("--list_batch_size", type=int, default=128)
    p.add_argument("--validation_split", type=float, default=0.2)
    p.add_argument("--drop_last", type=str, default="true")
    # Search
    p.add_argument("--search_lr", type=float, default=-1.0)
    p.add_argument("--search_lr_continuous", type=float, default=1e-3)
    p.add_argument("--search_lr_discrete", type=float, default=0.1)
    p.add_argument("--search_steps", type=int, default=-1)
    p.add_argument("--search_steps_continuous", type=int, default=200)
    p.add_argument("--search_steps_discrete", type=int, default=100)
    p.add_argument("--num_solutions", type=int, default=128)

    args = p.parse_args()
    args.maximize = str2bool(args.maximize_flag)
    args.conservative_only = str2bool(args.conservative_only)
    args.skip_ood_filter = str2bool(args.skip_ood_filter)
    args.drop_last = str2bool(args.drop_last)

    # GPU setup
    if args.gpu and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Load data ──
    X_off_full = np.load(args.data_X).astype(np.float32)
    y_off_full = np.load(args.data_y).astype(np.float32).reshape(-1)
    print(f"[Data] X={X_off_full.shape}, y={y_off_full.shape}")

    _max_samples = int(getattr(args, 'max_samples', -1))
    if _max_samples > 0 and len(X_off_full) > _max_samples:
        sub_rng = np.random.RandomState(args.seed)
        sub_idx = sub_rng.choice(len(X_off_full), size=_max_samples, replace=False)
        X_off, y_off = X_off_full[sub_idx], y_off_full[sub_idx]
        print(f"[Data] Subsampled to {_max_samples}")
    else:
        X_off, y_off = X_off_full, y_off_full

    is_discrete = infer_is_discrete_task(args.task_name)
    normalize_x = str2bool(args.normalize_xs) and (not is_discrete or str2bool(args.normalize_logits))
    normalize_y = str2bool(args.normalize_ys)
    x_stats = fit_standardizer(X_off) if normalize_x else None
    y_stats = fit_standardizer(y_off.reshape(-1, 1)) if normalize_y else None
    X_off_m = apply_standardizer(X_off, x_stats) if x_stats else X_off.copy()
    y_off_m = apply_standardizer(y_off.reshape(-1, 1), y_stats).reshape(-1) if y_stats else y_off.copy()

    # ── Teacher ──
    use_distill = float(args.alpha) > 0.0
    X_dis, y_soft, y_unc = (np.empty((0, X_off.shape[1]), dtype=np.float32),
                              np.empty((0,), dtype=np.float32),
                              np.empty((0,), dtype=np.float32))
    d_stats = {}

    if use_distill:
        ratio = float(args.teacher_synth_ratio)
        t_adv = max(1, int(round(len(X_off) * ratio))) if ratio > 0 else 0
        t_inn = 0 if args.conservative_only else (max(1, int(round(len(X_off) * ratio))) if ratio > 0 else 0)

        teacher_ctx_max = int(getattr(args, 'teacher_context_max', 0))
        X_ctx, y_ctx = _prepare_teacher_context(
            X_off_full, y_off_full, X_off, y_off, teacher_ctx_max, t_adv + t_inn)

        teacher = TabPFNTeacher(
            task_name=args.task_name,
            num_adversarial=t_adv,
            num_innovative=t_inn,
            conservative_only=args.conservative_only,
            margin=args.margin,
            device=str(device),
            random_state=args.seed,
        )
        teacher.fit(X_ctx, y_ctx)
        print(f"[Teacher] Context: {len(X_ctx)} offline + synth = {len(teacher.X_)} total")

        X_dis, y_soft, y_unc, d_stats = build_distill_set(
            teacher, X_off, y_off, len(X_off), args, is_discrete)
        del teacher
        clear_cuda_memory()
    else:
        t_adv, t_inn = 0, 0

    # ── Student ──
    X_dis_m = apply_standardizer(X_dis, x_stats) if x_stats else X_dis.copy()
    y_soft_m = (apply_standardizer(y_soft.reshape(-1, 1), y_stats).reshape(-1)
                if (y_stats and len(y_soft) > 0) else y_soft.copy())

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    student = SimpleMLP(input_dim=X_off_m.shape[1], hidden_dims=hidden_dims, output_dim=1).to(device)
    best_val, tr_stats = train_student(
        student, X_off_m, y_off_m, X_dis_m, y_soft_m, y_unc, args, device)

    # ── Search ──
    cands, pred, s_stats = optimize_candidates(
        student, X_off_m, y_off_m, x_stats, args, device, is_discrete)

    # ── Save ──
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.save(args.save_path, cands.astype(np.float32))

    info = {
        "task_name": args.task_name, "seed": args.seed,
        "pipeline": "PPD",
        "alpha": args.alpha, "is_discrete": is_discrete,
        "best_validation_loss": float(best_val),
        "num_distill": len(X_dis), "t_adv": t_adv, "t_inn": t_inn,
        "teacher_model": "V2_6",
        "global_sampling": args.global_sampling_method,
        "local_sampling": args.local_sampling_method,
        "global_refine_ratio": args.global_refine_ratio,
        "uw_beta": args.uncertainty_weight_beta,
        "distill_min_pred_quantile": args.distill_min_pred_quantile,
        "candidate_pred_max": float(np.max(pred)) if len(pred) > 0 else 0.0,
        "candidate_pred_min": float(np.min(pred)) if len(pred) > 0 else 0.0,
    }
    info.update(d_stats)
    info.update(tr_stats)
    info.update(s_stats)
    with open(args.save_path.replace(".npy", "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
