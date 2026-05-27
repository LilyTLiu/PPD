"""Two-stage distillation set construction.

Stage 1: Global Exploration — Sobol sampling + teacher scoring + local refinement.
Stage 2: Local Augmentation  — density-aware perturbation around top offline seeds.
Post-processing: OOD filter → quality gate (pred quantile) → uncertainty filter.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

from .utils import _normalize_01


# ── Sobol low-discrepancy sampling ──────────────────────────
def _sample_sobol(low, high, n, d, seed, skip=0):
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=d, scramble=True, seed=seed)
    samples = sampler.random(n + skip)[skip:]
    return (low + samples * (high - low)).astype(np.float32)


# ── Block structure detection (discrete tasks) ─────────────
def _detect_block_structure(X_off, min_bs=2, max_bs=32):
    X_int = np.rint(X_off).astype(np.int64)
    n, d = X_int.shape
    if d < min_bs * 2:
        return None
    for bs in range(min_bs, min(max_bs + 1, d // 2 + 1)):
        if d % bs != 0:
            continue
        blocks = [(i * bs, (i + 1) * bs) for i in range(d // bs)]
        if all(len(np.unique(X_int[:, s:e], axis=0)) <= bs * 2
               and len(np.unique(X_int[:, s:e], axis=0)) >= 2
               for s, e in blocks):
            return blocks
    return None


def _make_discrete_mutator(X_off, rng, blocks=None):
    X_int = np.rint(X_off).astype(np.int64)
    n, d = X_int.shape
    if blocks is not None:
        num_pos = len(blocks)
        bv_i, bv_f = [], []
        for s, e in blocks:
            ip = np.unique(X_int[:, s:e], axis=0)
            fp = np.empty_like(ip, dtype=np.float32)
            for pi, pat in enumerate(ip):
                fp[pi] = X_off[np.argmax(np.all(X_int[:, s:e] == pat, axis=1)), s:e]
            bv_i.append(ip)
            bv_f.append(fp)

        def mutate(seq, nm):
            seq = seq.copy()
            nm = min(nm, num_pos)
            for pos in rng.choice(num_pos, size=nm, replace=False):
                s, e = blocks[pos]
                cur = np.rint(seq[s:e]).astype(np.int64)
                alt = np.where(np.any(bv_i[pos] != cur, axis=1))[0]
                if len(alt) > 0:
                    seq[s:e] = bv_f[pos][alt[rng.randint(0, len(alt))]]
            return seq
        return mutate, num_pos
    else:
        vocab = [np.unique(X_int[:, j]) for j in range(d)]

        def mutate(seq, nm):
            seq = np.rint(seq).astype(np.int64).copy()
            nm = min(nm, d)
            for pos in rng.choice(d, size=nm, replace=False):
                alt = vocab[pos][vocab[pos] != seq[pos]]
                if len(alt) > 0:
                    seq[pos] = int(rng.choice(alt))
            return seq.astype(np.float32)
        return mutate, d


# ═══════════════════════════════════════════════════════════
# Stage 1: Global Exploration
# ═══════════════════════════════════════════════════════════

def build_global_continuous(teacher, X_off, target, seed, margin,
                            batch_size, maximize,
                            pool_mult=4.0, pool_max=60000,
                            pred_w=0.7, if_w=0.3, power=2.5,
                            if_contam=0.1,
                            sampling_method="sobol",
                            refine_ratio=0.0, refine_scale=0.05):
    rng = np.random.RandomState(seed)
    n, d = X_off.shape
    pool_size = int(min(max(target, round(target * pool_mult)), pool_max))
    x_min, x_max = X_off.min(axis=0), X_off.max(axis=0)
    span = np.maximum(x_max - x_min, 1e-8)
    lo = (x_min - margin * span).astype(np.float32)
    hi = (x_max + margin * span).astype(np.float32)

    if sampling_method == "sobol":
        x_pool = _sample_sobol(lo, hi, pool_size, d, seed)
    else:
        x_pool = rng.uniform(low=lo, high=hi, size=(pool_size, d)).astype(np.float32)

    y_pred, _ = teacher.predict(x_pool, batch_size=batch_size)
    pred_01 = _normalize_01(y_pred if maximize else -y_pred)

    iso = IsolationForest(contamination=float(if_contam), random_state=seed, n_jobs=-1)
    iso.fit(X_off)
    norm_01 = _normalize_01(iso.decision_function(x_pool).astype(np.float32))

    wp, wi = max(0.0, pred_w), max(0.0, if_w)
    if wp + wi <= 0:
        wp, wi = 1.0, 1.0
    score = np.power(np.clip((wp * pred_01 + wi * norm_01) / (wp + wi), 1e-6, 1.0),
                     max(0.1, power))

    sel = min(target, len(x_pool))
    top = np.argsort(score)[-sel:]
    X_best = x_pool[top].astype(np.float32)

    # Local refinement around selected candidates
    n_refined = 0
    if refine_ratio > 0:
        n_refine_per = max(1, int(round(refine_ratio)))
        rng2 = np.random.RandomState(seed + 999)
        sigma = (refine_scale * span).astype(np.float32)
        refined_parts = []
        for i in range(len(X_best)):
            noise = rng2.normal(0, sigma, size=(n_refine_per, d)).astype(np.float32)
            refined_parts.append(X_best[i] + noise)
        X_refined = np.vstack(refined_parts).astype(np.float32)
        X_refined = np.clip(X_refined, lo, hi)
        X_best = np.vstack([X_best, X_refined]).astype(np.float32)
        n_refined = len(X_refined)

    return X_best, {"global_pool": pool_size, "global_sel": sel,
                     "global_sampling": sampling_method,
                     "global_refine_ratio": refine_ratio,
                     "global_n_refined": n_refined}


def build_global_discrete(teacher, X_off, y_off, target, seed,
                          mut_rate=0.12, max_mut=2,
                          batch_size=1024, maximize=True,
                          pool_mult=4.0, pool_max=60000,
                          pred_w=0.7, if_w=0.3, power=2.5,
                          if_contam=0.1):
    rng = np.random.RandomState(seed)
    n, d = X_off.shape
    blocks = _detect_block_structure(X_off)
    mutate, num_pos = _make_discrete_mutator(X_off, rng, blocks)

    if teacher is not None and blocks is not None:
        pool_size = int(min(max(target, round(target * pool_mult)), pool_max))
        X_int = np.rint(X_off).astype(np.int64)
        bv_f = []
        for s, e in blocks:
            ip = np.unique(X_int[:, s:e], axis=0)
            fp = np.empty_like(ip, dtype=np.float32)
            for pi, pat in enumerate(ip):
                fp[pi] = X_off[np.argmax(np.all(X_int[:, s:e] == pat, axis=1)), s:e]
            bv_f.append(fp)
        x_pool = np.empty((pool_size, d), dtype=np.float32)
        for pi, (s, e) in enumerate(blocks):
            chosen = rng.randint(0, len(bv_f[pi]), size=pool_size)
            x_pool[:, s:e] = bv_f[pi][chosen]

        y_pred, _ = teacher.predict(x_pool, batch_size=batch_size)
        pred_01 = _normalize_01(y_pred if maximize else -y_pred)

        iso = IsolationForest(contamination=float(if_contam), random_state=seed, n_jobs=-1)
        iso.fit(X_off)
        norm_01 = _normalize_01(iso.decision_function(x_pool).astype(np.float32))

        wp, wi = max(0.0, pred_w), max(0.0, if_w)
        if wp + wi <= 0:
            wp, wi = 1.0, 1.0
        score = np.power(np.clip((wp * pred_01 + wi * norm_01) / (wp + wi), 1e-6, 1.0),
                         max(0.1, power))

        n_seeds = min(max(target, int(len(x_pool) * 0.35)), len(x_pool))
        top_idx = np.argsort(score)[-n_seeds:]
        x_seeds = x_pool[top_idx]

        cands = list(x_seeds)
        for _ in range(target * 2):
            anc = x_seeds[rng.randint(0, len(x_seeds))]
            nm = 1 if rng.rand() < 0.7 else 2
            if mut_rate > 0 and rng.rand() < mut_rate:
                nm = min(num_pos, nm + 1)
            cands.append(mutate(anc, min(nm, max_mut)))
        cands = np.asarray(cands, dtype=np.float32)
        _, ui = np.unique(np.rint(cands).astype(np.int64), axis=0, return_index=True)
        xu = cands[ui]
        if len(xu) >= target:
            xu = xu[rng.choice(len(xu), size=target, replace=False)]

        print(f"[Global-Discrete] pool={pool_size}, seeds={n_seeds}, "
              f"pred_top={y_pred[top_idx].max():.4f}, dedup={len(xu)}")
        return xu, {"global_pool": pool_size, "global_seeds": n_seeds,
                     "global_teacher": True, "global_dedup": len(xu),
                     "global_block": True}

    # Fallback: no teacher or no block structure
    cands = []
    for _ in range(target * 2):
        anc = X_off[rng.randint(0, n)]
        nm = 1 if rng.rand() < 0.7 else 2
        if mut_rate > 0 and rng.rand() < mut_rate:
            nm = min(num_pos, nm + 1)
        cands.append(mutate(anc, min(nm, max_mut)))
    cands = np.asarray(cands, dtype=np.float32)
    _, ui = np.unique(np.rint(cands).astype(np.int64), axis=0, return_index=True)
    xu = cands[ui]
    if len(xu) >= target:
        xu = xu[rng.choice(len(xu), size=target, replace=False)]
    return xu, {"global_pool": 0, "global_seeds": 0, "global_teacher": False}


# ═══════════════════════════════════════════════════════════
# Stage 2: Local Augmentation
# ═══════════════════════════════════════════════════════════

def build_local_continuous(X_off, y_off, target, seed, margin,
                           sel_ratio=0.35, cluster_noise=0.35, knn_k=5):
    """Density-aware perturbation around top offline seeds."""
    rng = np.random.RandomState(seed + 100)
    n, d = X_off.shape

    sel_n = int(max(2, min(n, round(n * sel_ratio))))
    top_idx = np.argsort(y_off)[-sel_n:]
    x_seed = X_off[top_idx].astype(np.float32)

    knn_k_eff = min(knn_k, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=knn_k_eff, metric="euclidean")
    nn.fit(X_off)
    dist_seed, _ = nn.kneighbors(x_seed, return_distance=True)
    density_dist = np.maximum(dist_seed.mean(axis=1).astype(np.float32), 1e-8)

    # Allocation proportional to sparsity
    alloc_weights = density_dist.copy()
    alloc_weights /= max(1e-12, alloc_weights.sum())
    alloc = np.round(alloc_weights * target).astype(int)
    alloc = np.maximum(alloc, 0)
    diff = target - alloc.sum()
    if diff != 0:
        order = np.argsort(density_dist)[::-1]
        for i in range(abs(diff)):
            alloc[order[i % len(order)]] += 1 if diff > 0 else -1
        alloc = np.maximum(alloc, 0)

    gspan = np.maximum(
        np.quantile(X_off, 0.99, axis=0) - np.quantile(X_off, 0.01, axis=0), 1e-8
    ).astype(np.float32)

    parts = []
    for i in range(sel_n):
        m = int(alloc[i])
        if m <= 0:
            continue
        sigma = max(float(density_dist[i]) * float(cluster_noise),
                    1e-4 * float(np.mean(gspan)))
        noise = rng.normal(0.0, sigma, size=(m, d)).astype(np.float32)
        parts.append(x_seed[i] + noise)

    x_local = np.vstack(parts).astype(np.float32) if parts else np.empty((0, d), dtype=np.float32)

    clip_lo = np.quantile(X_off, 0.01, axis=0) - margin * gspan
    clip_hi = np.quantile(X_off, 0.99, axis=0) + margin * gspan
    x_local = np.clip(x_local, clip_lo, clip_hi).astype(np.float32)

    return x_local, {"local_seeds": sel_n, "local_gen": len(x_local),
                      "local_mode": "density_offline_top"}


def build_local_discrete(X_off, y_off, target, seed, sel_ratio=0.35,
                         mut_rate=0.12, max_mut=2, maximize=True):
    rng = np.random.RandomState(seed + 100)
    n, d = X_off.shape
    sel_n = int(max(2, min(n, round(n * sel_ratio))))
    if maximize:
        top = np.argpartition(y_off, -sel_n)[-sel_n:]
    else:
        top = np.argpartition(y_off, sel_n - 1)[:sel_n]
    x_seed = X_off[top].astype(np.float32)

    blocks = _detect_block_structure(X_off)
    mutate, num_pos = _make_discrete_mutator(X_off, rng, blocks)
    cands = []
    for _ in range(target * 2):
        anc = x_seed[rng.randint(0, len(x_seed))]
        nm = 1 if rng.rand() < 0.85 else 2
        if mut_rate > 0 and rng.rand() < mut_rate:
            nm = min(num_pos, nm + 1)
        cands.append(mutate(anc, min(nm, max_mut)))
    cands = np.asarray(cands, dtype=np.float32)
    _, ui = np.unique(np.rint(cands).astype(np.int64), axis=0, return_index=True)
    xu = cands[ui]
    if len(xu) >= target:
        xu = xu[rng.choice(len(xu), size=target, replace=False)]
    return xu, {"local_seeds": sel_n, "local_dedup": len(xu),
                "local_block": blocks is not None}


# ═══════════════════════════════════════════════════════════
# OOD Filters
# ═══════════════════════════════════════════════════════════

def ood_filter_continuous(X_cand, X_off, q=0.95, scale=1.2, relax=2.5, min_keep=0.8):
    nn = NearestNeighbors(n_neighbors=min(5, len(X_off)), metric="euclidean")
    nn.fit(X_off)
    ref_d = nn.kneighbors(X_off, return_distance=True)[0][:, -1]
    radius = float(np.quantile(ref_d, q)) * scale * relax
    radius = max(radius, 1e-6)
    cd = nn.kneighbors(X_cand, return_distance=True)[0][:, 0]
    mask = cd <= radius
    mk = max(2, int(round(len(X_cand) * min_keep)))
    if mask.sum() < mk:
        mask[:] = False
        mask[np.argsort(cd)[:mk]] = True
    return X_cand[mask], mask, {"ood_radius": radius,
                                 "n_before_ood": len(X_cand),
                                 "n_after_ood": int(mask.sum())}


def ood_filter_discrete(X_cand, X_off, q=0.95, scale=1.5, min_keep=2):
    X_ci = np.rint(X_cand).astype(np.int64)
    X_oi = np.rint(X_off).astype(np.int64)
    min_h = np.full(len(X_ci), X_ci.shape[1], dtype=np.float32)
    bs = 500
    for i in range(0, len(X_ci), bs):
        d = np.sum(X_ci[i:i + bs, None, :] != X_oi[None, :, :], axis=2).astype(np.float32)
        min_h[i:i + bs] = d.min(axis=1)
    rng = np.random.RandomState(42)
    sub = min(1000, len(X_oi))
    si = rng.choice(len(X_oi), size=sub, replace=False)
    rd = []
    for i in range(0, sub, bs):
        d = np.sum(X_oi[si[i:i + bs], None, :] != X_oi[None, :, :], axis=2).astype(np.float32)
        for j, idx in enumerate(range(i, min(i + bs, sub))):
            d[j, si[idx]] = 9999
        rd.append(d.min(axis=1))
    ref_h = np.concatenate(rd)
    thresh = max(1.0, float(np.quantile(ref_h, q)) * scale)
    mask = min_h <= thresh
    mk = max(min_keep, int(round(len(X_cand) * 0.8)))
    if mask.sum() < mk:
        mask[:] = False
        mask[np.argsort(min_h)[:mk]] = True
    return X_cand[mask], mask, {"ood_hamming_thresh": thresh,
                                 "n_before_ood": len(X_cand),
                                 "n_after_ood": int(mask.sum())}


# ═══════════════════════════════════════════════════════════
# Quality & Uncertainty Filters
# ═══════════════════════════════════════════════════════════

def filter_by_pred_quantile(X, ys, yu, y_off, min_q, maximize):
    if min_q <= 0:
        return X, ys, yu, {}
    thresh = float(np.quantile(y_off, min_q))
    mask = ys >= thresh if maximize else ys <= thresh
    if mask.sum() < 2:
        return X, ys, yu, {"pred_thresh": thresh}
    return X[mask], ys[mask], yu[mask], {"pred_thresh": thresh,
                                          "pred_removed": int(len(X) - mask.sum())}


def filter_by_uncertainty(X, ys, yu, drop):
    drop = float(min(max(drop, 0.0), 0.95))
    n = len(X)
    if drop <= 0 or n == 0:
        return X, ys, yu, {"n_before_unc": n, "n_after_unc": n}
    keep_n = max(2, int(round(n * (1.0 - drop))))
    ki = np.argsort(yu)[:keep_n]
    return X[ki], ys[ki], yu[ki], {"unc_drop": drop,
                                    "n_before_unc": n,
                                    "n_after_unc": len(ki)}


# ═══════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════

def build_distill_set(teacher, X_off, y_off, target, args, is_discrete):
    """Two-stage distillation set construction.

    Returns:
        X_dis: distill set designs
        y_soft: teacher mean predictions
        y_unc: teacher uncertainty estimates
        stats: dict of statistics
    """
    n_global = int(round(target * float(args.global_ratio)))
    n_local = target - n_global
    discrete_os = max(1, int(getattr(args, 'discrete_oversample', 3))) if is_discrete else 1
    n_global_gen = n_global * discrete_os
    n_local_gen = n_local * discrete_os

    print(f"[Distill] target={target}, global={n_global} ({args.global_ratio:.0%}), local={n_local}"
          + (f", discrete_oversample={discrete_os}x" if discrete_os > 1 else ""))

    if is_discrete:
        Xg, gs = build_global_discrete(
            teacher, X_off, y_off, n_global_gen, args.seed,
            mut_rate=args.mutation_rate,
            max_mut=args.max_mutations_per_seq,
            batch_size=args.teacher_predict_batch_size,
            maximize=args.maximize,
            pool_mult=args.priority_pool_multiplier,
            pool_max=args.priority_pool_max,
            pred_w=args.score_pred_weight,
            if_w=args.score_if_weight,
            power=args.priority_power,
            if_contam=args.if_contamination,
        )
        Xl, ls = build_local_discrete(
            X_off, y_off, n_local_gen, args.seed,
            args.priority_select_ratio,
            args.mutation_rate, args.max_mutations_per_seq, args.maximize)
    else:
        Xg, gs = build_global_continuous(
            teacher, X_off, n_global, args.seed, args.margin,
            args.teacher_predict_batch_size, args.maximize,
            args.priority_pool_multiplier, args.priority_pool_max,
            args.score_pred_weight, args.score_if_weight,
            args.priority_power, args.if_contamination,
            sampling_method=args.global_sampling_method,
            refine_ratio=float(getattr(args, 'global_refine_ratio', 0.0)),
            refine_scale=float(getattr(args, 'global_refine_scale', 0.05)),
        )
        Xl, ls = build_local_continuous(
            X_off, y_off, n_local, args.seed, args.margin,
            args.priority_select_ratio, args.cluster_noise_scale)

    origin = np.concatenate([np.zeros(len(Xg), dtype=np.int32),
                              np.ones(len(Xl), dtype=np.int32)])
    Xm = np.vstack([Xg, Xl]).astype(np.float32) if len(Xg) + len(Xl) > 0 else np.empty((0, X_off.shape[1]), dtype=np.float32)
    print(f"[Distill] Pre-OOD: global={len(Xg)}, local={len(Xl)}, total={len(Xm)}")

    skip_ood = bool(getattr(args, 'skip_ood_filter', False))
    if skip_ood:
        Xk, mask, ok = Xm, np.ones(len(Xm), dtype=bool), origin
        os_ = {"ood_mode": "skipped"}
        print(f"[Distill] OOD skipped, total={len(Xk)}")
    elif is_discrete:
        Xk, mask, os_ = ood_filter_discrete(
            Xm, X_off, args.ood_hamming_quantile, args.ood_hamming_scale)
        ok = origin[mask]
    else:
        Xk, mask, os_ = ood_filter_continuous(
            Xm, X_off, args.structured_ood_quantile, args.structured_ood_scale,
            args.ood_relax_scale, args.ood_min_keep_ratio)
        ok = origin[mask]
    print(f"[Distill] Post-OOD: global={int((ok == 0).sum())}, local={int((ok == 1).sum())}, total={len(Xk)}")

    # Teacher-guided selection for discrete (score + pick top)
    if len(Xk) > target:
        if is_discrete and discrete_os > 1 and teacher is not None:
            print(f"[Distill] Teacher-guided selection: scoring {len(Xk)} candidates...")
            y_pred, y_unc = teacher.predict(Xk, batch_size=int(args.teacher_predict_batch_size))
            pred_01 = _normalize_01(y_pred if args.maximize else -y_pred)
            unc_01 = _normalize_01(-y_unc)
            combined = 0.8 * pred_01 + 0.2 * unc_01
            ki = np.argsort(combined)[-target:]
            Xk, ok = Xk[ki], ok[ki]
        else:
            rng = np.random.RandomState(args.seed + 42)
            ki = rng.choice(len(Xk), size=target, replace=False)
            Xk, ok = Xk[ki], ok[ki]

    # Teacher labeling
    y_soft, y_unc = teacher.predict(Xk, batch_size=int(args.teacher_predict_batch_size))

    # Quality gate: only keep candidates teacher deems promising
    if args.distill_min_pred_quantile > 0:
        Xk, y_soft, y_unc, _ = filter_by_pred_quantile(
            Xk, y_soft, y_unc, y_off, args.distill_min_pred_quantile, args.maximize)

    # Uncertainty filter (hard drop — used when uncertainty_weight_beta=0)
    Xk, y_soft, y_unc, u_stats = filter_by_uncertainty(
        Xk, y_soft, y_unc, args.uncertainty_drop_ratio)

    print(f"[Distill] Final: {len(Xk)} samples")

    stats = {
        "pipeline": "v2", "target": target,
        "global_ratio": float(args.global_ratio),
        "discrete_oversample": discrete_os,
        "n_global_gen": len(Xg), "n_local_gen": len(Xl),
        "n_global_ood": int((ok == 0).sum()),
        "n_local_ood": int((ok == 1).sum()),
        "n_final": len(Xk),
    }
    stats.update({f"g_{k}": v for k, v in gs.items()})
    stats.update({f"l_{k}": v for k, v in ls.items()})
    stats.update(os_)
    stats.update(u_stats)

    return Xk, y_soft, y_unc, stats
