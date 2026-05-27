"""TabPFN V2.6 teacher with conservative context augmentation."""

import numpy as np

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


def augment_dataset(X, y, num_adversarial=100, num_innovative=50,
                    margin=0.1, random_state=None):
    """Generate synthetic adversarial and innovative samples for conservative context.

    Adversarial: low-density regions → low target values.
    Innovative:  high-density regions → high target values.
    """
    N, D = X.shape

    if num_adversarial <= 0 and num_innovative <= 0:
        return np.empty((0, D)), np.empty(0)

    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    range_X = X_max - X_min
    range_X[range_X == 0] = 1.0

    X_min_ext = X_min - margin * range_X
    X_max_ext = X_max + margin * range_X

    num_candidates = (num_adversarial + num_innovative) * 200
    rng = np.random.RandomState(random_state)

    # Latin Hypercube Sampling
    X_candidates = np.zeros((num_candidates, D))
    for d in range(D):
        intervals = np.arange(num_candidates) / num_candidates
        random_offsets = rng.uniform(0, 1 / num_candidates, size=num_candidates)
        perm = rng.permutation(num_candidates)
        X_candidates[:, d] = intervals + random_offsets
        X_candidates[:, d] = X_candidates[:, d][perm]
    X_candidates = X_min_ext + X_candidates * (X_max_ext - X_min_ext)

    from sklearn.ensemble import IsolationForest

    if N > 50000:
        idx = rng.choice(N, 50000, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X

    iso_forest = IsolationForest(contamination=0.1, random_state=random_state,
                                 n_estimators=500, max_samples=256)
    perm = rng.permutation(len(X_fit))
    iso_forest.fit(X_fit[perm])
    scores = iso_forest.score_samples(X_candidates)

    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = 1.0

    sorted_indices = np.argsort(scores)

    X_syn_parts, y_syn_parts = [], []

    if num_adversarial > 0:
        adv_idx = sorted_indices[:num_adversarial]
        X_syn_parts.append(X_candidates[adv_idx])
        y_syn_parts.append(np.full(num_adversarial, y_min - 0.5 * y_range))

    if num_innovative > 0:
        inn_idx = sorted_indices[-num_innovative:]
        X_syn_parts.append(X_candidates[inn_idx])
        y_syn_parts.append(np.full(num_innovative, y_max + 0.1 * y_range))

    if not X_syn_parts:
        return X, y

    X_syn = np.vstack(X_syn_parts)
    y_syn = np.concatenate(y_syn_parts)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


class TabPFNTeacher:
    """TabPFN V2.6 teacher for MBO distillation.

    Uses context-based inference with conservative synthetic augmentation.
    No fine-tuning — the teacher predicts via in-context learning.
    """

    def __init__(self, task_name, num_adversarial=0, num_innovative=0,
                 conservative_only=False, margin=0.1,
                 device="cuda", random_state=0):
        self.task_name = task_name
        self.num_adversarial = num_adversarial
        self.num_innovative = 0 if conservative_only else num_innovative
        self.margin = margin
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        n_synth = self.num_adversarial + self.num_innovative

        X_aug, y_aug = augment_dataset(
            X, y,
            num_adversarial=self.num_adversarial,
            num_innovative=self.num_innovative,
            margin=self.margin,
            random_state=self.random_state,
        )

        print(f"[Teacher] {len(X)} offline + {n_synth} synthetic = {len(X_aug)} total context")

        # Shuffle
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(len(X_aug))
        self.X_ = X_aug[idx]
        self.y_ = y_aug[idx]

        self.model_ = TabPFNRegressor.create_default_for_version(
            version=ModelVersion.V2_6,
            device=self.device,
            fit_mode="batched",
            differentiable_input=False,
            ignore_pretraining_limits=True,
            n_estimators=8,
            random_state=self.random_state,
        )
        # Switch to inference mode (preprocess context once, not batched training)
        self.model_.fit_mode = "fit_preprocessors"
        self.model_.fit(self.X_, self.y_)

    def predict(self, X, batch_size=4096, output_type="main",
                quantiles=(0.1, 0.9)):
        """Predict mean and uncertainty for query points."""
        preds, unc = [], []
        for i in range(0, len(X), batch_size):
            out = self.model_.predict(
                X[i:i + batch_size],
                output_type=output_type,
                quantiles=[float(quantiles[0]), float(quantiles[1])],
            )
            preds.append(np.asarray(out["mean"]).reshape(-1))
            q_lo = np.asarray(out["quantiles"][0]).reshape(-1)
            q_hi = np.asarray(out["quantiles"][1]).reshape(-1)
            unc.append(np.maximum(q_hi - q_lo, 0.0))
        return np.concatenate(preds).astype(np.float32), np.concatenate(unc).astype(np.float32)
