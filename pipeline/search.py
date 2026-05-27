"""Gradient-based design optimization using the trained MLP student."""

import numpy as np
import torch
from torch.optim import Adam

from .utils import invert_standardizer, is_logit_encoded_discrete


def optimize_candidates(model, X_off, y_off, x_stats, args, device, is_discrete):
    """Run gradient ascent from top-K offline seeds.

    Returns optimized candidates in original (un-normalized) space.
    """
    lr = float(args.search_lr) if float(args.search_lr) > 0 else float(
        args.search_lr_discrete if is_discrete else args.search_lr_continuous)
    steps = int(args.search_steps) if int(args.search_steps) > 0 else int(
        args.search_steps_discrete if is_discrete else args.search_steps_continuous)
    k = min(max(0, int(args.num_solutions)), len(X_off))

    top_idx = np.argsort(y_off)[-k:] if args.maximize else np.argsort(y_off)[:k]
    x_init = torch.from_numpy(X_off[top_idx]).float().to(device)
    x_res = x_init.clone().detach().requires_grad_(True)
    opt = Adam([x_res], lr=lr)
    model.eval()

    for _ in range(steps):
        opt.zero_grad()
        pred = model(x_res).squeeze(-1)
        (-pred.sum() if args.maximize else pred.sum()).backward()
        opt.step()

    xnp = x_res.detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        pf = model(x_res).squeeze(-1).cpu().numpy()

    # Convert back to original space for saving
    cands_save = invert_standardizer(xnp, x_stats) if x_stats else xnp
    if is_discrete and not is_logit_encoded_discrete(args.task_name):
        cands_save = np.clip(np.rint(cands_save),
                             X_off.min(axis=0), X_off.max(axis=0)).astype(np.float32)

    return cands_save, pf, {"search_lr": lr, "search_steps": steps, "num_solutions": k}
