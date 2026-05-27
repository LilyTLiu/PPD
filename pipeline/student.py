"""MLP student training with uncertainty-weighted listwise distillation."""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split


# ── MLP Architecture ──────────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ── ListNet Loss ──────────────────────────────────────────
def listnet_loss_per_list(y_pred, y_true):
    return -torch.sum(
        F.softmax(y_true, dim=1) * torch.log(F.softmax(y_pred, dim=1).clamp(min=1e-10)),
        dim=1,
    )


def listnet_loss_per_list_weighted(y_pred, y_true, u_weight):
    """Sample-level uncertainty-weighted ListNet.

    u_weight: (B, L) weights in [0, 1], where low-uncertainty samples get higher weight.
    """
    return -torch.sum(
        u_weight * F.softmax(y_true, dim=1) * torch.log(F.softmax(y_pred, dim=1).clamp(min=1e-10)),
        dim=1,
    )


def rankcosine_loss_per_list(y_pred, y_true):
    pc = y_pred - y_pred.mean(dim=1, keepdim=True)
    tc = y_true - y_true.mean(dim=1, keepdim=True)
    return 1.0 - torch.sum(pc * tc, dim=1) / (
        torch.sqrt(torch.sum(pc ** 2, dim=1)) * torch.sqrt(torch.sum(tc ** 2, dim=1)) + 1e-8
    )


def get_listwise_loss_fn(name):
    if name == "listnet":
        return listnet_loss_per_list
    if name == "rankcosine":
        return rankcosine_loss_per_list
    raise ValueError(f"Unknown loss: {name}")


def _compute_uncertainty_weights(u, beta):
    """w_i = 1 / (1 + beta * sigma_i_norm), per-list min-max normalized."""
    if beta <= 0:
        return torch.ones_like(u)
    u_min = u.min(dim=1, keepdim=True).values
    u_max = u.max(dim=1, keepdim=True).values
    denom = (u_max - u_min).clamp(min=1e-8)
    u_norm = (u - u_min) / denom
    return 1.0 / (1.0 + beta * u_norm)


# ── Data Loading ──────────────────────────────────────────
def create_special_dataset_fast_unique(x, y, list_length, num_samples, seed):
    n, m = len(x), min(int(list_length), len(x))
    rng = np.random.RandomState(seed)
    if n > m * 10:
        idx = rng.randint(0, n, size=(int(num_samples), m))
    else:
        idx = np.stack([rng.permutation(n)[:m] for _ in range(int(num_samples))])
    return x[idx].astype(np.float32), y[idx].astype(np.float32)


def build_list_loaders(x_l, y_l, batch_size, val_split, drop_last, seed):
    ds = TensorDataset(torch.from_numpy(x_l).float(), torch.from_numpy(y_l).float())
    vs = int(val_split * len(ds))
    ts = len(ds) - vs
    if ts <= 0:
        ts, vs = len(ds), 0
    if vs > 0:
        gen = torch.Generator().manual_seed(seed)
        tr, vl = random_split(ds, [ts, vs], generator=gen)
        return (DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=drop_last),
                DataLoader(vl, batch_size=batch_size, shuffle=False, drop_last=drop_last))
    return (DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=drop_last),
            None)


def build_distill_list_loaders(x_l, y_l, u_l, batch_size, val_split, drop_last, seed):
    ds = TensorDataset(torch.from_numpy(x_l).float(),
                       torch.from_numpy(y_l).float(),
                       torch.from_numpy(u_l).float())
    vs = int(val_split * len(ds))
    ts = len(ds) - vs
    if ts <= 0:
        ts, vs = len(ds), 0
    if vs > 0:
        gen = torch.Generator().manual_seed(seed)
        tr, vl = random_split(ds, [ts, vs], generator=gen)
        return (DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=drop_last),
                DataLoader(vl, batch_size=batch_size, shuffle=False, drop_last=drop_last))
    return (DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=drop_last),
            None)


def forward_listwise(model, x):
    B, L, D = x.shape
    return model(x.reshape(B * L, D)).reshape(B, L)


# ── Training ──────────────────────────────────────────────
def train_student(model, X_off, y_off, X_dis, y_dis, u_dis, args, device):
    """Train MLP student jointly on offline data + distillation set.

    Supports:
    - Pure offline training (alpha=0 or no distill set)
    - Uncertainty-weighted distillation (uw_beta > 0)
    - Hard uncertainty filtering (uw_beta=0, uncertainty_drop_ratio > 0)
    """
    loss_fn = get_listwise_loss_fn(args.list_loss)
    ll, ns = int(args.list_length), int(args.list_num_samples)
    beta = float(getattr(args, 'uncertainty_weight_beta', 0.0))
    use_distill = len(X_dis) > 0
    use_weighting = beta > 0 and use_distill

    xo_l, yo_l = create_special_dataset_fast_unique(X_off, y_off, ll, ns, args.seed)
    off_tr, off_vl = build_list_loaders(xo_l, yo_l, args.list_batch_size,
                                         args.validation_split, args.drop_last, args.seed)

    dis_tr, dis_vl = None, None
    if use_distill:
        xd_l, yd_l = create_special_dataset_fast_unique(X_dis, y_dis, ll, ns, args.seed + 7)
        _, ud_l = create_special_dataset_fast_unique(X_dis, u_dis, ll, ns, args.seed + 7)
        dis_tr, dis_vl = build_distill_list_loaders(
            xd_l, yd_l, ud_l, args.list_batch_size,
            args.validation_split, args.drop_last, args.seed + 17)

    opt = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_s = copy.deepcopy(model.state_dict())
    best_v = float("inf")
    ts = min(len(off_tr), len(dis_tr)) if dis_tr else len(off_tr)
    vs = 0
    if off_vl and (not use_distill or dis_vl):
        vs = min(len(off_vl), len(dis_vl)) if dis_vl else len(off_vl)

    print(f"[Train] loss={args.list_loss}, ll={ll}, ns={ns}, bs={args.list_batch_size}, "
          f"ep={args.epochs}" + (f", uw_beta={beta}" if use_weighting else ""))

    for ep in range(args.epochs):
        model.train()
        tl, tr_, td_ = 0.0, 0.0, 0.0
        if use_distill and dis_tr:
            for (xo, yo), (xd, yd, ud_b) in zip(off_tr, dis_tr):
                xo = xo.to(device, dtype=torch.float32)
                yo = yo.to(device, dtype=torch.float32)
                xd = xd.to(device, dtype=torch.float32)
                yd = yd.to(device, dtype=torch.float32)
                ud_b = ud_b.to(device, dtype=torch.float32)
                po = forward_listwise(model, xo)
                pd = forward_listwise(model, xd)
                lo = loss_fn(po, yo)
                if use_weighting:
                    w = _compute_uncertainty_weights(ud_b, beta)
                    ld = listnet_loss_per_list_weighted(pd, yd, w)
                else:
                    ld = loss_fn(pd, yd)
                alpha = float(getattr(args, 'alpha', 0.1))
                ram = lo.mean()
                dis = (alpha * ld).mean()
                loss = ram + dis
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item()
                tr_ += ram.item()
                td_ += dis.item()
        else:
            for xo, yo in off_tr:
                xo = xo.to(device, dtype=torch.float32)
                yo = yo.to(device, dtype=torch.float32)
                loss = loss_fn(forward_listwise(model, xo), yo).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item()
                tr_ += loss.item()

        model.eval()
        vl = float("nan")
        if vs > 0:
            vls = 0.0
            with torch.no_grad():
                if use_distill and dis_vl:
                    for (xo, yo), (xd, yd, ud_b) in zip(off_vl, dis_vl):
                        xo = xo.to(device, dtype=torch.float32)
                        yo = yo.to(device, dtype=torch.float32)
                        xd = xd.to(device, dtype=torch.float32)
                        yd = yd.to(device, dtype=torch.float32)
                        ud_b = ud_b.to(device, dtype=torch.float32)
                        lo = loss_fn(forward_listwise(model, xo), yo)
                        pd = forward_listwise(model, xd)
                        if use_weighting:
                            w = _compute_uncertainty_weights(ud_b, beta)
                            ld = listnet_loss_per_list_weighted(pd, yd, w)
                        else:
                            ld = loss_fn(pd, yd)
                        alpha = float(getattr(args, 'alpha', 0.1))
                        vls += (lo.mean() + (alpha * ld).mean()).item()
                else:
                    for xo, yo in off_vl:
                        xo = xo.to(device, dtype=torch.float32)
                        yo = yo.to(device, dtype=torch.float32)
                        vls += loss_fn(forward_listwise(model, xo), yo).mean().item()
            vl = vls / max(1, vs)

        avg_tl = tl / max(1, ts)
        if (ep + 1) % 10 == 1 or ep < 2 or ep == args.epochs - 1:
            print(f"Ep [{ep + 1}/{args.epochs}] tr={avg_tl:.6f} "
                  f"(ram={tr_ / max(1, ts):.6f} dis={td_ / max(1, ts):.6f}) vl={vl:.6f}")
        mon = avg_tl if np.isnan(vl) else vl
        if mon < best_v:
            best_v = mon
            best_s = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_s)
    return best_v, {"list_loss": args.list_loss, "epochs": args.epochs,
                    "train_steps": ts, "val_steps": vs,
                    "uw_beta": beta if use_weighting else 0.0}
