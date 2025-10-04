# train_hier_bsts.py
import torch, pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from torch.nn.utils.rnn import pad_sequence
from hierarchical_bsts import HierBSTS, fourier_design
from ssm_kalman import kf_filter, rts_smoother


def collate(stations):
    # stations = list of dicts: {'y': (T,), 'X': (T,p), 'g': int}
    ys = [torch.tensor(s["y"]).float().unsqueeze(-1) for s in stations]
    Xs = [torch.tensor(s["X"]).float() for s in stations]
    masks = [~torch.isnan(y.squeeze(-1)) for y in ys]
    y_pad = pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    X_pad = pad_sequence(Xs, batch_first=True, padding_value=0.0)
    mask_pad = pad_sequence(masks, batch_first=True, padding_value=False)
    group_idx = torch.tensor([s["g"] for s in stations], dtype=torch.long)
    return y_pad, X_pad, mask_pad, group_idx


def build_F_H_Q(device, periods=None, K=None, seasonal_scale=0.05):
    # Level random-walk: x_0,t = x_0,t-1 + noise
    # Implement via F = Identity on level; Q's (0,0) is learned per station.
    if periods:
        F_season, H_season = fourier_design(
            T=None, periods=periods, K_per_period=K, device=device
        )
        state_dim = 1 + F_season.shape[0]
        F = torch.eye(state_dim, device=device)
        F[1:, 1:] = F_season
        H = torch.zeros(1, state_dim, device=device)
        H[0, 0] = 1.0
        H[:, 1:] = H_season
        Q_season = torch.zeros(state_dim, state_dim, device=device)
        Q_season[1:, 1:] = seasonal_scale * torch.eye(F_season.shape[0], device=device)
    else:
        state_dim = 1
        F = torch.eye(1, device=device)
        H = torch.ones(1, 1, device=device)
        Q_season = None
    return F, H, Q_season, state_dim


def fit_hier_bsts(
    stations,
    p,
    groups,
    device="cpu",
    periods=[24.0, 24.0 * 365.25],
    K=[3, 3],
    seasonal_scale=0.01,
    steps=3000,
    lr=5e-3,
):
    y_pad, X_pad, mask, group_idx = collate(stations)
    y_pad, X_pad, mask = y_pad.to(device), X_pad.to(device), mask.to(device)
    F, H, Q_season, state_dim = build_F_H_Q(device, periods, K, seasonal_scale)

    model = HierBSTS(p=p, state_dim_extra=state_dim - 1, groups=groups, device=device)
    guide = AutoMultivariateNormal(model.model)
    optim = pyro.optim.ClippedAdam({"lr": lr})
    svi = SVI(model.model, guide, optim, loss=Trace_ELBO())

    pyro.clear_param_store()
    for s in range(steps):
        loss = svi.step(y_pad, X_pad, mask, group_idx, F, H, Q_season)
        if (s + 1) % 500 == 0:
            print(f"step {s+1}  ELBO: {-loss:.2f}")

    # Posterior samples for intervals + smoothing/denoising
    posterior = pyro.infer.Predictive(model.model, guide=guide, num_samples=200)
    samples = posterior(
        y_pad=None,
        X_pad=X_pad,
        mask=mask,
        group_idx=group_idx,
        F=F,
        H=H,
        Q_season=Q_season,
    )

    # Extract posterior draws of parameters for smoothing with classical RTS
    # (single draw shown; average over many draws for intervals)
    # Here we just demonstrate how you'd smooth with one draw:
    # You can loop over 200 draws, run KF/RTS, and take quantiles.
    return guide, (F, H, Q_season), (y_pad, X_pad, mask, group_idx)
