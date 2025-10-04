# sm_to_ssm.py
import torch
from ssm_kalman import kf_filter, rts_smoother


def ssm_from_celerite_terms(dt, terms):
    """
    terms: list of dicts with keys {'amp','decay','freq'}
    State is concatenation of [cos, sin] for each term.
    """
    J = len(terms)
    F_blocks, Q_blocks, H_blocks = [], [], []
    for term in terms:
        a, c, w = term["amp"], term["decay"], term["freq"]
        # Discrete-time rotation-damping
        phi = w * dt
        r = torch.exp(-c * dt)
        F = r * torch.tensor(
            [[torch.cos(phi), -torch.sin(phi)], [torch.sin(phi), torch.cos(phi)]],
            dtype=torch.float32,
        )
        # Stationary process noise for oscillator (approx):
        # Q = q * I (simple, practical choice); tune q using a and c
        q = (a**2) * (1 - r**2)
        Q = q * torch.eye(2)
        H = torch.tensor([[1.0, 0.0]])  # observe cosine coord
        F_blocks.append(F)
        Q_blocks.append(Q)
        H_blocks.append(H)

    F = torch.block_diag(*F_blocks) if J > 0 else torch.eye(1)
    Q = torch.block_diag(*Q_blocks) if J > 0 else torch.eye(1)
    H = torch.cat(H_blocks, dim=-1) if J > 0 else torch.ones(1, 1)
    return F, H, Q


def fast_sm_predict(y, dt, terms, exog=None, beta=None, obs_scale=0.1):
    """
    y: (T,) with NaNs ok. Optionally add exogenous mean: exog @ beta
    Returns smoothed mean and intervals via RTS on the SSM.
    """
    y = torch.tensor(y, dtype=torch.float32)
    mask = ~torch.isnan(y)
    y_obs = torch.where(mask, y, torch.zeros_like(y))
    if exog is not None and beta is not None:
        mean_exog = torch.tensor(exog, dtype=torch.float32) @ torch.tensor(
            beta, dtype=torch.float32
        )
        y_for_filter = (y_obs - mean_exog).unsqueeze(-1)
    else:
        y_for_filter = y_obs.unsqueeze(-1)

    F, H, Q = ssm_from_celerite_terms(dt, terms)
    R = (obs_scale**2) * torch.ones(1, 1)
    m0 = torch.zeros(F.shape[0])
    P0 = 10 * torch.eye(F.shape[0])

    m_filt, P_filt = kf_filter(y_for_filter, F, H, Q, R, m0, P0, mask=mask)
    m_smooth, P_smooth = rts_smoother(m_filt, P_filt, F, Q)
    y_sm = (H @ m_smooth.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    if exog is not None and beta is not None:
        y_sm = y_sm + mean_exog
    y_var = (H @ P_smooth @ H.transpose(-1, -2)).squeeze(-1).squeeze(-1) + obs_scale**2
    ci_lo = y_sm - 1.96 * torch.sqrt(y_var)
    ci_hi = y_sm + 1.96 * torch.sqrt(y_var)
    return y_sm, ci_lo, ci_hi
