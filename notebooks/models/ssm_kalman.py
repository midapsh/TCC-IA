# ssm_kalman.py
import torch


def kf_filter(y, F, H, Q, R, m0, P0, mask=None):
    """
    Standard Kalman filter for linear-Gaussian SSM:
      x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
      y_t = H x_t     + v_t,  v_t ~ N(0, R)
    y: (T, obs_dim) with possible NaNs; mask: (T,) True when observed
    Returns filtered means/covs and one-step-ahead predictive means/covs.
    """
    T, obs_dim = y.shape
    state_dim = m0.shape[-1]
    m = torch.zeros(T, state_dim, device=y.device)
    P = torch.zeros(T, state_dim, state_dim, device=y.device)
    mp = m0
    Pp = P0
    I = torch.eye(state_dim, device=y.device)
    if mask is None:
        mask = ~torch.isnan(y).any(dim=-1)

    for t in range(T):
        # Predict
        mp = F @ mp
        Pp = F @ Pp @ F.transpose(-1, -2) + Q

        if mask[t]:
            yt = y[t].unsqueeze(-1)  # (obs_dim,1)
            S = H @ Pp @ H.transpose(-1, -2) + R
            K = (
                Pp
                @ H.transpose(-1, -2)
                @ torch.linalg.solve(S, torch.eye(obs_dim, device=y.device))
            )
            innov = yt - (H @ mp).unsqueeze(-1)
            mp = (mp.unsqueeze(-1) + K @ innov).squeeze(-1)
            Pp = (I - K @ H) @ Pp

        m[t] = mp
        P[t] = Pp
    return m, P


def rts_smoother(m, P, F, Q):
    """
    Rauch–Tung–Striebel smoother.
    Inputs are filtered m,P and system matrices F,Q.
    Returns smoothed means/covs.
    """
    T, state_dim = m.shape
    ms = torch.zeros_like(m)
    Ps = torch.zeros_like(P)
    ms[-1] = m[-1]
    Ps[-1] = P[-1]
    I = torch.eye(state_dim, device=m.device)

    for t in reversed(range(T - 1)):
        P_pred = F @ P[t] @ F.transpose(-1, -2) + Q
        C = (
            P[t]
            @ F.transpose(-1, -2)
            @ torch.linalg.solve(P_pred, torch.eye(state_dim, device=m.device))
        )
        ms[t] = m[t] + C @ (ms[t + 1] - F @ m[t])
        Ps[t] = P[t] + C @ (Ps[t + 1] - P_pred) @ C.transpose(-1, -2)
    return ms, Ps
