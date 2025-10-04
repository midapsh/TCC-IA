# hierarchical_bsts.py
import torch, pyro, pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


def fourier_design(T, periods, K_per_period, device):
    """Create block of cos/sin seasonal states as part of H and F."""
    # We'll use seasonal states as part of x_t with stable rotations.
    blocks = []
    H_blocks = []
    for P, K in zip(periods, K_per_period):
        omegas = 2 * torch.pi * torch.arange(1, K + 1, device=device) / P
        # Each harmonic contributes 2-dim state (cos,sin) with rotation
        # F_block is block-diagonal of K rotation matrices
        F_block = torch.block_diag(
            *[
                torch.tensor(
                    [
                        [torch.cos(omega), -torch.sin(omega)],
                        [torch.sin(omega), torch.cos(omega)],
                    ],
                    device=device,
                )
                for omega in omegas
            ]
        )
        H_block = torch.zeros(1, 2 * K, device=device)
        # Observe only the first coordinate of each pair (cosine weight)
        for j in range(K):
            H_block[0, 2 * j] = 1.0
        blocks.append(F_block)
        H_blocks.append(H_block)
    return torch.block_diag(*blocks) if blocks else None, (
        torch.cat(H_blocks, dim=-1) if H_blocks else None
    )


class HierBSTS(PyroModule):
    def __init__(self, p, state_dim_extra=0, groups=1, device="cpu"):
        """
        p: number of exogenous covariates
        state_dim_extra: extra state dims (e.g. seasonal pairs). We always include 1 for local level.
        """
        super().__init__()
        self.device = device
        self.p = p
        self.groups = groups
        self.state_dim = 1 + state_dim_extra  # 1 for local level
        # Hyperpriors shared per group
        self.mu_beta = PyroSample(
            dist.Normal(torch.zeros(groups, p), 5 * torch.ones(groups, p)).to_event(2)
        )
        self.sigma_beta = PyroSample(
            dist.HalfCauchy(0.5 * torch.ones(groups, p)).to_event(2)
        )
        self.level_scale_group = PyroSample(dist.HalfCauchy(0.3 * torch.ones(groups)))
        self.obs_scale_group = PyroSample(dist.HalfCauchy(0.3 * torch.ones(groups)))

    def model(self, y_pad, X_pad, mask, group_idx, F, H, Q_season=None):
        """
        y_pad: (S, T_max, 1)
        X_pad: (S, T_max, p)
        mask:  (S, T_max) bool
        group_idx: (S,) long
        F: (state_dim, state_dim) transition (includes seasonal rotation and level random-walk)
        H: (1, state_dim) observation on state part
        Q_season: (state_dim, state_dim) process noise for seasonal block (optional)
        """
        S, T, _ = y_pad.shape
        pyro.module("hier", self)
        with pyro.plate("groups", self.groups):
            mu_beta = pyro.sample("mu_beta", dist.Delta(self.mu_beta))
            sigma_beta = pyro.sample("sigma_beta", dist.Delta(self.sigma_beta))
            level_scale_g = pyro.sample(
                "level_scale_g", dist.Delta(self.level_scale_group)
            )
            obs_scale_g = pyro.sample("obs_scale_g", dist.Delta(self.obs_scale_group))

        # Station-level parameters
        with pyro.plate("stations", S):
            g = group_idx
            beta = pyro.sample(
                "beta", dist.Normal(mu_beta[g], sigma_beta[g]).to_event(1)
            )  # (S,p)
            level_scale = pyro.sample("level_scale", dist.HalfCauchy(level_scale_g[g]))
            obs_scale = pyro.sample("obs_scale", dist.HalfCauchy(obs_scale_g[g]))

            # Process noise Q: level random walk on first state
            Q = torch.zeros(self.state_dim, self.state_dim, device=self.device)
            Q[0, 0] = level_scale**2
            if Q_season is not None:
                Q = Q + Q_season  # seasonal diffusion

            R = obs_scale**2  # scalar

            # Initial state prior
            m0 = pyro.sample(
                "m0",
                dist.Normal(
                    torch.zeros(S, self.state_dim, device=self.device),
                    5 * torch.ones(S, self.state_dim, device=self.device),
                ).to_event(1),
            )
            P0 = torch.diag_embed(5 * torch.ones(S, self.state_dim, device=self.device))

            # Kalman forward sample likelihood (vectorized over stations)
            x = m0
            for t in pyro.markov(range(T)):
                # state evolve
                x = pyro.sample(
                    f"x_{t}",
                    dist.MultivariateNormal(
                        (F @ x.unsqueeze(-1)).squeeze(-1),
                        covariance_matrix=Q.expand(S, -1, -1),
                    ),
                )
                # observation mean
                mean_y = (H @ x.unsqueeze(-1)).squeeze(-1) + (
                    X_pad[:, t, :] * beta
                ).sum(
                    -1, keepdim=True
                )  # (S,1)
                # handle missing with masked likelihood
                obs = y_pad[:, t, :]
                # Only condition on observed entries
                mask_t = mask[:, t]
                pyro.sample(
                    f"y_{t}",
                    dist.Normal(mean_y[mask_t].squeeze(-1), obs_scale[mask_t]),
                    obs=obs[mask_t].squeeze(-1),
                )
