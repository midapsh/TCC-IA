# deep_ssm.py
import torch, torch.nn as nn, pyro, pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal


class TransitionNet(nn.Module):
    def __init__(self, z_dim, u_dim, h_dim=64):
        super().__init__()
        self.rnn = nn.GRU(z_dim + u_dim, h_dim, batch_first=True)
        self.to_loc = nn.Linear(h_dim, z_dim)
        self.to_scale = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

    def forward(self, z_prev, u_t, h_prev=None):
        x = torch.cat([z_prev, u_t], dim=-1).unsqueeze(1)
        h, h_n = self.rnn(x, h_prev)
        loc = self.to_loc(h.squeeze(1))
        scale = self.to_scale(h.squeeze(1)) + 1e-3
        return loc, scale, h_n


class EmissionNet(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim=1, h_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.loc_head = nn.Linear(h_dim, y_dim)
        self.scale_head = nn.Sequential(nn.Linear(h_dim, y_dim), nn.Softplus())

    def forward(self, z_t, x_t):
        h = self.net(torch.cat([z_t, x_t], dim=-1))
        loc = self.loc_head(h)
        scale = self.scale_head(h) + 1e-3
        return loc, scale


class DMM(PyroModule):
    def __init__(self, z_dim, x_dim, u_dim):
        super().__init__()
        self.z_dim = z_dim
        self.trans = TransitionNet(z_dim, u_dim)
        self.emit = EmissionNet(z_dim, x_dim, y_dim=1)

    def model(self, y, x, u, mask):
        """
        y: (B,T,1), x(exog for emission): (B,T,x_dim), u(exog for transition): (B,T,u_dim), mask: (B,T)
        """
        B, T, _ = y.shape
        pyro.module("dmm", self)
        z_loc_0 = y.new_zeros(B, self.z_dim)
        z_scale_0 = y.new_ones(B, self.z_dim)
        z_prev = pyro.sample("z_0", dist.Normal(z_loc_0, z_scale_0).to_event(1))
        h_prev = None
        for t in pyro.markov(range(T)):
            loc_t, scale_t, h_prev = self.trans(z_prev, u[:, t, :], h_prev)
            z_t = pyro.sample(f"z_{t+1}", dist.Normal(loc_t, scale_t).to_event(1))
            y_loc, y_scale = self.emit(z_t, x[:, t, :])
            obs_t = y[:, t, 0]
            pyro.sample(
                f"y_{t+1}",
                dist.Normal(y_loc.squeeze(-1), y_scale.squeeze(-1)),
                obs=obs_t.where(
                    mask[:, t], torch.tensor(float("nan"), device=y.device)
                ),
            )
            z_prev = z_t

    def guide(self, y, x, u, mask):
        # Amortized mean-field posterior over z_t with a small encoder
        B, T, _ = y.shape
        enc = nn.GRU(
            input_size=1 + x.size(-1) + u.size(-1), hidden_size=64, batch_first=True
        ).to(y.device)
        pyro.module("encoder", enc)
        packed = torch.cat([y, x, u], dim=-1)
        h, _ = enc(packed)
        to_loc = nn.Linear(64, self.z_dim).to(y.device)
        to_scale = nn.Sequential(nn.Linear(64, self.z_dim), nn.Softplus()).to(y.device)
        pyro.module("enc_heads", nn.ModuleList([to_loc, to_scale]))
        for t in pyro.markov(range(T)):
            loc = to_loc(h[:, t, :])
            scale = to_scale(h[:, t, :]) + 1e-3
            pyro.sample(f"z_{t+1}", dist.Normal(loc, scale).to_event(1))


def train_dmm(batches, z_dim=8, steps=3000, lr=2e-3, multi_horizons=(1, 3, 6, 24)):
    """
    batches: iterable of (y, x, u, mask) with padding to same T per batch.
    Multi-step objective: additionally predict t+H targets from z_t rollouts.
    """
    dmm = DMM(z_dim=z_dim, x_dim=batches[0][1].size(-1), u_dim=batches[0][2].size(-1))
    svi = SVI(
        dmm.model, dmm.guide, pyro.optim.ClippedAdam({"lr": lr}), loss=Trace_ELBO()
    )
    pyro.clear_param_store()
    for s in range(steps):
        loss_epoch = 0.0
        for y, x, u, mask in batches:
            loss = svi.step(y, x, u, mask)
            loss_epoch += loss
        if (s + 1) % 500 == 0:
            print(f"step {s+1}  ELBO: {-loss_epoch:.2f}")
    # For multi-step forecasts: sample z_t from posterior, rollout transitions H steps, decode at each step.
    return dmm
