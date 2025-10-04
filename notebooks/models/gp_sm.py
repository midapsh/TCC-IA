# gp_sm.py
import torch, gpytorch


class SMGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_mixtures=6):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=n_mixtures
        )
        # Helpful init: place mixture means near diurnal/annual frequencies
        # For hourly data: 1/day ≈ 1/24, 1/year ≈ 1/(24*365.25)
        if train_x.dim() == 1:
            freq_init = torch.tensor(
                [1 / 24.0, 1 / (24 * 365.25)],
                dtype=train_x.dtype,
                device=train_x.device,
            )
            means = 2 * torch.pi * freq_init[None, :]  # to rad/h if x is hours
            self.covar_module.initialize_from_data_empspect(train_x, train_y)
            # (gpytorch will refine; the helper aligns initial frequencies)
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_smgp(x, y, n_mixtures=6, iters=1000, lr=0.1):
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    mask = ~torch.isnan(y)
    x, y = x[mask], y[mask]
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SMGP(x, y, likelihood, n_mixtures=n_mixtures)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(iters):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 200 == 0:
            print(f"{i+1}: loss {loss.item():.3f}")
    model.eval()
    likelihood.eval()
    return model, likelihood


@torch.no_grad()
def predict_smgp(model, likelihood, x_star, return_ci=True):
    x_star = torch.tensor(x_star).float()
    with gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x_star))
    mean = pred.mean
    if return_ci:
        lower, upper = pred.confidence_region()
        return mean, lower, upper
    return mean
