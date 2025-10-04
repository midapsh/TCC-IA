# dkl_sm.py
import torch, torch.nn as nn, gpytorch


class Featurizer(nn.Module):
    def __init__(self, in_dim, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):  # x: (..., in_dim)
        return self.net(x)


class DKL_SMGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, n_mixtures=6):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=n_mixtures, ard_num_dims=feature_extractor.net[-1].out_features
        )

    def forward(self, x_raw):
        z = self.feature_extractor(x_raw)
        mean = self.mean_module(z)
        covar = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
