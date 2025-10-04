# sbi_calibration.py
import torch
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference


def run_sbi(simulator_fn, prior_low, prior_high, num_sims=20000, device="cpu"):
    """
    simulator_fn: function(theta, x) -> y_sim  (vectorized over theta)
    prior: uniform box over theta
    """
    prior = sbi_utils.BoxUniform(
        low=torch.tensor(prior_low, device=device),
        high=torch.tensor(prior_high, device=device),
    )
    inference = sbi_inference.SNPE(prior=prior)
    # Draw parameter proposals and simulate
    theta = prior.sample((num_sims,))
    # x_cond could be station covariates; here we pass None for simplicity
    y = simulator_fn(theta)  # shape (num_sims, summary_dim)
    density_estimator = inference.append_simulations(theta, y).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior  # callable: posterior.sample((N,), x=...)
