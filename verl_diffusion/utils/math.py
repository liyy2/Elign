import math
import numpy as np
import torch


def kl_divergence_normal(mu_P, sigma_P, mu_Q):
    """
    Compute the Kullback-Leibler (KL) divergence between two normal distributions using PyTorch.
    
    Args:
    - mu_P (float or tensor): Mean of the first normal distribution (P)
    - sigma_P (float or tensor): Standard deviation of the first normal distribution (P)
    - mu_Q (float or tensor): Mean of the second normal distribution (Q)
    - sigma_Q (float or tensor): Standard deviation of the second normal distribution (Q)
    
    Returns:
    - float or tensor: KL divergence from P to Q
    """
    # First term: log(sigma_Q / sigma_P)
    
    # Second term: (sigma_P^2 + (mu_P - mu_Q)^2) / (2 * sigma_Q^2)
    term2 = (sigma_P**2 + (mu_P - mu_Q)**2) / (2 * sigma_P**2+1e-8)
    
    # Third term: -1/2
    term3 = -0.5
    
    # Calculate the KL divergence
    kl_div = term2 + term3
    return kl_div

def rmsd(A, B=None):
    """
    Calculate the RMSD (Root Mean Square Deviation) between two 2D matrices A and B.
    If B is not provided, it defaults to a zero matrix.

    Parameters:
    A: numpy.ndarray, shape (m, n)
    B: numpy.ndarray, shape (m, n), default is None. If None, B will be set to a zero matrix.
    
    Returns:
    float: RMSD value
    """
    # If B is None, set B to a zero matrix with the same shape as A
    if B is None:
        B = np.zeros_like(A)

    # Ensure matrices A and B have the same shape
    if A.shape != B.shape:
        raise ValueError("The input matrices A and B must have the same shape")
    
    # Calculate the squared differences between matrices A and B
    diff = A - B
    squared_diff = np.square(diff)
    
    # Compute the root mean square deviation (RMSD)
    rmsd_value = np.sqrt(np.mean(squared_diff))

    return rmsd_value


def policy_step_logprob(z_s, mu, sigma, mask):
    """Return the log-probability of diffusion policy actions per sample.

    This helper is intentionally explicit about the Normal distribution terms so
    reviewers can map each tensor operation to the analytical expression:

    ``log N(z_s; mu, sigma) = -0.5 * ((z_s - mu) / sigma)^2``
    ``                           - log(sigma) - 0.5 * log(2π)``.

    Args:
        z_s (torch.Tensor): Sampled latents for the current diffusion step.
        mu (torch.Tensor): Mean predicted by the policy for the step.
        sigma (torch.Tensor): Standard deviation predicted for the step.
        mask (torch.Tensor): Node mask used to ignore padded atoms.

    Returns:
        torch.Tensor: Scalar log-probability per sample with masked dimensions summed out.
    """

    if mask is None:
        mask = torch.ones_like(z_s[..., :1], dtype=z_s.dtype, device=z_s.device)
    else:
        mask = mask.to(device=z_s.device, dtype=z_s.dtype)

    mu = mu.to(dtype=z_s.dtype, device=z_s.device)
    sigma = torch.clamp(sigma.to(dtype=z_s.dtype, device=z_s.device), min=1e-12)

    diff = z_s - mu
    inv_sq = (diff / sigma) ** 2
    # Combine the per-dimension normalizer terms: log(sigma) + 0.5 log(2π).
    normalizer = torch.log(sigma) + sigma.new_tensor(0.5 * math.log(2 * math.pi))
    lp = -0.5 * inv_sq - normalizer

    if mask.dim() < lp.dim():
        # Allow callers to pass [B, N, 1] masks while we work with [B, N, C]
        # tensors by appending singleton dimensions lazily.
        mask = mask.view(*mask.shape, *([1] * (lp.dim() - mask.dim())))

    reduce_dims = tuple(range(1, lp.dim()))
    # Collapse every non-batch dimension so the caller receives a scalar per
    # sample, matching PPO-style objective expectations.
    return (lp * mask).sum(dim=reduce_dims)
