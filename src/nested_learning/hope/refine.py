"""FSRM-inspired inner refinement block.

Implements a weight-shared refinement operator ``R`` that is applied ``T`` times
between the SelfMod output and the CMS write path.  Each iteration refines the
latent representation before it is committed to memory, following the FSRM paper's
key insight that *repeated* application of a *shared* module improves
out-of-distribution generalisation (arXiv:2604.01577, Table 1).

The design choices here are deliberately minimal:

* A small MLP with RMSNorm (not AKOrN / Kuramoto oscillators) — simpler and
  already proven competitive in FSRM's ablations (Appendix C.3).
* A learnable step-size ``alpha`` (analogous to FSRM's ``gamma``), initialised
  small so that T=1 degrades gracefully to the identity.
* Optional sphere normalisation after each step (ties into the ``output_l2_norm``
  flag on ``SelfModifyingTitansConfig``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineBlock(nn.Module):
    """Weight-shared inner refinement operator.

    Applied ``T`` times in a loop to iteratively refine the latent
    representation before a CMS memory write.

    Parameters
    ----------
    dim : int
        Model / latent dimension.
    hidden_multiplier : int
        Width multiplier for the internal MLP (hidden = dim * hidden_multiplier).
    init_alpha : float
        Initial value for the learnable step-size parameter.  Kept small
        (0.1) so the first few training steps behave close to identity.
    sphere_norm : bool
        If True, apply L2-normalisation (sphere projection) after each
        refinement step — matches FSRM's Π(x) = x / ||x||.
    eps : float
        Epsilon for the normalisation operations.
    """

    def __init__(
        self,
        dim: int,
        hidden_multiplier: int = 2,
        init_alpha: float = 0.1,
        sphere_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.sphere_norm = sphere_norm
        self.eps = eps

        self.norm = nn.RMSNorm(dim, eps=eps)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_multiplier, bias=False),
            nn.GELU(),
            nn.Linear(dim * hidden_multiplier, dim, bias=False),
        )
        # Learnable step size (analogous to FSRM's γ).
        # Initialised small so T=1 ≈ identity at start of training.
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

        # Zero-init the last projection so freshly-initialised refine
        # steps produce near-zero updates (warm-start friendly).
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor, T: int = 1) -> torch.Tensor:
        """Apply T weight-shared refinement steps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, L, D)`` or ``(B, D)``.
        T : int
            Number of inner-loop iterations.

        Returns
        -------
        torch.Tensor
            Refined tensor, same shape as *x*.
        """
        for _ in range(T):
            # Cast to input dtype to avoid fp16/fp32 mismatch in RMSNorm
            normed = self.norm(x.to(self.norm.weight.dtype)).to(x.dtype)
            x = x + self.alpha * self.net(normed)
            if self.sphere_norm:
                x = F.normalize(x, dim=-1, eps=self.eps)
        return x
