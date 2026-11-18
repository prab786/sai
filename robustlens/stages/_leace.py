"""Minimal LEACE, used only when the ``concept-erasure`` package is unavailable (it needs Python >= 3.10).

A direct port of ``LeaceFitter`` (method "leace", affine, trace constraint, optimal linear shrinkage) from
concept-erasure by EleutherAI (MIT licence), Belrose et al. (2023), "LEACE: Perfect linear concept erasure
in closed form". ``tests/test_units.py`` checks it against the package when the package is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class LeaceEraser:
    proj_left: Tensor
    proj_right: Tensor
    bias: Optional[Tensor]

    def __call__(self, x: Tensor) -> Tensor:
        delta = x - self.bias if self.bias is not None else x
        return (x - (delta @ self.proj_right.mH) @ self.proj_left.mH).type_as(x)

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, svd_tol: float = 0.01) -> "LeaceEraser":
        return fit_leace(x, z, svd_tol)


def _trace(m: Tensor) -> Tensor:
    return torch.linalg.diagonal(m).sum(dim=-1, keepdim=True).unsqueeze(-1)


def optimal_linear_shrinkage(S_n: Tensor, n) -> Tensor:
    p = S_n.shape[-1]
    eye = torch.eye(p, dtype=S_n.dtype, device=S_n.device).expand_as(S_n)
    trace_S = _trace(S_n)
    sigma0 = eye * trace_S / p
    sigma0_norm_sq = sigma0.norm(dim=(-2, -1), keepdim=True) ** 2
    S_norm_sq = S_n.norm(dim=(-2, -1), keepdim=True) ** 2
    prod_trace = _trace(S_n @ sigma0)
    top = trace_S * trace_S.conj() * sigma0_norm_sq / n
    bottom = S_norm_sq * sigma0_norm_sq - prod_trace * prod_trace.conj()
    eps = torch.finfo(S_n.dtype).eps
    alpha = 1 - (top + eps) / (bottom + eps)
    beta = (1 - alpha) * (prod_trace + eps) / (sigma0_norm_sq + eps)
    return alpha * S_n + beta * sigma0


@torch.no_grad()
def fit_leace(x: Tensor, z: Tensor, svd_tol: float = 0.01) -> LeaceEraser:
    n, d = x.shape
    z = z.reshape(n, -1).type_as(x)
    mean_x, mean_z = x.mean(0), z.mean(0)
    dx, dz = x - mean_x, z - mean_z
    S_hat = dx.mH @ dx
    S_hat = (S_hat + S_hat.mH) / 2
    sigma = optimal_linear_shrinkage(S_hat / n, n)
    sigma_xz = dx.mH @ dz / (n - 1)

    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    L, V = torch.linalg.eigh(sigma)
    mask = L > (L[-1] * d * torch.finfo(L.dtype).eps)
    L = L.clamp_min(0.0)
    W = V * torch.where(mask, L.rsqrt(), torch.zeros_like(L)) @ V.mH
    W_inv = V * torch.where(mask, L.sqrt(), torch.zeros_like(L)) @ V.mH

    u, s, _ = torch.linalg.svd(W @ sigma_xz, full_matrices=False)
    u = u * (s > svd_tol)
    proj_left, proj_right = W_inv @ u, u.mH @ W

    # keep the trace of the covariance from growing after erasure
    P = eye - proj_left @ proj_right
    old_trace, new_trace = torch.trace(sigma), torch.trace(P @ sigma @ P.mH)
    if new_trace.real > old_trace.real:
        Q = eye - u @ u.mH
        a, b, c, w = new_trace, 2 * torch.trace(P @ sigma @ Q.mH), torch.trace(Q @ sigma @ Q.mH), old_trace
        discr = torch.sqrt(4 * w * a - 4 * w * b + 4 * w * c - 4 * a * c + b ** 2)
        alpha1 = (-b / 2 + c - discr / 2) / (a - b + c)
        alpha2 = (-b / 2 + c + discr / 2) / (a - b + c)
        alpha = torch.where(alpha1.real > 0, alpha1, alpha2).clamp(0, 1)
        P = alpha * P + (1 - alpha) * Q
        uu, ss, vh = torch.linalg.svd(eye - P)
        proj_left, proj_right = uu * ss.sqrt(), vh * ss.sqrt()
    return LeaceEraser(proj_left, proj_right, bias=mean_x)
