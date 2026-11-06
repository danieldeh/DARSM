# Copyright (c) 2026 Daniel Dehtyriov
# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate the pretrained DARSM checkpoints.

The network is a small gated MLP from 6 input features to the 7 closure
constants used by k-omega EARSM. The 6 features are:

    II_S    = tr(S^2)
    II_O    = tr(Omega^2)
    III_S   = tr(S^3)
    IV      = tr(S Omega^2)
    V       = tr(S^2 Omega^2)
    log_Re_T = log(1 + k / (omega nu))

each divided by a per-case normalisation constant in CNorm. S and Omega are
evaluated at cell centres using compute_S_O.
"""
import torch
import torch.nn as nn


class DARSM_NN(nn.Module):

    def __init__(self, num_inputs=6, num_outputs=7, hidden=10, C=1.5):
        super().__init__()
        H = hidden
        self.fc1  = nn.Linear(num_inputs, H, dtype=torch.float64)
        self.fcG  = nn.Linear(num_inputs, H, dtype=torch.float64)
        self.fcG2 = nn.Linear(num_inputs, H, dtype=torch.float64)
        self.fc2  = nn.Linear(H, H, dtype=torch.float64)
        self.fc3  = nn.Linear(H, H, dtype=torch.float64)
        self.fc4  = nn.Linear(H, num_outputs, dtype=torch.float64)

        # not used in forward(), only here so saved state_dicts load cleanly
        self.fcScalar = nn.Linear(H, 1, dtype=torch.float64)

        self.sigmoid = nn.Sigmoid()
        self.C = C

        # standard Wilcox k-omega + Wallin-Johansson values
        self.beta_star_ref   = 9 / 100
        self.gamma_ref       = 5 / 9
        self.beta_0_ref      = 3 / 40
        self.sigma_k_ref     = 1 / 2
        self.sigma_omega_ref = 1 / 2
        self.c1_ref          = 1.8
        self.c2_ref          = 5 / 9

    def forward(self, x):
        """x: (..., 6) normalised invariants.
        Returns (..., 7): [beta*, beta_0, gamma, sigma_k, sigma_omega, c1, c2].
        """
        H1 = torch.tanh(self.fc1(x))
        H2 = torch.tanh(self.fc2(H1))
        G  = torch.tanh(self.fcG(x))
        G2 = torch.tanh(self.fcG2(x))
        H3 = torch.tanh(self.fc3(G * H2))
        raw = self.sigmoid(self.fc4(H3 * G2))

        beta_star   = self.beta_star_ref   * (1 + self.C * (raw[..., 0] - 0.5))
        gamma       = self.gamma_ref       * (1 + self.C * (raw[..., 1] - 0.5))
        beta_0      = self.beta_0_ref      * (1 + self.C * (raw[..., 2] - 0.5))
        sigma_k     = self.sigma_k_ref     * (1 + self.C * (raw[..., 3] - 0.5))
        sigma_omega = self.sigma_omega_ref * (1 + self.C * (raw[..., 4] - 0.5))
        c1          = self.c1_ref          * (1 + self.C * (raw[..., 5] - 0.5))
        c2          = self.c2_ref          * (1 + self.C * (raw[..., 6] - 0.5))

        return torch.stack(
            [beta_star, beta_0, gamma, sigma_k, sigma_omega, c1, c2],
            dim=-1,
        )


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=True)
    a = ck["arch"]
    net = DARSM_NN(num_inputs=a["num_inputs"], num_outputs=a["num_outputs"],
                   hidden=a["hidden"], C=a["C"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def compute_invariants(S, O):
    """Pope-basis scalar invariants of S and Omega."""
    S2, O2 = S @ S, O @ O
    def tr(A): return A.diagonal(dim1=-2, dim2=-1).sum(-1)
    return tr(S2), tr(O2), tr(S2 @ S), tr(S @ O2), tr(S2 @ O2)


def _ddx(F, x):
    if torch.is_tensor(x):
        dx_int = (x[2:] - x[:-2]).reshape(-1, 1)
        dx_lo, dx_hi = (x[1] - x[0]), (x[-1] - x[-2])
    else:
        dx_int, dx_lo, dx_hi = 2 * x, x, x
    out = torch.empty_like(F)
    out[1:-1, :] = (F[2:, :] - F[:-2, :]) / dx_int
    out[0,    :] = (F[1, :]  - F[0, :])  / dx_lo
    out[-1,   :] = (F[-1, :] - F[-2, :]) / dx_hi
    return out


def _ddy(F, y):
    if torch.is_tensor(y):
        dy_int = (y[2:] - y[:-2]).reshape(1, -1)
        dy_lo, dy_hi = (y[1] - y[0]), (y[-1] - y[-2])
    else:
        dy_int, dy_lo, dy_hi = 2 * y, y, y
    out = torch.empty_like(F)
    out[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / dy_int
    out[:, 0]    = (F[:, 1]  - F[:, 0])  / dy_lo
    out[:, -1]   = (F[:, -1] - F[:, -2]) / dy_hi
    return out


def compute_S_O(U, V, W, x, y, tau):
    """S and Omega at cell centres, both shape (nx, ny, 3, 3)."""
    du_dx, du_dy = _ddx(U, x), _ddy(U, y)
    dv_dx, dv_dy = _ddx(V, x), _ddy(V, y)
    dw_dx, dw_dy = _ddx(W, x), _ddy(W, y)

    z = torch.zeros_like(U)
    S11 = tau * du_dx
    S22 = tau * dv_dy
    S12 = 0.5 * tau * (du_dy + dv_dx)
    S13 = 0.5 * tau * dw_dx
    S23 = 0.5 * tau * dw_dy
    O12 =  0.5 * tau * (du_dy - dv_dx)
    O13 = -0.5 * tau * dw_dx
    O23 = -0.5 * tau * dw_dy

    S = torch.stack([
        torch.stack([S11, S12, S13], dim=-1),
        torch.stack([S12, S22, S23], dim=-1),
        torch.stack([S13, S23, z],   dim=-1),
    ], dim=-2)

    O = torch.stack([
        torch.stack([z,    O12, O13], dim=-1),
        torch.stack([-O12, z,   O23], dim=-1),
        torch.stack([-O13, -O23, z],  dim=-1),
    ], dim=-2)

    return S, O


def build_input(U, V, W, k, omega, nu, x, y, CNorm):
    """Network input features, shape (nx, ny, 6), divided by CNorm."""
    eps = 1e-12
    S, O = compute_S_O(U, V, W, x, y, 1.0 / (omega + eps))
    II_S, II_O, III_S, IV, V_ = compute_invariants(S, O)
    log_Re_T = torch.log1p(k / (omega * nu + eps))
    feats = torch.stack([II_S, II_O, III_S, IV, V_, log_Re_T], dim=-1)
    return feats / CNorm


def compute_CNorm(U, V, W, k_baseline, omega_baseline, nu, x, y):
    """Recompute CNorm from DNS velocities and a baseline k-omega state."""
    eps = 1e-12
    S, O = compute_S_O(U, V, W, x, y, 1.0 / (omega_baseline + eps))
    II_S, II_O, III_S, IV, V_ = compute_invariants(S, O)
    log_Re_T = torch.log1p(k_baseline / (omega_baseline * nu + eps))
    pieces = (II_S, II_O, III_S, IV, V_, log_Re_T)
    return torch.stack([t.abs().max() for t in pieces]).clamp(min=1e-10)
