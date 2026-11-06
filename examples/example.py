# Copyright (c) 2026 Daniel Dehtyriov
# SPDX-License-Identifier: AGPL-3.0-only
"""Run the Reynolds-number DARSM checkpoint on a simple square-duct velocity
field and print the predicted closure constants at one cell.
Run from the repository root: python examples/example.py"""
import sys
sys.path.insert(0, "darsm")

import torch
from pretrained import load_checkpoint, build_input

net, ck = load_checkpoint("checkpoints/multi_re.pt")

# Square duct of side 2: streamwise velocity W, no secondary flow
n, L = 64, 2.0
h = L / n
x = (torch.arange(n, dtype=torch.float64) + 0.5) * h
X, Y = torch.meshgrid(x, x, indexing="ij")
W = 1.5 * (1 - (X - 1)**2) * (1 - (Y - 1)**2)
U = V = torch.zeros_like(W)
k = 0.01 * W + 1e-6
omega = 50.0 * torch.ones_like(W)
nu = L / 5000.0

features = build_input(U, V, W, k, omega, nu, h, h, ck["cnorm_reference"]["Re5000"])
with torch.no_grad():
    constants = net(features)[n // 4, n // 4]

names = ["beta*", "beta_0", "gamma", "sigma_k", "sigma_omega", "c1", "c2"]
defaults = [0.09, 0.075, 5 / 9, 0.5, 0.5, 1.8, 5 / 9]
for name, default, value in zip(names, defaults, constants.tolist()):
    print(name, round(default, 4), round(value, 4))
