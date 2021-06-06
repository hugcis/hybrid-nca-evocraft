#!/usr/bin/env python3
import pickle

import numpy as np
import torch
import torch.nn as nn

from evo_ca.models import NCAModel3D
from evo_ca.utils import spawn_state_nca_3d

STEPS = 4

ca3d = NCAModel3D()

weights = pickle.load(open("training_data_3d/3dtest4/24000.checkpoint.pkl", "rb"))

list(ca3d.dmodel.children())[0].weight = nn.Parameter(
    torch.moveaxis(torch.Tensor(weights[0]), (3, 4), (1, 0)))

list(ca3d.dmodel.children())[1].weight = nn.Parameter(
    torch.moveaxis(torch.Tensor(weights[1]), (3, 4), (1, 0)))
list(ca3d.dmodel.children())[1].bias = nn.Parameter(torch.Tensor(weights[2]))

list(ca3d.dmodel.children())[3].weight = nn.Parameter(
    torch.moveaxis(torch.Tensor(weights[3]), (3, 4), (1, 0)))
list(ca3d.dmodel.children())[3].bias = nn.Parameter(torch.Tensor(weights[4]))

s = 32
h, d = s, s
seed = np.zeros([2, s, s, s, 16], dtype=np.float)
w = 10
seed[0, h//2, w//2 - 4:w//2 - 1, d//2 + 3, 3:] = 1.0
seed[1, h//2, w//2 + 1, d//2 - 5, 3:] = 1.0
seed[1, h//2, w//2 + 1, d//2 - 7, 3:] = 1.0

seed[1, h//2, w//2, d//2 - 2:d//2 + 1, 3:] = 1.0
seed[0, h//2, w//2 + 3, d//2 - 6, 3:] = 1.0

x = torch.Tensor(seed[:])
x = torch.moveaxis(x, -1, 1)

while True:
    for _ in range(STEPS):
        x = ca3d(x)
    spawn_state_nca_3d(x[1, :4, :, :, :], z_offset=-150, x_offset=100, y_base=5)
