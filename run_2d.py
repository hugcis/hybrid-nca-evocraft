#!/usr/bin/env python3
import pickle

import numpy as np
import torch
import torch.nn as nn

from evo_ca.models import NCAModel
from evo_ca.utils import spawn_state_nca, get_state_nca, CLIENT, Blocks

STEPS=2

ca = NCAModel()

weights = pickle.load(open("training_data_2d/gorilla_giraffe_trex_cat_2seeds/\
60000.checkpoint.pkl", "rb"))

list(ca.dmodel.children())[1].weight = nn.Parameter(
    torch.moveaxis(torch.Tensor(weights[0]), (2, 3), (1, 0)))
list(ca.dmodel.children())[1].bias = nn.Parameter(torch.Tensor(weights[1]))

list(ca.dmodel.children())[3].weight = nn.Parameter(
    torch.moveaxis(torch.Tensor(weights[2]), (2, 3), (1, 0)))
list(ca.dmodel.children())[3].bias = nn.Parameter(torch.Tensor(weights[3]))


arr, s_blocks = get_state_nca((70, 1, 70))
CLIENT.spawnBlocks(Blocks(blocks=s_blocks.blocks))

x = torch.Tensor(arr)
while True:
    for t in range(STEPS):
        x = ca(x)
    spawn_state_nca(x[0, :4, :, :])
