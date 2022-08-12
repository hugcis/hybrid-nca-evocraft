#!/usr/bin/env python3
import pickle
import sys

import torch

from evo_ca.models import NCAModel
from evo_ca.utils import get_state_nca, spawn_state_nca, Blocks, CLIENT

STEPS = 2


def main() -> None:
    ca = NCAModel()
    fname = sys.argv[1]

    with open(fname, "rb") as f:
        weights = pickle.load(f)

    ca.load_weights(weights)

    arr, s_blocks = get_state_nca((70, 1, 70))
    CLIENT.spawnBlocks(Blocks(blocks=s_blocks.blocks))

    x = torch.tensor(arr, dtype=torch.float32)
    while True:
        for _ in range(STEPS):
            x = ca(x)
        spawn_state_nca(x[0, :4, :, :])


if __name__ == "__main__":
    main()
