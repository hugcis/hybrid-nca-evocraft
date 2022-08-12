#!/usr/bin/env python3
import pickle
import sys

import numpy as np
import numpy.typing as npt
import torch

from evo_ca.models import NCAModel3D
from evo_ca.utils import spawn_state_nca_3d

STEPS = 4


def create_seed(size: int = 32, channels: int = 16) -> npt.NDArray[np.float64]:
    """Create initial seed state for the 3D NCA."""
    seed = np.zeros([2, size, size, size, channels], dtype=np.float64)
    h, d = size, size
    w = 10

    seed[0, h // 2, w // 2 - 4 : w // 2 - 1, d // 2 + 3, 3:] = 1.0
    seed[1, h // 2, w // 2 + 1, d // 2 - 5, 3:] = 1.0
    seed[1, h // 2, w // 2 + 1, d // 2 - 7, 3:] = 1.0
    seed[1, h // 2, w // 2, d // 2 - 2 : d // 2 + 1, 3:] = 1.0
    seed[0, h // 2, w // 2 + 3, d // 2 - 6, 3:] = 1.0

    return seed


def main() -> None:
    ca3d = NCAModel3D()
    fname = sys.argv[1]

    with open(fname, "rb") as f:
        weights = pickle.load(f)

    ca3d.load_weights(weights)

    seed = create_seed()
    x = torch.tensor(seed, dtype=torch.float32)
    x = x.permute(0, 4, 1, 2, 3)  # Move channels to dim 1

    while True:
        for _ in range(STEPS):
            x = ca3d(x)
        spawn_state_nca_3d(x[1, :4, :, :, :], z_offset=-150, x_offset=100, y_base=5)


if __name__ == "__main__":
    main()
