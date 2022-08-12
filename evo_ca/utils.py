import itertools
from typing import Any

import grpc
import numpy as np
import numpy.typing as npt
import torch

import minecraft_pb2_grpc
from minecraft_pb2 import (
    AIR,
    BLUE_GLAZED_TERRACOTTA,
    BROWN_GLAZED_TERRACOTTA,
    COBBLESTONE,
    CONCRETE,
    CYAN_GLAZED_TERRACOTTA,
    DIRT,
    GRAY_GLAZED_TERRACOTTA,
    GREEN_GLAZED_TERRACOTTA,
    LIME_GLAZED_TERRACOTTA,
    NORTH,
    OBSIDIAN,
    ORANGE_GLAZED_TERRACOTTA,
    RED_GLAZED_TERRACOTTA,
    STONE,
    YELLOW_GLAZED_TERRACOTTA,
    Block,
    Blocks,
    Cube,
    FillCubeRequest,
    Point,
)

CHANNEL = grpc.insecure_channel("localhost:5001")
CLIENT = minecraft_pb2_grpc.MinecraftServiceStub(CHANNEL)

Y_OFFSET = 10
COLORS = np.array([
    [0, 0, 128],
    [144, 135, 147],
    [87, 72, 58],
    [0, 128, 0],
    [128, 0, 0],
    [2, 137, 209],
    [223.6294, 237.0306, 254.0567],
    [10, 10, 10],
    [255, 193, 0],
    [191, 255, 0],
    [89.9944, 98.8727, 104.9356],
    [257.6869, 160.5991, 54.7574],
    [137, 137, 137],
]) / 255

BLOCK_TYPES = [
    BLUE_GLAZED_TERRACOTTA,
    GRAY_GLAZED_TERRACOTTA,
    BROWN_GLAZED_TERRACOTTA,
    GREEN_GLAZED_TERRACOTTA,
    RED_GLAZED_TERRACOTTA,
    CYAN_GLAZED_TERRACOTTA,
    CONCRETE,
    OBSIDIAN,
    YELLOW_GLAZED_TERRACOTTA,
    LIME_GLAZED_TERRACOTTA,
    STONE,
    ORANGE_GLAZED_TERRACOTTA,
    STONE,
]
BLOCK_MAP: dict[int, int] = dict(zip(range(len(BLOCK_TYPES)), BLOCK_TYPES))
INV_BLOCK_MAP: dict[int, int] = dict(zip(BLOCK_TYPES, range(len(BLOCK_TYPES))))


def clear(size: int, bis: int, arena: Any | None = None) -> None:
    """Clear the Minecraft area and optionally create an arena border."""
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=0, y=5, z=0),
            max=Point(x=size, y=Y_OFFSET + size, z=size)
        ),
        type=AIR
    ))
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=0, y=0, z=0),
            max=Point(x=size, y=3, z=size)
        ),
        type=DIRT
    ))
    if arena is not None:
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=-1, y=Y_OFFSET, z=-1),
                max=Point(x=bis, y=Y_OFFSET, z=-1)
            ),
            type=COBBLESTONE
        ))
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=-1, y=Y_OFFSET, z=-1),
                max=Point(x=-1, y=Y_OFFSET, z=bis)
            ),
            type=COBBLESTONE
        ))
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=bis, y=Y_OFFSET, z=-1),
                max=Point(x=bis, y=Y_OFFSET, z=bis)
            ),
            type=COBBLESTONE
        ))
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=-1, y=Y_OFFSET, z=bis),
                max=Point(x=bis, y=Y_OFFSET, z=bis)
            ),
            type=COBBLESTONE
        ))
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=-1, y=Y_OFFSET - 1, z=-1),
                max=Point(x=bis, y=Y_OFFSET - 1, z=bis)
            ),
            type=COBBLESTONE
        ))


def get_state_nca(
    shape: tuple[int, int, int],
    y_base: int = Y_OFFSET,
) -> tuple[npt.NDArray[np.float64], Any]:
    """Read the current NCA state from Minecraft world."""
    blocks = CLIENT.readCube(Cube(
        min=Point(x=0, y=y_base, z=0),
        max=Point(x=shape[0] - 1, y=y_base + shape[1] - 1, z=shape[2] - 1)
    ))
    arr = np.zeros((1, 16, 70, 70))
    for b in blocks.blocks:
        if b.type == DIRT:
            arr[0, 3:, b.position.x, b.position.z] = 1
    return arr, blocks


def get_color(state: npt.NDArray[np.floating]) -> int:
    """Map NCA state to Minecraft block type based on color similarity."""
    if state[-1] > 0.4:
        return BLOCK_MAP[np.linalg.norm(state[:-1] - COLORS, axis=1).argmin()]
    return AIR


def spawn_state_nca(
    state: torch.Tensor | npt.NDArray[np.floating],
    y_base: int = Y_OFFSET,
) -> None:
    """Spawn 2D NCA state as blocks in Minecraft."""
    blocks = []
    if isinstance(state, torch.Tensor):
        state = state.detach().numpy()
    for i, j in itertools.product(
        *[range(state[0].shape[u]) for u in range(2)]
    ):
        item = get_color(state[:, i, j])
        blocks.append(Block(
            position=Point(x=i, y=y_base, z=j),
            type=item,
            orientation=NORTH,
        ))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))


def spawn_state_nca_3d(
    state: torch.Tensor | npt.NDArray[np.floating],
    y_base: int = Y_OFFSET,
    x_offset: int = 0,
    z_offset: int = 0,
) -> None:
    """Spawn 3D NCA state as blocks in Minecraft."""
    blocks = []
    if isinstance(state, torch.Tensor):
        state = state.detach().numpy()
    for i, j, k in itertools.product(
        *[range(state[0].shape[u]) for u in range(3)]
    ):
        item = get_color(state[:, i, j, k])
        blocks.append(Block(
            position=Point(x=i + x_offset, y=y_base + j, z=k + z_offset),
            type=item,
            orientation=NORTH,
        ))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
