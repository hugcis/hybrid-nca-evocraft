import grpc
import numpy as np
import torch
import itertools

import minecraft_pb2_grpc
from minecraft_pb2 import *


CHANNEL = grpc.insecure_channel('localhost:5001')
CLIENT = minecraft_pb2_grpc.MinecraftServiceStub(CHANNEL)

y_offset = 10
cols = np.array([
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
    [89.9944,  98.8727, 104.9356],
    [257.6869, 160.5991,  54.7574],
    [137, 137, 137],
]) / 255
b_map = [BLUE_GLAZED_TERRACOTTA, GRAY_GLAZED_TERRACOTTA,
         BROWN_GLAZED_TERRACOTTA, GREEN_GLAZED_TERRACOTTA,
         RED_GLAZED_TERRACOTTA, CYAN_GLAZED_TERRACOTTA, CONCRETE, OBSIDIAN,
         YELLOW_GLAZED_TERRACOTTA, LIME_GLAZED_TERRACOTTA, STONE,
         ORANGE_GLAZED_TERRACOTTA, STONE]
block_map = dict(zip(range(len(b_map)), b_map))
inv_block_map = dict(zip(b_map, range(len(b_map))))


def clear(size, bis, arena=None):
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=0, y=5, z=0),
            max=Point(x=size, y=y_offset + size, z=size)
        ),
        type=AIR
    ))
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-0, y=0, z=0),
            max=Point(x= size, y=3, z= size)
        ),
        type=DIRT
    ))
    if arena is not None:
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=-1, y = y_offset, z = -1),
                max=Point(x = bis, y = y_offset, z=-1)
            ),
            type=COBBLESTONE
        ))
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=-1, y = y_offset, z = -1),
                max=Point(x = -1, y = y_offset, z=bis)
            ),
            type=COBBLESTONE
        ))

        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=bis, y = y_offset, z = -1),
                max=Point(x =bis, y = y_offset, z=bis)
            ),
            type=COBBLESTONE
        ))
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=-1, y = y_offset, z =bis),
                max=Point(x = bis, y = y_offset, z=bis)
            ),
            type=COBBLESTONE
        ))
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=-1, y = y_offset-1, z =-1),
                max=Point(x = bis, y = y_offset-1, z=bis)
            ),
            type=COBBLESTONE
        ))


def get_state_nca(shape, y_base=y_offset):
    blocks = CLIENT.readCube(Cube(
        min=Point(x=0, y=y_base, z=0),
        max=Point(x=shape[0]-1, y=y_base + shape[1]-1, z=shape[2]-1)
    ))
    arr = np.zeros((1, 16, 70, 70))
    for b in blocks.blocks:
        if b.type == DIRT:
            arr[0, 3:, b.position.x, b.position.z] = 1
    return arr, blocks


def get_color(st):
    if st[-1] > 0.4:
        return block_map[np.linalg.norm(st[:-1] - cols, axis=1).argmin()]
    else:
        return AIR


def spawn_state_nca(state, y_base=y_offset):
    blocks = []
    if isinstance(state, torch.Tensor):
        state = state.detach().numpy()
    for (i, j) in itertools.product(
            *[range(state[0].shape[u]) for u in range(2)]):
        item = get_color(state[:, i, j])
        blocks.append(Block(position=Point(x=i, y=y_base, z=j),
                            type=item, orientation=NORTH))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))


def spawn_state_nca_3d(state, y_base=y_offset, x_offset=0, z_offset=0):
    blocks = []
    if isinstance(state, torch.Tensor):
        state = state.detach().numpy()
    for (i, j, k) in itertools.product(
            *[range(state[0].shape[u]) for u in range(3)]):
        item = get_color(state[:, i, j, k])
        blocks.append(Block(position=Point(x=i + x_offset, y=y_base + j,
                                           z=k + z_offset),
                            type=item, orientation=NORTH))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
