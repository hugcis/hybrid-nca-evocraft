import numpy as np
import torch
import torch.nn as nn

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5


class NCAModel(nn.Module):
    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.pool = nn.MaxPool2d((3, 3), stride=(1,1), padding=(1,1))


        perc = nn.Conv2d(self.channel_n, 48, (3, 3), groups=16, bias=False,
                         padding_mode="circular", padding=(1, 1))
        identify = np.array([0, 1, 0], dtype=np.float)
        identify = torch.Tensor(np.outer(identify, identify))

        dx = torch.Tensor(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0)  # Sobel filter
        dy = dx.T

        kernel = torch.stack([identify, dx, dy], -1)[:, :, None, :]
        kernel = kernel.repeat(1, 1, CHANNEL_N, 1)
        kernel = torch.moveaxis(
            kernel.reshape(3, 3, -1), -1, 0).reshape(-1, 1, 3, 3)
        perc.weight = nn.Parameter(kernel)

        self.dmodel = nn.Sequential(
            perc,
            nn.Conv2d(48, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, self.channel_n, (1, 1)),
        )

    def forward(self, x, fire_rate=None, step_size=1.0):
        pre_life_mask = self.get_living_mask(x)
        dx = self.dmodel(x) * step_size
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = torch.rand(size=x[:, :1, :, :].shape) <= fire_rate
        x += dx * update_mask.float()

        post_life_mask = self.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * life_mask.float()

    def get_living_mask(self, x):
        alpha = x[:, 3:4, :, :]
        return self.pool(alpha) > 0.1

class NCAModel3D(nn.Module):
    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.pool3d = nn.MaxPool3d((3, 3, 3), stride=(1, 1, 1),
                                   padding=(1, 1, 1))

        perc = nn.Conv3d(self.channel_n, 64, (3, 3, 3), groups=16, bias=False,
                         padding_mode="circular", padding=(1, 1, 1))
        identify = torch.zeros(3, 3, 3)
        identify[1, 1, 1] = 1

        # 3D Sobel filters
        dx = torch.Tensor(np.array([1, 2, 1])[None, None, :] *
                          np.outer([1, 2, 1], [-1, 0, 1])[:, :, None])
        dy = torch.Tensor(np.array([1, 2, 1])[None, None, :] *
                          np.outer([1, 2, 1], [-1, 0, 1]).T[:, :, None])
        dz = torch.Tensor((np.array([1, 2, 1])[None, None, :] *
                           np.outer([1, 2, 1], [-1, 0, 1]).T[:, :, None]).T)

        kernel = torch.stack([identify, dx, dy, dz], -1)[:, :, :, :, None]
        kernel = kernel.repeat(1, 1, 1, CHANNEL_N, 1)
        kernel = torch.moveaxis(
            kernel.reshape(3, 3, 3, -1), -1, 0).reshape(-1, 1, 3, 3, 3)
        perc.weight = nn.Parameter(kernel)

        self.dmodel = nn.Sequential(
            perc,
            nn.Conv3d(64, 128, (1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, self.channel_n, (1, 1, 1)),
        )

    def forward(self, x, fire_rate=None, step_size=1.0):
        pre_life_mask = self.get_living_mask3d(x)
        dx = self.dmodel(x) * step_size
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = torch.rand(size=x[:, :1, :, :, :].shape) <= fire_rate
        x += dx * update_mask.float()

        post_life_mask = self.get_living_mask3d(x)
        life_mask = pre_life_mask & post_life_mask
        return x * life_mask.float()

    def get_living_mask3d(self, x):
        alpha = x[:, 3:4, :, :, :]
        return self.pool3d(alpha) > 0.1
