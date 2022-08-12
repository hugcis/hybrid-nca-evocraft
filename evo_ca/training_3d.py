"""Training utilities for 3D Neural Cellular Automata.

Supports training NCA to grow Minecraft-style 3D voxel shapes.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from .models import CHANNEL_N, NCAModel3D


# -----------------------------------------------------------------------------
# 3D Target Shape Creation/Loading
# -----------------------------------------------------------------------------

def make_sphere(
    size: int = 16,
    radius: Optional[float] = None,
    color: tuple[float, float, float] = (0.8, 0.2, 0.2),
) -> npt.NDArray[np.float32]:
    """Create a 3D sphere target with RGBA channels."""
    if radius is None:
        radius = size / 3

    center = size / 2
    coords = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    dist = np.sqrt(
        (coords[0] - center) ** 2 +
        (coords[1] - center) ** 2 +
        (coords[2] - center) ** 2
    )

    alpha = (dist <= radius).astype(np.float32)
    rgba = np.zeros((size, size, size, 4), dtype=np.float32)
    rgba[..., 0] = color[0] * alpha
    rgba[..., 1] = color[1] * alpha
    rgba[..., 2] = color[2] * alpha
    rgba[..., 3] = alpha

    return rgba


def make_cube(
    size: int = 16,
    cube_size: Optional[int] = None,
    color: tuple[float, float, float] = (0.2, 0.6, 0.8),
) -> npt.NDArray[np.float32]:
    """Create a 3D cube target with RGBA channels."""
    if cube_size is None:
        cube_size = size // 2

    rgba = np.zeros((size, size, size, 4), dtype=np.float32)
    start = (size - cube_size) // 2
    end = start + cube_size

    rgba[start:end, start:end, start:end, 0] = color[0]
    rgba[start:end, start:end, start:end, 1] = color[1]
    rgba[start:end, start:end, start:end, 2] = color[2]
    rgba[start:end, start:end, start:end, 3] = 1.0

    return rgba


def make_pyramid(
    size: int = 16,
    color: tuple[float, float, float] = (0.9, 0.7, 0.2),
) -> npt.NDArray[np.float32]:
    """Create a 3D pyramid target with RGBA channels."""
    rgba = np.zeros((size, size, size, 4), dtype=np.float32)
    center = size // 2

    for y in range(size // 2):
        # Each layer is a smaller square
        layer_size = size // 2 - y
        start = center - layer_size // 2
        end = center + layer_size // 2 + layer_size % 2

        rgba[start:end, y, start:end, :3] = color
        rgba[start:end, y, start:end, 3] = 1.0

    return rgba


def make_torus(
    size: int = 24,
    major_radius: Optional[float] = None,
    minor_radius: Optional[float] = None,
    color: tuple[float, float, float] = (0.6, 0.2, 0.8),
) -> npt.NDArray[np.float32]:
    """Create a 3D torus target with RGBA channels."""
    if major_radius is None:
        major_radius = size / 4
    if minor_radius is None:
        minor_radius = size / 8

    center = size / 2
    rgba = np.zeros((size, size, size, 4), dtype=np.float32)

    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Distance from center axis in XZ plane
                dx = x - center
                dz = z - center
                dist_xz = np.sqrt(dx * dx + dz * dz)

                # Distance from torus tube center
                tube_dist = np.sqrt((dist_xz - major_radius) ** 2 + (y - center) ** 2)

                if tube_dist <= minor_radius:
                    rgba[x, y, z, :3] = color
                    rgba[x, y, z, 3] = 1.0

    return rgba


def make_cross(
    size: int = 16,
    arm_width: int = 3,
    color: tuple[float, float, float] = (0.2, 0.8, 0.2),
) -> npt.NDArray[np.float32]:
    """Create a 3D cross/plus target with RGBA channels."""
    rgba = np.zeros((size, size, size, 4), dtype=np.float32)
    center = size // 2
    half_width = arm_width // 2

    # X arm
    rgba[:, center - half_width:center + half_width + 1,
         center - half_width:center + half_width + 1, :3] = color
    rgba[:, center - half_width:center + half_width + 1,
         center - half_width:center + half_width + 1, 3] = 1.0

    # Y arm
    rgba[center - half_width:center + half_width + 1, :,
         center - half_width:center + half_width + 1, :3] = color
    rgba[center - half_width:center + half_width + 1, :,
         center - half_width:center + half_width + 1, 3] = 1.0

    # Z arm
    rgba[center - half_width:center + half_width + 1,
         center - half_width:center + half_width + 1, :, :3] = color
    rgba[center - half_width:center + half_width + 1,
         center - half_width:center + half_width + 1, :, 3] = 1.0

    return rgba


def load_voxel_npy(path: str) -> npt.NDArray[np.float32]:
    """Load a voxel shape from a .npy file.

    Expects shape (D, H, W, 4) with RGBA channels or (D, H, W) binary mask.
    """
    data = np.load(path)

    if data.ndim == 3:
        # Binary mask, convert to RGBA
        rgba = np.zeros((*data.shape, 4), dtype=np.float32)
        rgba[..., :3] = 0.5  # Gray color
        rgba[..., 3] = data.astype(np.float32)
    elif data.ndim == 4 and data.shape[-1] == 4:
        rgba = data.astype(np.float32)
    else:
        raise ValueError(f"Unexpected voxel shape: {data.shape}")

    return rgba


# Built-in shapes registry
BUILTIN_SHAPES = {
    "sphere": make_sphere,
    "cube": make_cube,
    "pyramid": make_pyramid,
    "torus": make_torus,
    "cross": make_cross,
}


def get_target_shape(
    name: str,
    size: int = 16,
    **kwargs,
) -> npt.NDArray[np.float32]:
    """Get a target shape by name or load from file."""
    if name in BUILTIN_SHAPES:
        return BUILTIN_SHAPES[name](size=size, **kwargs)
    elif name.endswith(".npy"):
        return load_voxel_npy(name)
    else:
        raise ValueError(f"Unknown shape: {name}. Available: {list(BUILTIN_SHAPES.keys())}")


# -----------------------------------------------------------------------------
# 3D Training Utilities
# -----------------------------------------------------------------------------

def to_rgba_3d(x: torch.Tensor) -> torch.Tensor:
    """Extract RGBA channels from 3D state (first 4 channels)."""
    return x[:, :4, :, :, :]


def to_alpha_3d(x: torch.Tensor) -> torch.Tensor:
    """Extract and clamp alpha channel from 3D state."""
    return torch.clamp(x[:, 3:4, :, :, :], 0.0, 1.0)


def to_rgb_3d(x: torch.Tensor) -> torch.Tensor:
    """Convert 3D state to RGB assuming premultiplied alpha."""
    rgb, a = x[:, :3, :, :, :], to_alpha_3d(x)
    return 1.0 - a + rgb


def make_seed_3d(
    size: int,
    n: int = 1,
    channel_n: int = CHANNEL_N,
) -> torch.Tensor:
    """Create initial 3D seed state with a single living cell in center."""
    x = torch.zeros(n, channel_n, size, size, size, dtype=torch.float32)
    c = size // 2
    x[:, 3:, c, c, c] = 1.0
    return x


def make_circle_masks_3d(
    n: int,
    d: int,
    h: int,
    w: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create random spherical masks for 3D damage."""
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, None, :]
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, None, :, None]
    z = torch.linspace(-1.0, 1.0, d, device=device)[None, :, None, None]

    # Random centers and radii
    center = torch.rand(3, n, 1, 1, 1, device=device) * 1.0 - 0.5
    r = torch.rand(n, 1, 1, 1, device=device) * 0.3 + 0.1

    x_c = (x - center[0]) / r
    y_c = (y - center[1]) / r
    z_c = (z - center[2]) / r

    mask = (x_c ** 2 + y_c ** 2 + z_c ** 2 < 1.0).float()
    return mask


class SamplePool3D:
    """Pool of 3D training samples for persistent/regenerating training."""

    def __init__(
        self,
        *,
        _parent: Optional["SamplePool3D"] = None,
        _parent_idx: Optional[npt.NDArray] = None,
        **slots,
    ):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = list(slots.keys())
        self._size: Optional[int] = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n: int) -> "SamplePool3D":
        """Sample n items from the pool."""
        idx = np.random.choice(self._size, n, replace=False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        return SamplePool3D(**batch, _parent=self, _parent_idx=idx)

    def commit(self) -> None:
        """Commit changes back to parent pool."""
        if self._parent is not None:
            for k in self._slot_names:
                getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


@dataclass
class Training3DConfig:
    """Configuration for 3D NCA training."""

    channel_n: int = CHANNEL_N
    target_padding: int = 4
    target_size: int = 16
    batch_size: int = 4
    pool_size: int = 256
    cell_fire_rate: float = 0.5

    # Training settings
    learning_rate: float = 2e-3
    lr_decay_steps: int = 2000
    lr_decay_factor: float = 0.1
    num_steps: int = 8000

    # Experiment type
    use_pattern_pool: bool = True
    damage_n: int = 1  # Number of patterns to damage per batch (less for 3D)

    # Logging
    log_every: int = 100
    save_every: int = 500
    log_dir: str = "train_log_3d"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def growing(cls) -> "Training3DConfig":
        """Config for growing experiment (no pool, no damage)."""
        return cls(use_pattern_pool=False, damage_n=0)

    @classmethod
    def persistent(cls) -> "Training3DConfig":
        """Config for persistent experiment (pool, no damage)."""
        return cls(use_pattern_pool=True, damage_n=0)

    @classmethod
    def regenerating(cls) -> "Training3DConfig":
        """Config for regenerating experiment (pool + damage)."""
        return cls(use_pattern_pool=True, damage_n=1)


class NCATrainer3D:
    """Trainer for 3D Neural Cellular Automata."""

    def __init__(
        self,
        model: NCAModel3D,
        target_voxels: npt.NDArray[np.float32],
        config: Training3DConfig,
    ):
        """Initialize 3D NCA trainer."""
        self.model = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)

        # Pad target
        p = config.target_padding
        self.pad_target = np.pad(
            target_voxels,
            [(p, p), (p, p), (p, p), (0, 0)]
        )
        self.d, self.h, self.w = self.pad_target.shape[:3]

        # Convert to torch tensor (NCDHW format)
        # Input is DHWC, need CDHW
        self.pad_target_tensor = torch.tensor(
            self.pad_target.transpose(3, 0, 1, 2)[None, :4, :, :, :],
            dtype=torch.float32,
            device=self.device,
        )

        # Create seed (DHWC format for storage, NCDHW for model)
        self.seed = np.zeros([self.d, self.h, self.w, config.channel_n], np.float32)
        self.seed[self.d // 2, self.h // 2, self.w // 2, 3:] = 1.0

        self.seed_tensor = torch.tensor(
            self.seed.transpose(3, 0, 1, 2)[None, :, :, :, :],
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize sample pool
        seed_batch = np.repeat(self.seed[None, ...], config.pool_size, axis=0)
        self.pool = SamplePool3D(x=seed_batch)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_steps,
            gamma=config.lr_decay_factor,
        )

        self.loss_log: list[float] = []

        os.makedirs(config.log_dir, exist_ok=True)

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE loss against target RGBA."""
        rgba = to_rgba_3d(x)
        diff = rgba - self.pad_target_tensor
        return torch.mean(diff ** 2, dim=(1, 2, 3, 4))

    def train_step(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single training step with variable iteration count."""
        self.model.train()

        # Fewer iterations for 3D (more expensive)
        iter_n = np.random.randint(48, 64)

        x = x0.clone().to(self.device)

        self.optimizer.zero_grad()

        x.requires_grad_(True)
        for _ in range(iter_n):
            x = self.model(x)

        loss = torch.mean(self.loss_fn(x))
        loss.backward()

        # Normalize gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data / (torch.norm(param.grad.data) + 1e-8)

        self.optimizer.step()

        return x.detach(), loss.detach()

    def train(self, num_steps: Optional[int] = None) -> None:
        """Run training loop."""
        if num_steps is None:
            num_steps = self.config.num_steps

        for step in range(num_steps + 1):
            if self.config.use_pattern_pool:
                batch = self.pool.sample(self.config.batch_size)
                x0 = batch.x.copy()

                # Convert to NCDHW for loss computation
                x0_tensor = torch.tensor(
                    x0.transpose(0, 4, 1, 2, 3),
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.no_grad():
                    loss_rank = self.loss_fn(x0_tensor).cpu().numpy().argsort()[::-1]
                x0 = x0[loss_rank]

                # Replace highest-loss with seed
                x0[0] = self.seed

                # Apply damage
                if self.config.damage_n > 0:
                    damage_masks = make_circle_masks_3d(
                        self.config.damage_n, self.d, self.h, self.w, self.device
                    )
                    damage = 1.0 - damage_masks.cpu().numpy()[..., None]
                    x0[-self.config.damage_n:] *= damage
            else:
                x0 = np.repeat(self.seed[None, ...], self.config.batch_size, axis=0)

            # Convert to NCDHW
            x0_tensor = torch.tensor(
                x0.transpose(0, 4, 1, 2, 3),
                dtype=torch.float32,
                device=self.device,
            )

            x, loss = self.train_step(x0_tensor)

            if self.config.use_pattern_pool:
                # Convert back to NDHWC
                batch.x[:] = x.cpu().numpy().transpose(0, 2, 3, 4, 1)
                batch.commit()

            self.scheduler.step()
            self.loss_log.append(loss.item())

            if step % 10 == 0:
                print(f"\r step: {step}, log10(loss): {np.log10(loss.item()):.3f}", end="")

            if step % self.config.log_every == 0:
                self._visualize_batch(x0_tensor, x, step)

            if step % self.config.save_every == 0 and step > 0:
                self.save_checkpoint(step)

        print()

    def _visualize_batch(
        self,
        x0: torch.Tensor,
        x: torch.Tensor,
        step: int,
    ) -> None:
        """Save visualization of batch (center slice)."""
        try:
            import matplotlib.pyplot as plt

            with torch.no_grad():
                # Take center slice along each axis
                d, h, w = x.shape[2:]

                # Before
                vis0_xy = to_rgb_3d(x0)[0, :, d // 2, :, :].cpu().numpy().transpose(1, 2, 0)
                vis0_xz = to_rgb_3d(x0)[0, :, :, h // 2, :].cpu().numpy().transpose(1, 2, 0)
                vis0_yz = to_rgb_3d(x0)[0, :, :, :, w // 2].cpu().numpy().transpose(1, 2, 0)

                # After
                vis1_xy = to_rgb_3d(x)[0, :, d // 2, :, :].cpu().numpy().transpose(1, 2, 0)
                vis1_xz = to_rgb_3d(x)[0, :, :, h // 2, :].cpu().numpy().transpose(1, 2, 0)
                vis1_yz = to_rgb_3d(x)[0, :, :, :, w // 2].cpu().numpy().transpose(1, 2, 0)

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle(f"Step {step} - Before (top) / After (bottom)")

            axes[0, 0].imshow(np.clip(vis0_xy, 0, 1))
            axes[0, 0].set_title("XY slice (before)")
            axes[0, 1].imshow(np.clip(vis0_xz, 0, 1))
            axes[0, 1].set_title("XZ slice (before)")
            axes[0, 2].imshow(np.clip(vis0_yz, 0, 1))
            axes[0, 2].set_title("YZ slice (before)")

            axes[1, 0].imshow(np.clip(vis1_xy, 0, 1))
            axes[1, 0].set_title("XY slice (after)")
            axes[1, 1].imshow(np.clip(vis1_xz, 0, 1))
            axes[1, 1].set_title("XZ slice (after)")
            axes[1, 2].imshow(np.clip(vis1_yz, 0, 1))
            axes[1, 2].set_title("YZ slice (after)")

            for ax in axes.flat:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(f"{self.config.log_dir}/batch_{step:05d}.png", bbox_inches="tight")
            plt.close()

        except ImportError:
            pass

    def save_checkpoint(self, step: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.log_dir) / f"checkpoint_{step:05d}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_log": self.loss_log,
            "config": self.config,
        }, checkpoint_path)
        print(f"\n  Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns the step number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_log = checkpoint.get("loss_log", [])
        return checkpoint.get("step", 0)

    def export_weights_pickle(self, path: str) -> None:
        """Export weights in pickle format compatible with evolved weights loader."""
        import pickle

        children = list(self.model.dmodel.children())

        # Match the format expected by NCAModel3D.load_weights
        weights = [
            children[0].weight.detach().cpu().numpy().transpose(2, 3, 4, 1, 0),
            children[1].weight.detach().cpu().numpy().transpose(2, 3, 4, 1, 0),
            children[1].bias.detach().cpu().numpy(),
            children[3].weight.detach().cpu().numpy().transpose(2, 3, 4, 1, 0),
            children[3].bias.detach().cpu().numpy(),
        ]

        with open(path, "wb") as f:
            pickle.dump(weights, f)

    def plot_loss(self, save_path: Optional[str] = None) -> None:
        """Plot training loss history."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 4))
            plt.title("Loss history (log10)")
            plt.plot(np.log10(self.loss_log), ".", alpha=0.1)
            plt.xlabel("Step")
            plt.ylabel("log10(loss)")
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            plt.show()

        except ImportError:
            print("matplotlib not available for plotting")


def render_voxels_to_image(
    voxels: npt.NDArray[np.float32],
    axis: int = 1,
    slice_idx: Optional[int] = None,
) -> npt.NDArray[np.float32]:
    """Render voxels to 2D image by taking a slice or max projection."""
    if voxels.shape[-1] in [3, 4]:
        # DHWC format
        pass
    elif voxels.shape[0] in [3, 4, 16]:
        # CDHW format, convert
        voxels = voxels.transpose(1, 2, 3, 0)

    if slice_idx is None:
        slice_idx = voxels.shape[axis] // 2

    if axis == 0:
        return voxels[slice_idx, :, :, :]
    elif axis == 1:
        return voxels[:, slice_idx, :, :]
    else:
        return voxels[:, :, slice_idx, :]

