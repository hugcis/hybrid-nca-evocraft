"""Training utilities for Neural Cellular Automata.

"""

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import PIL.Image
import requests
import torch
import torch.nn.functional as F

from .models import CHANNEL_N


def load_image(url: str, max_size: int = 40) -> npt.NDArray[np.float32]:
    """Load and preprocess an image from URL."""
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.Resampling.LANCZOS)
    img = np.float32(img) / 255.0
    # Premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def load_emoji(emoji: str, max_size: int = 40) -> npt.NDArray[np.float32]:
    """Load an emoji image from Google Noto Emoji."""
    code = hex(ord(emoji))[2:].lower()
    url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true"
    return load_image(url, max_size)


def load_image_file(path: str, max_size: int = 40) -> npt.NDArray[np.float32]:
    """Load and preprocess an image from local file."""
    img = PIL.Image.open(path)
    img.thumbnail((max_size, max_size), PIL.Image.Resampling.LANCZOS)
    # Convert to RGBA if needed
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img = np.float32(img) / 255.0
    # Premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def to_rgba(x: torch.Tensor) -> torch.Tensor:
    """Extract RGBA channels (first 4 channels)."""
    return x[:, :4, :, :]


def to_alpha(x: torch.Tensor) -> torch.Tensor:
    """Extract and clamp alpha channel."""
    return torch.clamp(x[:, 3:4, :, :], 0.0, 1.0)


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Convert to RGB assuming premultiplied alpha."""
    rgb, a = x[:, :3, :, :], to_alpha(x)
    return 1.0 - a + rgb


def make_seed(
    size: int,
    n: int = 1,
    channel_n: int = CHANNEL_N,
) -> torch.Tensor:
    """Create initial seed state with a single living cell in center."""
    x = torch.zeros(n, channel_n, size, size, dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1.0
    return x


def make_circle_masks(
    n: int,
    h: int,
    w: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create random circular masks for damage."""
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]
    center = torch.rand(2, n, 1, 1, device=device) * 1.0 - 0.5
    r = torch.rand(n, 1, 1, device=device) * 0.3 + 0.1
    x_centered = (x - center[0]) / r
    y_centered = (y - center[1]) / r
    mask = (x_centered * x_centered + y_centered * y_centered < 1.0).float()
    return mask


class SamplePool:
    """Pool of training samples for persistent/regenerating training."""

    def __init__(self, *, _parent: Optional["SamplePool"] = None, _parent_idx: Optional[npt.NDArray] = None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = list(slots.keys())
        self._size: Optional[int] = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n: int) -> "SamplePool":
        """Sample n items from the pool."""
        idx = np.random.choice(self._size, n, replace=False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        return SamplePool(**batch, _parent=self, _parent_idx=idx)

    def commit(self) -> None:
        """Commit changes back to parent pool."""
        if self._parent is not None:
            for k in self._slot_names:
                getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


@dataclass
class TrainingConfig:
    """Configuration for NCA training."""

    channel_n: int = CHANNEL_N
    target_padding: int = 16
    target_size: int = 40
    batch_size: int = 8
    pool_size: int = 1024
    cell_fire_rate: float = 0.5

    # Training settings
    learning_rate: float = 2e-3
    lr_decay_steps: int = 2000
    lr_decay_factor: float = 0.1
    num_steps: int = 8000

    # Experiment type: 0=Growing, 1=Persistent, 2=Regenerating
    experiment_type: int = 2
    use_pattern_pool: bool = True
    damage_n: int = 3  # Number of patterns to damage per batch

    # Logging
    log_every: int = 100
    save_every: int = 500
    log_dir: str = "train_log"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def growing(cls) -> "TrainingConfig":
        """Config for growing experiment (no pool, no damage)."""
        return cls(experiment_type=0, use_pattern_pool=False, damage_n=0)

    @classmethod
    def persistent(cls) -> "TrainingConfig":
        """Config for persistent experiment (pool, no damage)."""
        return cls(experiment_type=1, use_pattern_pool=True, damage_n=0)

    @classmethod
    def regenerating(cls) -> "TrainingConfig":
        """Config for regenerating experiment (pool + damage)."""
        return cls(experiment_type=2, use_pattern_pool=True, damage_n=3)


class NCATrainer:
    """Trainer for Neural Cellular Automata."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_img: npt.NDArray[np.float32],
        config: TrainingConfig,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)

        # Pad target image
        p = config.target_padding
        self.pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
        self.h, self.w = self.pad_target.shape[:2]

        # Convert to torch tensor (NCHW format)
        self.pad_target_tensor = torch.tensor(
            self.pad_target.transpose(2, 0, 1)[None, :4, :, :],
            dtype=torch.float32,
            device=self.device,
        )

        # Create seed
        self.seed = np.zeros([self.h, self.w, config.channel_n], np.float32)
        self.seed[self.h // 2, self.w // 2, 3:] = 1.0
        # Convert to NCHW format
        self.seed_tensor = torch.tensor(
            self.seed.transpose(2, 0, 1)[None, :, :, :],
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize sample pool
        seed_batch = np.repeat(self.seed[None, ...], config.pool_size, axis=0)
        self.pool = SamplePool(x=seed_batch)

        # Optimizer with learning rate schedule
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_steps,
            gamma=config.lr_decay_factor,
        )

        self.loss_log: list[float] = []

        # Create log directory
        os.makedirs(config.log_dir, exist_ok=True)

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE loss against target RGBA."""
        rgba = to_rgba(x)
        diff = rgba - self.pad_target_tensor
        return torch.mean(diff ** 2, dim=(1, 2, 3))

    def train_step(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single training step with variable iteration count."""
        self.model.train()

        # Random number of iterations between 64 and 96
        iter_n = np.random.randint(64, 96)

        x = x0.clone().requires_grad_(False)
        x = x.to(self.device)

        # Forward pass through multiple CA steps
        self.optimizer.zero_grad()

        # Need to track gradients through all steps
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
                # Sample from pool
                batch = self.pool.sample(self.config.batch_size)
                x0 = batch.x.copy()

                # Sort by loss (highest first)
                x0_tensor = torch.tensor(
                    x0.transpose(0, 3, 1, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.no_grad():
                    loss_rank = self.loss_fn(x0_tensor).cpu().numpy().argsort()[::-1]
                x0 = x0[loss_rank]

                # Replace highest-loss sample with seed
                x0[0] = self.seed

                # Apply damage to some samples
                if self.config.damage_n > 0:
                    damage_masks = make_circle_masks(
                        self.config.damage_n, self.h, self.w, self.device
                    )
                    damage = 1.0 - damage_masks.cpu().numpy()[..., None]
                    x0[-self.config.damage_n:] *= damage
            else:
                # Use fresh seeds
                x0 = np.repeat(self.seed[None, ...], self.config.batch_size, axis=0)

            # Convert to NCHW format for PyTorch
            x0_tensor = torch.tensor(
                x0.transpose(0, 3, 1, 2),
                dtype=torch.float32,
                device=self.device,
            )

            # Training step
            x, loss = self.train_step(x0_tensor)

            # Update pool
            if self.config.use_pattern_pool:
                # Convert back to NHWC for pool storage
                batch.x[:] = x.cpu().numpy().transpose(0, 2, 3, 1)
                batch.commit()

            self.scheduler.step()
            self.loss_log.append(loss.item())

            # Logging
            if step % 10 == 0:
                print(f"\r step: {step}, log10(loss): {np.log10(loss.item()):.3f}", end="")

            if step % self.config.log_every == 0:
                self._visualize_batch(x0_tensor, x, step)

            if step % self.config.save_every == 0 and step > 0:
                self.save_checkpoint(step)

        print()  # Newline after training

    def _visualize_batch(
        self,
        x0: torch.Tensor,
        x: torch.Tensor,
        step: int,
    ) -> None:
        """Save visualization of batch before/after."""
        try:
            import matplotlib.pyplot as plt

            with torch.no_grad():
                vis0 = to_rgb(x0).cpu().numpy()
                vis1 = to_rgb(x).cpu().numpy()

            # Stack horizontally then vertically
            vis0_row = np.hstack([vis0[i].transpose(1, 2, 0) for i in range(vis0.shape[0])])
            vis1_row = np.hstack([vis1[i].transpose(1, 2, 0) for i in range(vis1.shape[0])])
            vis = np.vstack([vis0_row, vis1_row])

            plt.figure(figsize=(16, 4))
            plt.imshow(np.clip(vis, 0, 1))
            plt.title(f"Step {step} - Before (top) / After (bottom)")
            plt.axis("off")
            plt.savefig(f"{self.config.log_dir}/batch_{step:05d}.png", bbox_inches="tight")
            plt.close()

        except ImportError:
            pass  # Skip visualization if matplotlib not available

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

        # Extract weights in the format expected by load_weights
        children = list(self.model.dmodel.children())

        weights = [
            children[1].weight.detach().cpu().numpy().transpose(2, 3, 1, 0),
            children[1].bias.detach().cpu().numpy(),
            children[3].weight.detach().cpu().numpy().transpose(2, 3, 1, 0),
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


def generate_video(
    model: torch.nn.Module,
    seed: torch.Tensor,
    num_steps: int = 200,
    output_path: str = "growth.mp4",
    fps: float = 30.0,
    scale: int = 4,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Generate a video of the NCA growing from seed.

    Args:
        model: Trained NCA model
        seed: Initial seed tensor (1, C, H, W)
        num_steps: Number of CA steps to simulate
        output_path: Path to save the video
        fps: Frames per second
        scale: Upscaling factor for visualization
        device: Device to run on
    """
    try:
        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
    except ImportError:
        print("moviepy not available for video generation")
        return

    model = model.to(device)
    model.eval()

    x = seed.clone().to(device)

    # Get dimensions
    _, _, h, w = x.shape

    with FFMPEG_VideoWriter(output_path, (w * scale, h * scale), fps) as writer:
        with torch.no_grad():
            for _ in range(num_steps):
                # Convert to RGB image
                rgb = to_rgb(x).cpu().numpy()[0]  # (3, H, W)
                rgb = rgb.transpose(1, 2, 0)  # (H, W, 3)
                rgb = np.clip(rgb, 0, 1)

                # Upscale
                img = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
                img = (img * 255).astype(np.uint8)

                writer.write_frame(img)

                # Step the CA
                x = model(x)

    print(f"Video saved to {output_path}")


def generate_growth_sequence(
    model: torch.nn.Module,
    seed: torch.Tensor,
    steps: list[int],
    device: torch.device = torch.device("cpu"),
) -> list[npt.NDArray[np.float32]]:
    """Generate images at specific steps during growth.

    Args:
        model: Trained NCA model
        seed: Initial seed tensor (1, C, H, W)
        steps: List of step numbers to capture
        device: Device to run on

    Returns:
        List of RGB images as numpy arrays
    """
    model = model.to(device)
    model.eval()

    x = seed.clone().to(device)
    images = []
    current_step = 0

    steps = sorted(steps)
    step_idx = 0

    with torch.no_grad():
        while step_idx < len(steps):
            if current_step == steps[step_idx]:
                rgb = to_rgb(x).cpu().numpy()[0].transpose(1, 2, 0)
                images.append(np.clip(rgb, 0, 1))
                step_idx += 1

            x = model(x)
            current_step += 1

    return images


def tile_images(
    images: list[npt.NDArray[np.float32]],
    cols: Optional[int] = None,
) -> npt.NDArray[np.float32]:
    """Tile a list of images into a grid.

    Args:
        images: List of images (H, W, C)
        cols: Number of columns (default: sqrt of num images)

    Returns:
        Tiled image as numpy array
    """
    n = len(images)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    h, w, c = images[0].shape

    # Pad list to fill grid
    while len(images) < rows * cols:
        images.append(np.ones((h, w, c), dtype=np.float32))

    # Create grid
    grid = []
    for row in range(rows):
        row_imgs = images[row * cols : (row + 1) * cols]
        grid.append(np.hstack(row_imgs))

    return np.vstack(grid)


def zoom_image(img: npt.NDArray, scale: int = 4) -> npt.NDArray:
    """Upscale an image by repeating pixels."""
    img = np.repeat(img, scale, axis=0)
    img = np.repeat(img, scale, axis=1)
    return img

