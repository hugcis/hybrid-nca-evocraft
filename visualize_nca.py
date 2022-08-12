#!/usr/bin/env python3
"""Visualize trained NCA models.

Generate videos and figures from trained NCA checkpoints.

Usage:
    # Generate growth video from checkpoint
    python visualize_nca.py --checkpoint train_log/checkpoint_08000.pt --video growth.mp4

    # Generate growth video from pickle weights
    python visualize_nca.py --weights models/duck.pkl --video growth.mp4

    # Generate growth sequence figure
    python visualize_nca.py --checkpoint train_log/checkpoint_08000.pt --sequence growth_seq.png

    # Test regeneration (apply damage and watch recovery)
    python visualize_nca.py --checkpoint train_log/checkpoint_08000.pt --regen regen.mp4
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from evo_ca.models import NCAModel
from evo_ca.training import (
    generate_growth_sequence,
    generate_video,
    make_seed,
    tile_images,
    to_rgb,
    zoom_image,
)


def load_model_from_checkpoint(path: str, device: str = "cpu") -> tuple[NCAModel, int, int]:
    """Load model from training checkpoint.

    Returns:
        model: Loaded NCAModel
        h: Height of the training grid
        w: Width of the training grid
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config")

    channel_n = config.channel_n if config else 16
    fire_rate = config.cell_fire_rate if config else 0.5

    model = NCAModel(channel_n=channel_n, fire_rate=fire_rate)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Estimate size from config
    target_size = config.target_size if config else 40
    padding = config.target_padding if config else 16
    size = target_size + 2 * padding

    return model, size, size


def load_model_from_weights(path: str) -> tuple[NCAModel, int, int]:
    """Load model from pickle weights file.

    Returns:
        model: Loaded NCAModel
        h: Default height
        w: Default width
    """
    with open(path, "rb") as f:
        weights = pickle.load(f)

    model = NCAModel()
    model.load_weights(weights)
    model.eval()

    # Default size for evolved models
    return model, 72, 72


def generate_regen_video(
    model: torch.nn.Module,
    seed: torch.Tensor,
    output_path: str = "regen.mp4",
    grow_steps: int = 200,
    damage_step: int = 200,
    total_steps: int = 600,
    fps: float = 30.0,
    scale: int = 4,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Generate video showing growth and regeneration after damage.

    Args:
        model: Trained NCA model
        seed: Initial seed tensor (1, C, H, W)
        output_path: Path to save the video
        grow_steps: Steps before applying damage
        damage_step: Step at which to apply damage
        total_steps: Total steps to simulate
        fps: Frames per second
        scale: Upscaling factor
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
    _, c, h, w = x.shape

    with FFMPEG_VideoWriter(output_path, (w * scale, h * scale), fps) as writer:
        with torch.no_grad():
            for step in range(total_steps):
                # Apply damage at specified step
                if step == damage_step:
                    # Damage: remove bottom half
                    x[:, :, h // 2:, :] = 0

                # Convert to RGB image
                rgb = to_rgb(x).cpu().numpy()[0]
                rgb = rgb.transpose(1, 2, 0)
                rgb = np.clip(rgb, 0, 1)

                # Upscale
                img = zoom_image(rgb, scale)
                img = (img * 255).astype(np.uint8)

                writer.write_frame(img)

                # Step the CA
                x = model(x)

    print(f"Regeneration video saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize trained NCA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to training checkpoint (.pt file)",
    )
    model_group.add_argument(
        "--weights",
        type=str,
        help="Path to pickle weights file (.pkl file)",
    )

    # Output options
    parser.add_argument(
        "--video",
        type=str,
        help="Generate growth video and save to this path",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Generate growth sequence figure and save to this path",
    )
    parser.add_argument(
        "--regen",
        type=str,
        help="Generate regeneration video and save to this path",
    )

    # Parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of CA steps for video/sequence (default: 300)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS (default: 30)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Image scale factor (default: 4)",
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Override grid size (default: auto-detect)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    if args.checkpoint:
        model, h, w = load_model_from_checkpoint(args.checkpoint, args.device)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        model, h, w = load_model_from_weights(args.weights)
        print(f"  Loaded weights: {args.weights}")

    # Override size if specified
    if args.size:
        h = w = args.size

    print(f"  Grid size: {h}x{w}")

    device = torch.device(args.device)
    model = model.to(device)

    # Create seed
    seed = make_seed(h, n=1, channel_n=model.channel_n).to(device)

    # Generate outputs
    if args.video:
        print(f"\nGenerating growth video ({args.steps} steps)...")
        generate_video(
            model, seed,
            num_steps=args.steps,
            output_path=args.video,
            fps=args.fps,
            scale=args.scale,
            device=device,
        )

    if args.sequence:
        print(f"\nGenerating growth sequence...")
        # Capture at regular intervals
        capture_steps = [0, 20, 50, 100, 150, 200, 250, 300]
        capture_steps = [s for s in capture_steps if s <= args.steps]

        images = generate_growth_sequence(model, seed, capture_steps, device)

        # Create tiled figure
        tiled = tile_images(images, cols=4)
        tiled = zoom_image(tiled, args.scale)

        # Save
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(16, 8))
            plt.imshow(tiled)
            plt.title(f"Growth sequence (steps: {capture_steps})")
            plt.axis("off")
            plt.savefig(args.sequence, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved to {args.sequence}")
        except ImportError:
            # Fallback to PIL
            import PIL.Image
            img = (np.clip(tiled, 0, 1) * 255).astype(np.uint8)
            PIL.Image.fromarray(img).save(args.sequence)
            print(f"  Saved to {args.sequence}")

    if args.regen:
        print(f"\nGenerating regeneration video...")
        generate_regen_video(
            model, seed,
            output_path=args.regen,
            grow_steps=200,
            damage_step=200,
            total_steps=args.steps + 200,
            fps=args.fps,
            scale=args.scale,
            device=device,
        )

    if not any([args.video, args.sequence, args.regen]):
        print("\nNo output specified. Use --video, --sequence, or --regen to generate output.")


if __name__ == "__main__":
    main()

