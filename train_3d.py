#!/usr/bin/env python3
"""Train a 3D Neural Cellular Automata to grow Minecraft-style voxel shapes.

Usage:
    # Train on a built-in shape
    python train_3d.py --shape sphere
    python train_3d.py --shape cube
    python train_3d.py --shape pyramid
    python train_3d.py --shape torus
    python train_3d.py --shape cross

    # Train on a custom voxel file (.npy)
    python train_3d.py --voxel path/to/shape.npy

    # Different experiment types
    python train_3d.py --shape sphere --experiment growing      # No pool, no damage
    python train_3d.py --shape sphere --experiment persistent   # Pool, no damage
    python train_3d.py --shape sphere --experiment regenerating # Pool + damage (default)

    # Customize training
    python train_3d.py --shape sphere --size 20 --steps 5000 --batch-size 2

    # Resume from checkpoint
    python train_3d.py --shape sphere --resume train_log_3d/checkpoint_02000.pt
"""

import argparse
from pathlib import Path

import numpy as np

from evo_ca.models import NCAModel3D
from evo_ca.training_3d import (
    BUILTIN_SHAPES,
    NCATrainer3D,
    Training3DConfig,
    get_target_shape,
    render_voxels_to_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 3D NCA to grow voxel shapes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target shape options (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--shape",
        type=str,
        choices=list(BUILTIN_SHAPES.keys()),
        help=f"Built-in shape: {list(BUILTIN_SHAPES.keys())}",
    )
    target_group.add_argument(
        "--voxel",
        type=str,
        help="Path to voxel file (.npy with shape DxHxWx4 RGBA or DxHxW binary)",
    )

    # Experiment type
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["growing", "persistent", "regenerating"],
        default="regenerating",
        help="Experiment type (default: regenerating)",
    )

    # Shape parameters
    parser.add_argument(
        "--size",
        type=int,
        default=16,
        help="Target shape size (default: 16, used for built-in shapes)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default=None,
        help="Shape color as 'R,G,B' (0-1 range), e.g., '0.8,0.2,0.2'",
    )

    # Training parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=8000,
        help="Number of training steps (default: 8000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
        help="Learning rate (default: 2e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4, smaller for 3D due to memory)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=256,
        help="Sample pool size (default: 256)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=4,
        help="Target padding (default: 4)",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="train_log_3d",
        help="Directory for logs and checkpoints (default: train_log_3d)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log visualization every N steps (default: 100)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (default: cuda, falls back to cpu)",
    )

    args = parser.parse_args()

    # Parse color if provided
    color_kwargs = {}
    if args.color:
        try:
            r, g, b = map(float, args.color.split(","))
            color_kwargs["color"] = (r, g, b)
        except ValueError:
            print(f"Warning: Could not parse color '{args.color}', using default")

    # Load target shape
    print("Loading target shape...")
    if args.shape:
        target_voxels = get_target_shape(args.shape, size=args.size, **color_kwargs)
        print(f"  Created {args.shape} shape")
    else:
        target_voxels = get_target_shape(args.voxel)
        print(f"  Loaded voxel file: {args.voxel}")

    print(f"  Shape: {target_voxels.shape}")
    num_voxels = np.sum(target_voxels[..., 3] > 0.5)
    print(f"  Active voxels: {num_voxels}")

    # Create config
    if args.experiment == "growing":
        config = Training3DConfig.growing()
    elif args.experiment == "persistent":
        config = Training3DConfig.persistent()
    else:
        config = Training3DConfig.regenerating()

    # Override with command line args
    config.num_steps = args.steps
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.pool_size = args.pool_size
    config.target_size = args.size
    config.target_padding = args.padding
    config.log_dir = args.log_dir
    config.log_every = args.log_every
    config.save_every = args.save_every
    config.device = args.device

    print(f"\nTraining config:")
    print(f"  Experiment type: {args.experiment}")
    print(f"  Use pattern pool: {config.use_pattern_pool}")
    print(f"  Damage N: {config.damage_n}")
    print(f"  Steps: {config.num_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device: {config.device}")

    # Create model
    model = NCAModel3D(channel_n=config.channel_n, fire_rate=config.cell_fire_rate)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {num_params:,} parameters")

    # Create trainer
    trainer = NCATrainer3D(model, target_voxels, config)

    # Resume if specified
    start_step = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_step = trainer.load_checkpoint(args.resume)
        print(f"  Resumed at step {start_step}")

    # Create log directory
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Save target visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle("Target Shape (XY, XZ, YZ slices)")

        for i, axis in enumerate([0, 1, 2]):
            slice_img = render_voxels_to_image(target_voxels, axis=axis)
            if slice_img.shape[-1] == 4:
                # Composite RGBA over white
                rgb = slice_img[..., :3]
                alpha = slice_img[..., 3:4]
                slice_img = 1.0 - alpha + rgb * alpha
            axes[i].imshow(np.clip(slice_img, 0, 1))
            axes[i].set_title(f"Slice axis {axis}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(f"{config.log_dir}/target.png", bbox_inches="tight")
        plt.close()
        print(f"\nSaved target visualization to {config.log_dir}/target.png")
    except ImportError:
        pass

    # Train
    print(f"\nStarting training...")
    trainer.train(num_steps=config.num_steps - start_step)

    # Plot loss
    trainer.plot_loss(save_path=f"{config.log_dir}/loss.png")

    # Export final weights
    final_weights_path = f"{config.log_dir}/final_weights.pkl"
    trainer.export_weights_pickle(final_weights_path)
    print(f"\nExported final weights to {final_weights_path}")

    # Save final checkpoint
    trainer.save_checkpoint(config.num_steps)

    print("\nTraining complete!")
    print(f"\nTo run the trained model in Minecraft:")
    print(f"  python run_3d.py {final_weights_path}")


if __name__ == "__main__":
    main()

