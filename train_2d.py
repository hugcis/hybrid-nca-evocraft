#!/usr/bin/env python3
"""Train a 2D Neural Cellular Automata to grow a target image.

Converted from TensorFlow implementation:
https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb

Usage:
    # Train on an emoji (requires network to download)
    python train_2d.py --emoji 🦎

    # Train on a local image file
    python train_2d.py --image path/to/image.png

    # Train with different experiment types
    python train_2d.py --emoji 🦋 --experiment growing      # No pool, no damage
    python train_2d.py --emoji 🦋 --experiment persistent   # Pool, no damage
    python train_2d.py --emoji 🦋 --experiment regenerating # Pool + damage (default)

    # Resume from checkpoint
    python train_2d.py --emoji 🦎 --resume train_log/checkpoint_02000.pt
"""

import argparse
from pathlib import Path

import numpy as np

from evo_ca.models import NCAModel
from evo_ca.training import (
    NCATrainer,
    TrainingConfig,
    load_emoji,
    load_image_file,
    to_rgb,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 2D NCA to grow a target image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target image options (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--emoji",
        type=str,
        help="Target emoji character (e.g., 🦎)",
    )
    target_group.add_argument(
        "--image",
        type=str,
        help="Path to target image file (PNG with alpha channel)",
    )

    # Experiment type
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["growing", "persistent", "regenerating"],
        default="regenerating",
        help="Experiment type (default: regenerating)",
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
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1024,
        help="Sample pool size (default: 1024)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=40,
        help="Target image size (default: 40)",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="train_log",
        help="Directory for logs and checkpoints (default: train_log)",
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

    # Load target image
    print("Loading target image...")
    if args.emoji:
        target_img = load_emoji(args.emoji, max_size=args.target_size)
        print(f"  Loaded emoji: {args.emoji}")
    else:
        target_img = load_image_file(args.image, max_size=args.target_size)
        print(f"  Loaded image: {args.image}")

    print(f"  Target shape: {target_img.shape}")

    # Create config based on experiment type
    if args.experiment == "growing":
        config = TrainingConfig.growing()
    elif args.experiment == "persistent":
        config = TrainingConfig.persistent()
    else:
        config = TrainingConfig.regenerating()

    # Override with command line args
    config.num_steps = args.steps
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.pool_size = args.pool_size
    config.target_size = args.target_size
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
    model = NCAModel(channel_n=config.channel_n, fire_rate=config.cell_fire_rate)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    trainer = NCATrainer(model, target_img, config)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_step = trainer.load_checkpoint(args.resume)
        print(f"  Resumed at step {start_step}")

    # Create log directory
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Save target image for reference
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.imshow(target_img)
        plt.title("Target")
        plt.axis("off")
        plt.savefig(f"{config.log_dir}/target.png", bbox_inches="tight")
        plt.close()
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


if __name__ == "__main__":
    main()

