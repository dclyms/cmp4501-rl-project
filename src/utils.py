import os
import imageio
import numpy as np


def combine_gifs(gif_paths: list[str], output_path: str) -> None:
    """Combine multiple GIFs into one sequential GIF."""
    all_frames: list[np.ndarray] = []

    for path in gif_paths:
        frames = imageio.mimread(path)
        all_frames.extend(frames)

    imageio.mimsave(output_path, all_frames, fps=15)
    print(f"Combined GIF saved to {output_path}")


def ensure_dirs(dirs: list[str]) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)