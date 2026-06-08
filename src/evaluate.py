import gymnasium as gym
import highway_env
import imageio
import numpy as np
from stable_baselines3 import PPO
from config import CONFIG


def record_agent(model_path: str, output_path: str, steps: int = 150) -> None:
    # render_mode="rgb_array" is needed on Windows since the display window doesn't work
    env = gym.make(CONFIG["env_id"], render_mode="rgb_array")

    if model_path == "random":
        model = None
    else:
        model = PPO.load(model_path)

    frames: list[np.ndarray] = []
    obs, _ = env.reset()

    for _ in range(steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        frames.append(env.render())
        if done or truncated:
            obs, _ = env.reset()

    imageio.mimsave(output_path, frames, fps=15)
    print(f"Saved: {output_path}")
    env.close()


def evaluate() -> None:
    record_agent("random",                   "assets/stage1_untrained.gif")
    record_agent("checkpoints/half_trained", "assets/stage2_half.gif")
    record_agent("checkpoints/trained",      "assets/stage3_trained.gif")


if __name__ == "__main__":
    evaluate()
