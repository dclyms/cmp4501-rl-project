import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from model import make_env, create_model
from config import CONFIG


class RewardCallback(BaseCallback):
    """Callback to track and save episode rewards during training."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []
        self.current_rewards: list[float] = []

    def _on_step(self) -> bool:
        self.current_rewards.append(np.mean(self.locals["rewards"]))
        if any(self.locals["dones"]):
            self.episode_rewards.append(np.sum(self.current_rewards))
            self.current_rewards = []
        return True


def plot_rewards(rewards: list[float], path: str) -> None:
    """Plot and save the reward curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Training Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print(f"Reward plot saved to {path}")


def train() -> None:
    """Main training loop with checkpoint saving."""
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["assets_dir"], exist_ok=True)

    env = make_env()
    model = create_model(env)
    callback = RewardCallback()

    # Save untrained checkpoint
    model.save(f"{CONFIG['model_dir']}/untrained")
    print("Saved untrained checkpoint")

    # Train to halfway
    half_steps = CONFIG["total_timesteps"] // 2
    model.learn(total_timesteps=half_steps, callback=callback)
    model.save(f"{CONFIG['model_dir']}/half_trained")
    print("Saved half-trained checkpoint")

    # Train to completion
    model.learn(total_timesteps=half_steps, callback=callback, reset_num_timesteps=False)
    model.save(f"{CONFIG['model_dir']}/trained")
    print("Saved trained checkpoint")

    # Plot rewards
    plot_rewards(callback.episode_rewards, f"{CONFIG['assets_dir']}/reward_plot.png")
    env.close()


if __name__ == "__main__":
    train()