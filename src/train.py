import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from model import make_env, create_model
from config import CONFIG


class RewardCallback(BaseCallback):
    # tracks total reward per episode so we can plot it later
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
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["assets_dir"], exist_ok=True)

    env = make_env()
    model = create_model(env)
    callback = RewardCallback()

    # save before any training so we can show random behavior in the video
    model.save(f"{CONFIG['model_dir']}/untrained")

    half_steps = CONFIG["total_timesteps"] // 2
    model.learn(total_timesteps=half_steps, callback=callback)
    model.save(f"{CONFIG['model_dir']}/half_trained")

    # reset_num_timesteps=False continues the step counter rather than resetting
    model.learn(total_timesteps=half_steps, callback=callback, reset_num_timesteps=False)
    model.save(f"{CONFIG['model_dir']}/trained")

    plot_rewards(callback.episode_rewards, f"{CONFIG['assets_dir']}/reward_plot.png")
    env.close()


if __name__ == "__main__":
    train()
