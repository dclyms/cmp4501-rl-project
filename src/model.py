from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import highway_env
from config import CONFIG, ENV_CONFIG


def make_env() -> object:
    """Create and configure the highway environment."""
    env = make_vec_env(
        CONFIG["env_id"],
        n_envs=4,
        env_kwargs={"config": ENV_CONFIG},
    )
    return env


def create_model(env: object) -> PPO:
    """Create a PPO model with configured hyperparameters."""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        gamma=CONFIG["gamma"],
        verbose=1,
    )
    return model


def load_model(path: str, env: object) -> PPO:
    """Load a saved PPO model from disk."""
    return PPO.load(path, env=env)