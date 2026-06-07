
CONFIG = {
    "env_id": "highway-v0",
    "learning_rate": 5e-4,
    "n_steps": 256,
    "batch_size": 64,
    "gamma": 0.8,
    "total_timesteps": 20000,
    "model_dir": "checkpoints",
    "assets_dir": "assets",
}

ENV_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "reward_speed_range": [20, 30],
}