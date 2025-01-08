from stable_baselines3 import PPO
from snakeenv import SnakeEnv
from stable_baselines3.common.env_util import make_vec_env

env = SnakeEnv(render='human')

# Cargar el modelo previamente guardado
model = PPO.load("ppo_basic_snake_sin_paralelizar.zip")

for _ in range(10):
    obs, info = env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncate, info = env.step(action)