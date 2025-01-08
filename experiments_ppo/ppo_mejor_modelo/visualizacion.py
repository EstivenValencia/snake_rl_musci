from stable_baselines3 import PPO
from snakeenv import SnakeEnv
from stable_baselines3.common.env_util import make_vec_env

env = SnakeEnv(render='human')

# Cargar el modelo previamente guardado
model = PPO.load("/mnt/d/home-2/Documentos/master/RL/snake_env/ppo_mejor_modelo/checkpoints/dqn_snake_130000_steps.zip")

obs, _ = env.reset()
for _ in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncate, info = env.step(action)

    if dones:
        obs, _ = env.reset()
        print("Reward:", rewards, 'dones:', dones, 'truncate:', truncate)