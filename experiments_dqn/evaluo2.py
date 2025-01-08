from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from snakeenv import SnakeEnv

env = SnakeEnv()

# Cargas tu modelo ya entrenado
model = DQN.load("dqn_snake", env=env)

# Definimos el callback para crear un checkpoint cada 10.000 pasos
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./checkpoints/',
    name_prefix='dqn_snake'
)

# Continuar el entrenamiento 
model.learn(total_timesteps=100000, callback=checkpoint_callback)
