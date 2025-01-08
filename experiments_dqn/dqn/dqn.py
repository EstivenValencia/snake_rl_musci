from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from snakeenv import SnakeEnv

# Crear el entorno
env = SnakeEnv()

# Callback para guardar checkpoints cada 10.000 pasos
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./checkpoints/',
    name_prefix='dqn_snake'
)

# Entrenamiento con registro de TensorBoard
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"  # Directorio de logs para TensorBoard
)

model.learn(
    total_timesteps=100000,  # Entrenamiento corto inicial
    callback=checkpoint_callback
)

# Guardar el modelo final
model.save("dqn_snake_final")
