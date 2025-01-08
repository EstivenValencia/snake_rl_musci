from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from snakeenv import SnakeEnv

# Crear el entorno
env = SnakeEnv()

# Mejores hiperparámetros encontrados en la búsqueda
best_params = {
    "learning_rate": 0.008985977014133648,
    "buffer_size": 200000,
    "batch_size": 32,
    "gamma": 0.99,
    "train_freq": 4,
    "tau": 0.3415295548171653,
}

# Configurar callback para guardar checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10000,            # Guardar cada 10.000 pasos
    save_path='./checkpoints_refinal/',
    name_prefix='dqn_snake_reoptimized'
)

# Crear y configurar el modelo con los mejores hiperparámetros
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=best_params["learning_rate"],
    buffer_size=best_params["buffer_size"],
    batch_size=best_params["batch_size"],
    gamma=best_params["gamma"],
    train_freq=best_params["train_freq"],
    tau=best_params["tau"],
    verbose=1,
    tensorboard_log="./tensorboard_logs_optimized/"
)

# Entrenar el modelo por 200.000 pasos
model.learn(total_timesteps=200000, callback=checkpoint_callback)

# Guardar el modelo final optimizado
model.save("dqn_snake_reoptimized")
