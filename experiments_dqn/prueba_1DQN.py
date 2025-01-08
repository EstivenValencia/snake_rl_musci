from stable_baselines3 import DQN
from snakeenv import SnakeEnv

# Crear el entorno
env = SnakeEnv()

# Configurar la carpeta de logs para TensorBoard
log_dir = "./tensorboard_logs/DQN_run_1"  # Cambia el nombre para cada ejecución

# Crear el modelo DQN con logs habilitados
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,  # Guardar logs aquí
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
)

# Entrenar el modelo durante un número determinado de timesteps
model.learn(total_timesteps=200000)  # Ajusta este número según el tiempo y recursos disponibles

# Guardar el modelo entrenado
model.save("dqn_snake")

print(f"Logs guardados en: {log_dir}")
