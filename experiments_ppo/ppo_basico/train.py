from stable_baselines3 import PPO
from snakeenv import SnakeEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Se instancia el entorno
env = SnakeEnv(render=None)
env_human = SnakeEnv(render='human')

# Crear el modelo PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, n_steps=1024, batch_size=32, n_epochs=3, tensorboard_log='logs', device='cpu')

###########################
#       ENTRENAMIENTO
###########################

# Entrenar el modelo durante un número determinado de timesteps
model.learn(total_timesteps=100000) 

# Guardar el modelo entrenado
model.save("ppo_basic_snake")
del model

###########################
#      EVALUACIÓN
###########################

# Cargar el modelo previamente guardado
model = PPO.load("ppo_basic_snake.zip")

# Evaluar el modelo cargado en el entorno
# n_eval_episodes: número de episodios para la evaluación
mean_reward, std_reward = evaluate_policy(model, env_human, n_eval_episodes=20)

# Imprimir la recompensa media y la desviación estándar de la evaluación
print(f'Mean reward: {mean_reward} +/- {std_reward}')