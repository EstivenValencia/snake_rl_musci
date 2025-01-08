from stable_baselines3 import PPO
from snakeenv import SnakeEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback


# Se instancia el entorno
env = SnakeEnv(render='human')
env_human = SnakeEnv(render='human')

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./checkpoints/',
    name_prefix='dqn_snake'
)


# Crear el modelo PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0008644803614195511, n_steps=512, batch_size=128, n_epochs=13, gamma=0.9, tensorboard_log='logs', device='cpu')

###########################
#       ENTRENAMIENTO
###########################

# Entrenar el modelo durante un número determinado de timesteps
model.learn(total_timesteps=1500000, callback=checkpoint_callback) 

# Guardar el modelo entrenado
model.save("ppo_basic_snake_sin_paralelizar")
del model

###########################
#      EVALUACIÓN
###########################

# Cargar el modelo previamente guardado
model = PPO.load("ppo_snake_final.zip")

# Evaluar el modelo cargado en el entorno
# n_eval_episodes: número de episodios para la evaluación
mean_reward, std_reward = evaluate_policy(model, env_human, n_eval_episodes=20)

# Imprimir la recompensa media y la desviación estándar de la evaluación
print(f'Mean reward: {mean_reward} +/- {std_reward}')