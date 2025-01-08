from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from snakeenv import SnakeEnv

# Crear el entorno
env = SnakeEnv()

# Cargar el modelo optimizado
model = DQN.load("dqn_snake_optimized", env=env)

# Evaluar el modelo
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Recompensa media del modelo optimizado: {mean_reward} ± {std_reward}")

# Visualizar el comportamiento del modelo
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
