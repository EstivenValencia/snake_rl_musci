import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from snakeenv import SnakeEnv
import os
import json

# Definir la función objetivo para la optimización
def objective(trial):
    # Espacio de búsqueda para los hiperparámetros
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_uniform("gamma", 0.8, 0.99)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
    tau = trial.suggest_uniform("tau", 0.01, 1.0)

    # Crear el entorno
    env = SnakeEnv()

    # Configurar el modelo con los hiperparámetros actuales
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        tau=tau,
        verbose=0
    )

    # Entrenar por 100.000 pasos como prueba
    model.learn(total_timesteps=100000)

    # Evaluar el modelo
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    return mean_reward  # Queremos maximizar la recompensa media

# Crear el estudio de Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # Probar 20 combinaciones de hiperparámetros

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros:", study.best_params)

# Guardar los mejores hiperparámetros en un archivo JSON
os.makedirs("results", exist_ok=True)  # Crear carpeta para guardar resultados
with open("results/best_params.json", "w") as f:
    json.dump(study.best_params, f)

# Generar gráficos interactivos y guardarlos como HTML
os.makedirs("results/HTML", exist_ok=True)
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html("results/HTML/optimization_history.html")

fig = optuna.visualization.plot_contour(study)
fig.write_html("results/HTML/contour.html")

fig = optuna.visualization.plot_slice(study)
fig.write_html("results/HTML/slice.html")

fig = optuna.visualization.plot_param_importances(study)
fig.write_html("results/HTML/param_importances.html")

print("Gráficos interactivos guardados en la carpeta 'results/HTML'.")
