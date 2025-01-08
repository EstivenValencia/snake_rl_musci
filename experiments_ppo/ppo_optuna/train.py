import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from snakeenv import SnakeEnv
import os

# Definir la función objetivo para la optimización
def objective(trial):
    # Espacio de búsqueda para los hiperparámetros
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    n_epochs = trial.suggest_int("n_epochs", 1, 20, step=2)
    gamma = trial.suggest_float("gamma", 0.8, 1.0, step=0.05)

    # Crear el entorno
    env = SnakeEnv(render=None)

    # Configurar el modelo con los hiperparámetros actuales
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=n_epochs,
        verbose=0,
        device='cpu'
    )

    # Entrenar por 50.000 pasos como prueba
    model.learn(total_timesteps=30000)

    # Evaluar el modelo
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    return mean_reward 

# Crear el estudio de Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)  # Probar 20 combinaciones de hiperparámetros

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros:", study.best_params)

# Crear la carpeta HTML si no existe
os.makedirs("HTML", exist_ok=True)

# Generate the improtant figures of the results
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html(f"HTML/optimization_history.html")
fig = optuna.visualization.plot_contour(study)
fig.write_html(f"HTML/contour.html")
fig = optuna.visualization.plot_slice(study)
fig.write_html(f"HTML/slice.html")
fig = optuna.visualization.plot_param_importances(study)
fig.write_html(f"HTML/param_importances.html")