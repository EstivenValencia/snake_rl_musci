from stable_baselines3 import DQN
from snakeenv import SnakeEnv
import time
import cv2  # Importar OpenCV para usar imshow


# Crear el entorno
env = SnakeEnv()

# Cargar un checkpoint específico
checkpoint_path = "./checkpoints_final/dqn_snake_optimized_110000_steps.zip"
model = DQN.load(checkpoint_path, env=env)


# Evaluar el modelo en tiempo real
obs, info = env.reset()
for _ in range(1000):  # Número de pasos a visualizar
    action, _ = model.predict(obs, deterministic=True)  # Acción del modelo
    obs, reward, done, truncated, info = env.step(action)
    
    # Visualización con OpenCV (renderizado personalizado)
    cv2.imshow("SnakeEnv", env.img)  # `env.img` es la imagen que se genera en SnakeEnv
    cv2.waitKey(50)  # Controlar la velocidad de visualización (50 ms)

    if done or truncated:
        obs, info = env.reset()  # Reinicia el entorno si termina

cv2.destroyAllWindows()  # Cierra todas las ventanas al finalizar