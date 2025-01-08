# Experimentos 

El siguiente repositorio presenta la documentación de los experimentos llevados acabo para el desarrollo y entrenamiento del agente de Snake. Se presentan dos carpetas experiments_dqn y experiments_ppo.

# Experiments_dqn

Esta carpeta contiene los experimentos realizados utilizando el algoritmo DQN.

### Estructura

- `prueba_1DQN.py`: Archivo principal que ejecuta el primer experimento, representando un entrenamiento simple e indagatorio.

### Carpeta `dqn`

Contiene los siguientes archivos y carpetas:

- `dqnoptuna.py`: Experimento Optuna con 50,000 pasos.
- `dqnoptuna2.py`: Experimento Optuna con 100,000 pasos.
- `train_largo.py`: Archivo para el entrenamiento de larga duración.
- Carpetas de logs y archivos de evaluación correspondientes.

### Carpeta `politica2`

Contiene:

- `snakeenv.py`: Implementación del entorno de Snake con la política modificada.
- `optuna2.py`: Archivo para el experimento Optuna con la nueva política.

## Descripción de los Experimentos

1. **Experimento Inicial**: Representado por `prueba_1DQN.py`, este fue el primer acercamiento al entrenamiento con DQN.

2. **Optimización con Optuna**: Se realizaron dos experimentos de optimización de hiperparámetros:
   - Con 50,000 pasos (`dqnoptuna.py`)
   - Con 100,000 pasos (`dqnoptuna2.py`)

3. **Entrenamiento de Larga Duración**: Implementado en `train_largo.py`, este experimento se enfocó en un entrenamiento más extenso.

4. **Política Modificada**: En la carpeta `politica2`, se experimentó con una versión modificada del entorno de Snake y se realizó un nuevo experimento Optuna.

## Ejecución de los Experimentos

Para ejecutar cualquiera de los experimentos, navega a la carpeta correspondiente y ejecuta el script Python. Por ejemplo:

```bash
python prueba_1DQN.py
```
o
```bash
python dqn/dqnoptuna.py
```

## Resultados y Logs

Los resultados de los experimentos y los logs se pueden encontrar en las carpetas correspondientes dentro de `dqn/`.

# Experiments_ppo

Esta carpeta contiene los experimentos realizados utilizando el algoritmo PPO.

### Estructura

- `ppo_basico`: Carpeta que contiene el experimento básico de PPO.
- `ppo_optuna`: Carpeta que contiene el experimento de optimización de hiperparámetros con Optuna.
- `ppo_mejor_modelo`: Carpeta con el entrenamiento de los mejores hiperparámetros de optuna con 1600000 pasos.
- `cambio_ambiente`: Carpeta que contiene el experimento con una función de recompensa modificada.

### Carpeta `ppo_basico`

Contiene los siguientes archivos:

- `train.py`: Archivo principal para el entrenamiento básico de PPO y su evaluación.
- `snakeenv.py`: Implementación del entorno de Snake.

### Carpeta `ppo_optuna`

Contiene:

- `train.py`: Archivo para el experimento de optimización de hiperparámetros con Optuna.
- `snakeenv.py`: Implementación del entorno de Snake.

### Carpeta `ppo_mejor_modelo`

Contiene:

- `train.py`: Archivo para el entrenamiento con los mejores hiperparámetros.
- `snakeenv.py`: Implementación del entorno de Snake.

### Carpeta `cambio_ambiente`

Contiene:

- `train.py`: Archivo para el entrenamiento con una función de recompensa modificada.
- `snakeenv.py`: Implementación del entorno de Snake con la recompensa modificada.

En cada una de las carpetas de los experimentos de este apartado se encuentran los modelos entrenados, los checkpoints para el entrenamiento con mejores 
hiperparametros, así como los logs recpectivos con los que se obtuvieron los análisis de tensorboard.
