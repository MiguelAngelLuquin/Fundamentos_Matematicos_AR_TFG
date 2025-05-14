from environment import CartPoleEnv
from agent import DQNAgent
import os
import numpy as np
import sys

# Configuración inicial
env = CartPoleEnv()  # No renderizar en entrenamiento
agent = DQNAgent(state_dim=env.state_shape[0], action_dim=env.action_space)

num_episodes = 300

threshold = 350  # Umbral para resolver el entorno

#Para guardar puntos intermedios
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
reward_history = []
max_avg_reward = float('-inf')

# Entrenamiento
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    
    # Actualizar la red objetivo cada ciertos episodios
    if episode % agent.target_update_freq == 0:
        agent.update_target_net()
    
    reward_history.append(total_reward)
    current_avg_reward = np.mean(reward_history[-100:])
    
    max_avg_reward = max(max_avg_reward, current_avg_reward)
    
    # Imprimir y borrar la línea anterior
    sys.stdout.write(f"\rEpisodio {episode}, Recompensa Total: {total_reward:.2f}, Recompensa Media Actual: {current_avg_reward:.2f}, Recompensa Media Máxima: {max_avg_reward:.2f}")
    sys.stdout.flush()

    # Dentro del bucle de episodios:
    if episode != 0 and episode % 50 == 0:  # Guardar cada 100 episodios
        agent.save_model(f"{CHECKPOINT_DIR}/model_ep{episode}.pth")
        np.save(f"{CHECKPOINT_DIR}/rewards_ep{episode}.npy", reward_history)
    
    if current_avg_reward >= threshold:
        print(f"\n\nEntorno resuelto en {episode} episodios")
        break
    
# Después del bucle de entrenamiento
print("\nEntrenamiento completado")
agent.save_model("dqn_model.pth")
print("Modelo guardado exitosamente")

env.close()
