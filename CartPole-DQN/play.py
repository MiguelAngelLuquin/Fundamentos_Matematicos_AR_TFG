import torch
from environment import CartPoleEnv
from agent import DQNAgent
import time

# Configuración inicial
env = CartPoleEnv(render_mode=True)  # Renderizar entorno para visualizar las partidas
state_dim = env.state_shape[0]
action_dim = env.action_space

# Crear el agente y cargar el modelo entrenado
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
agent.policy_net.load_state_dict(torch.load("dqn_model.pth"))  # Cargar el modelo entrenado
agent.policy_net.eval()  # Poner la red en modo evaluación (no entrenamiento)

# Jugar 2 partidas
num_test_episodes = 2

for episode in range(num_test_episodes):
    state = env.reset()  # Reiniciar el entorno para cada partida
    done = False
    total_reward = 0
    
    print(f"Partida {episode + 1}:")
    
    while not done:
        env.render()  # Renderizar entorno para visualizar la simulación
        
        # Seleccionar acción basada en la política entrenada (sin exploración)
        action = agent.select_action(state, exploit_only=True)
        
        # Ejecutar la acción en el entorno y obtener resultados
        next_state, reward, done = env.step(action)
        
        state = next_state
        total_reward += reward
    
    print(f"Recompensa Total: {total_reward}")
    time.sleep(1)  # Esperar 1 segundo entre partidas

# Cerrar el entorno después de jugar las partidas
env.close()
