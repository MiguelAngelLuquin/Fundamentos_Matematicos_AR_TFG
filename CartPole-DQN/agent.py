import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from replay_buffer import ReplayBuffer
from model import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_capacity=2000,
                 gamma=0.99, lr=25e-4, batch_size=64, target_update_freq=1):
        """
        Inicializa el agente DQN.

        Args:
            state_dim (int): Dimensión del espacio de estados.
            action_dim (int): Número de acciones posibles.
            replay_buffer_capacity (int): Capacidad máxima del Replay Buffer.
            gamma (float): Factor de descuento.
            lr (float): Tasa de aprendizaje.
            batch_size (int): Tamaño del mini-batch para entrenamiento.
            target_update_freq (int): Frecuencia para actualizar la red objetivo.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Redes neuronales, son 2, una para la política y otra para el objetivo
        # la red objetivo se actualiza cada "target_update_freq" pasos mientras que la política 
        # se actualiza en cada paso
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

        # Parámetros para epsilon-greedy
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 500  # Decaimiento exponencial
        
        self.steps_done = 0

    def select_action(self, state, exploit_only=False):
        """
        Selecciona una acción usando epsilon-greedy o solo explotación.

        Args:
            state (numpy.ndarray): Estado actual.
            exploit_only (bool): Si es True, elige siempre la mejor acción sin exploración.

        Returns:
            int: Acción seleccionada.
        """
        if not exploit_only:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.exp(-1. * self.steps_done / self.epsilon_decay)

            self.steps_done += 1

            if random.random() < epsilon:
                return random.randrange(self.action_dim)  # Acción aleatoria

        # Si exploit_only=True o pasa la condición epsilon-greedy, usa la red neuronal
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor) # Esta función invoca por detras a la función forward de la clase DQN
            return q_values.argmax().item()  # Selecciona la mejor acción


    def store_experience(self, state, action, reward, next_state, done):
        """
        Almacena una transición en el Replay Buffer.

        Args:
            state (numpy.ndarray): Estado actual.
            action (int): Acción tomada.
            reward (float): Recompensa recibida.
            next_state (numpy.ndarray): Estado siguiente.
            done (bool): Indicador si el episodio terminó.
        """
        self.replay_buffer.add((state, action, reward, next_state, done))

    def train(self):
        """
        Entrena la red neuronal utilizando un mini-batch del Replay Buffer.
        """
        if self.replay_buffer.size() < self.batch_size:
            return  # No hay suficientes muestras para entrenar

        # Muestreo aleatorio del Replay Buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir listas de numpy.ndarrays a numpy.ndarray antes de crear tensores para mejorar rendimiento
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Calcular los valores Q actuales
        current_q_values = self.policy_net(states).gather(1, actions) # Esta función invoca por detras a la función forward de la clase DQN

        # Calcular los valores Q objetivos usando la red objetivo
        with torch.no_grad(): # No se calculan gradientes para mejorar rendimiento ya que no se usan
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))  # Función de Bellman

        # Pérdida Huber entre valores actuales y objetivos
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Clipping para estabilidad numérica
        self.optimizer.step()

    def update_target_net(self):
        """
        Copia los pesos de la red principal a la red objetivo.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath):
        """Guarda los parámetros del modelo entrenado"""
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Carga un modelo pre-entrenado"""
        self.policy_net.load_state_dict(torch.load(filepath))