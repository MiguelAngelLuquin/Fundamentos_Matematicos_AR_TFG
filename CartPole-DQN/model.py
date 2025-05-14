import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Red neuronal para aproximar Q(s, a).
        
        Args:
            state_dim (int): Dimensión del espacio de estados.
            action_dim (int): Número de acciones posibles.
        """
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)  # Capa oculta 1
        self.fc2 = nn.Linear(128, 64)        # Capa oculta 2
        self.fc3 = nn.Linear(64, action_dim)  # Capa de salida'

    def forward(self, state):
        """
        Propagación hacia adelante para calcular los valores Q.
        
        Args:
            state (torch.Tensor): Estado actual.
        
        Returns:
            torch.Tensor: Valores Q estimados para cada acción.
        """
        x = torch.relu(self.fc1(state))  # Activación ReLU en la primera capa
        x = torch.relu(self.fc2(x))      # Activación ReLU en la segunda capa
        return self.fc3(x)
