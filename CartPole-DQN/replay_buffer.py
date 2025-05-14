import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Inicializa el buffer de repetición.
        
        Args:
            capacity (int): Capacidad máxima del buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        """
        Añade una experiencia al buffer.
        
        Args:
            experience (tuple): (estado, acción, recompensa, estado_siguiente, done).
                - estado: numpy.ndarray o lista representando el estado actual.
                - acción: int representando la acción tomada.
                - recompensa: float representando la recompensa obtenida.
                - estado_siguiente: numpy.ndarray o lista representando el siguiente estado.
                - done: bool indicando si el episodio terminó.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Muestra un lote aleatorio de experiencias.
        
        Args:
            batch_size (int): Tamaño del lote a muestrear.
        
        Returns:
            list: Lote de experiencias seleccionadas aleatoriamente.
                  Cada elemento es un tuple (estado, acción, recompensa, estado_siguiente, done).
        """
        return random.sample(self.buffer, batch_size)

    def size(self):
        """
        Devuelve el tamaño actual del buffer.
        
        Returns:
            int: Número de elementos actualmente almacenados en el buffer.
        """
        return len(self.buffer)
