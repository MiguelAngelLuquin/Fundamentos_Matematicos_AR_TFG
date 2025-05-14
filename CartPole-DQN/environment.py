import gymnasium as gym

class CartPoleEnv:
    def __init__(self, env_name='CartPole-v1', render_mode=False):
        """
        Inicializa el entorno Gym.
        
        Args:
            render_mode (bool): Si es True, el entorno se renderiza.
        """
        if render_mode:
            self.env = gym.make(env_name, render_mode="human")
        else:
            self.env = gym.make(env_name)
        self.state_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space.n
        self.render_mode = render_mode

    def reset(self):
        """
        Reinicia el entorno y devuelve el estado inicial.
        """
        state, _ = self.env.reset()
        return state

    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action (int): Acción a tomar (0 o 1).
        
        Returns:
            tuple: (estado siguiente, recompensa, done)
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def render(self):
        """
        Renderiza el entorno (solo si render_mode es True).
        """
        if self.render_mode:
            self.env.render()

    def close(self):
        """
        Cierra el entorno.
        """
        self.env.close()
    

    def normalize_state(self, state):
        """
        Normaliza el estado entre 0 y 1.
        """
        return (state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

