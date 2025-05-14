import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class FrozenLakeQLearning:
    def __init__(self, grid_size=3,alpha=0.1):
        self.grid_size = grid_size
        self.actions = ['up', 'down', 'left', 'right']
        self.q_table = np.zeros((grid_size, grid_size, len(self.actions)))
        
        # Hiperparámetros dependientes del agente
        self.gamma = 0.95   # Factor de descuento
        self.alpha = alpha   # Tasa de aprendizaje
            
        self.epsilon = 1.0 # Tasa de exploración inicial
        self.epsilon_decay = 0.99
        self.min_epsilon = 0

    def get_reward(self, state):
        x, y = state
        if x == 2 and y == 2:  # Meta
            return 1
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return -0.5  # Fuera del tablero
        return 0    # Paso normal

    def step(self, state, action, test=0):
        x, y = state
        new_state = state
        rand = np.random.uniform(0, 1)
        action2 = action
        while(rand < 0.1 and action2 == action and test == 0):
            action2 = np.random.choice(self.actions)
        if action2 == 'up':
            new_state = (x, y+1)
        elif action2 == 'down':
            new_state = (x, y-1)
        elif action2 == 'left':
            new_state = (x-1, y)
        elif action2 == 'right':
            new_state = (x+1, y)
            
        reward = self.get_reward(new_state)
        done = (reward == 1) or (reward == -0.5)
        
        # Mantener estado dentro de límites
        if reward == -0.5:
            new_state = state  # No se mueve
            
        return new_state, reward, done

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions)
        else:
            x, y = state
            return self.actions[np.argmax(self.q_table[x, y])]

    def train(self, episodes=100):
        for _ in range(episodes):
            state = (0, 0)
            done = False
            
            while not done:
                action = self.choose_action(state, self.epsilon)
                action_idx = self.actions.index(action)
                
                next_state, reward, done = self.step(state, action)
                
                # Actualizar Q-table
                x, y = state
                next_x, next_y = next_state
                current_q = self.q_table[x, y, action_idx]
                
                if done:
                    max_next_q = 0
                else:
                    max_next_q = np.max(self.q_table[next_x, next_y])
                
                new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
                self.q_table[x, y, action_idx] = new_q
                
                state = next_state
                
            # Decaimiento de epsilon
            self.epsilon = self.epsilon * self.epsilon_decay

    def get_optimal_path(self):
        path = [(0, 0)]
        state = (0, 0)
        count = 0
        while state != (2, 2) and count < self.grid_size**2:
            x, y = state
            action_idx = np.argmax(self.q_table[x, y])
            action = self.actions[action_idx]
            
            next_state, _, _ = self.step(state, action, 1)
            path.append(next_state)
            state = next_state
            
            if len(path) > self.grid_size**2:  # Prevenir loops
                break
            count += 1
        if(path[2] == (1,1) and path[4] == (2,2)):
            return 1
        return 0
    
    def print_qtable_as_table(self):
        # Crear índices para las 9 casillas
        indices = [f"({i},{j})" for i in range(self.grid_size) for j in range(self.grid_size)]
        
        # Crear DataFrame
        df = pd.DataFrame(columns=self.actions, index=indices)
        
        # Llenar la tabla con valores Q
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for a_idx, action in enumerate(self.actions):
                    df.loc[f"({i},{j})", action] = f"{self.q_table[i,j,a_idx]:.2f}"
        
        print("\nQ-Table (valor por estado-acción):")
        print(df)



if __name__ == "__main__":
    entrenamientos = 100
    episodios = list(range(10, 201, 10))
    resultados = []
    alphas = np.linspace(0.01, 1, 100)  # Valores de alpha para cada agente
    for j in range(1,101):
        print(f"Entrenando agente {j}...")
        for k in episodios:
            count = 0
            for i in range(entrenamientos):
                lake = FrozenLakeQLearning(alpha=alphas[j-1])
                lake.train(episodes=k)
                count += lake.get_optimal_path()
            porcentaje = 100 * count / entrenamientos
            if porcentaje >= 70:
                resultados.append(k)
                break
            if k == episodios[-1]:
                resultados.append(k)
    
    print("Entrenamiento completo.")

    # Ahora graficamos
    plt.figure(figsize=(10, 6))
    plt.plot(alphas,resultados, marker='o')

    plt.title("Convergencia según factor de aprendizaje")
    plt.xlabel("Factor de aprendizaje (alpha)")
    plt.ylabel("Episodios para convergencia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
