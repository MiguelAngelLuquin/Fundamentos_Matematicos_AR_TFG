import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class FrozenLakeQLearning:
    def __init__(self, grid_size=3, agent=1):
        self.grid_size = grid_size
        self.actions = ['up', 'down', 'left', 'right']
        self.q_table = np.zeros((grid_size, grid_size, len(self.actions)))
        
        # Hiperparámetros dependientes del agente
        self.agent = agent
        if agent == 1:
            self.gamma = 0.25   # Factor de descuento
            self.alpha = 0.9   # Tasa de aprendizaje
        elif agent == 2:
            self.gamma = 0.95   # Factor de descuento
            self.alpha = 0.1   # Tasa de aprendizaje
        elif agent == 3:
            self.gamma = 0.25   # Factor de descuento
            self.alpha = 0.1   # Tasa de aprendizaje
        else:
            self.gamma = 0.95   # Factor de descuento
            self.alpha = 0.1   # Tasa de aprendizaje
            
        self.epsilon = 1.0 # Tasa de exploración inicial
        self.epsilon_decay = 0.999
        self.alphastart = self.alpha

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
        for i in range(episodes):
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

            # Si usa decaimineto de factor de aprendizaje:
            if(self.agent == 3 or self.agent == 4):
                self.alpha = self.alphastart * (1 - i / episodes)

    def get_optimal_path(self):
        if (np.argmax(self.q_table[0, 0]) == 0 or np.argmax(self.q_table[0, 0]) == 3):
            if (np.argmax(self.q_table[0, 1]) == 3):
                if (np.argmax(self.q_table[1, 0]) == 0):    
                    if (np.argmax(self.q_table[0, 2]) == 3):
                        if (np.argmax(self.q_table[1, 1]) == 0 or np.argmax(self.q_table[1, 1]) == 3):
                            if (np.argmax(self.q_table[2, 0]) == 0):
                                if (np.argmax(self.q_table[1, 2]) == 3):
                                    if (np.argmax(self.q_table[2, 1]) == 0):
                                        return 1 #Se encontró una política óptima para todo estado
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
    episodios = list(range(100, 1001, 100))

    resultados = {j: [] for j in range(1, 5)}  # Porcentajes de éxito
    errores = {j: [] for j in range(1, 5)}     # Intervalos de error (para errorbars)

    z = 1.96  # Para un 95% de confianza

    for j in range(1, 5):
        print(f"\nEntrenando agente {j}...")
        for k in episodios:
            count = 0
            for i in range(entrenamientos):
                lake = FrozenLakeQLearning(agent=j)
                lake.train(episodes=k)
                count += lake.get_optimal_path()
            p = count / entrenamientos
            porcentaje = 100 * p
            resultados[j].append(porcentaje)

            # Cálculo del intervalo de confianza de Wald
            error = z * 100 * np.sqrt(p * (1 - p) / entrenamientos)
            errores[j].append(error)

            print(f"Agente {j} con {k} episodios: {porcentaje:.2f}% ± {error:.2f}%")

    print("Entrenamiento completo.")
    # Colores y estilos consistentes
    colors = ['blue', 'orange', 'green', 'red']
    labels = [f"Agente {j}" for j in range(1, 5)]

    plt.figure(figsize=(10, 6))

    for idx, j in enumerate(range(1, 5)):
        y = resultados[j]
        err = errores[j]
        y = np.array(y)
        err = np.array(err)

        # Línea principal
        plt.plot(episodios, y, label=labels[idx], color=colors[idx], marker='o')

        # Intervalo de confianza como superficie
        plt.fill_between(episodios, y - err, y + err, color=colors[idx], alpha=0.2)

    plt.title("Porcentaje de éxito vs Número de episodios")
    plt.xlabel("Número de episodios")
    plt.ylabel("Porcentaje de éxito (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
