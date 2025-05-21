#################################################################################
# Autor: Miguel Ángel Luquín Guerrero
# Fecha de creación: 23/01/2025
# Última modificación: 19/05/2025
# Versión: 3.0
# Descripción: Programa que entrena a la máquina a jugar al 3 en raya
#              usando el algoritmo Q-Learning contra sí mismo. En esta 
#              versión se ha añadido la posibilidad de guardar la tabla 
#              Q en un .csv para poder usarla en el programa "play_tictactoe.py"
#################################################################################

import numpy as np
import random
import csv

# Parámetros del entorno
ROWS = 3
COLS = 3
STATES = 3**(ROWS*COLS)
EPISODES = 100000 #100 000 iteraciones
INI = 0

# Crear el tablero inicial
def tablero_inicial():
    tab = np.zeros((ROWS, COLS))
    return tab

# Obtener la siguiente jugada dados el estado y la acción
def next_tab(tab, action, player):
    tab[((action-1)//COLS)][(action-1)%COLS] = player
    return tab

# Definimos el siguiente estado
def next_states(state, action, player):
    state += player*3**(9-action)
    return state

#Comprobamos si alguien ha ganado o si la partida ha terminado en empate
def checkresult(tab):
    ganador = 0
    for i in range(3):
        if tab[i][0] == tab[i][1] == tab[i][2] != 0:
            ganador = tab[i][0]
        if tab[0][i] == tab[1][i] == tab[2][i] != 0:
            ganador = tab[0][i]
    if tab[0][0] == tab[1][1] == tab[2][2] != 0:
        ganador = tab[0][0]
    if tab[0][2] == tab[1][1] == tab[2][0] != 0:
        ganador = tab[0][2]
    return ganador

# Algoritmo Q-Learning para entrenar a la máquina a jugar en ambos roles
def q_learning(episodes=EPISODES, alpha=0.1, gamma=0.95):
    q_table = np.zeros((STATES, ROWS*COLS))  # Inicializar tabla Q

    #Contadores de victorias, empates y derrotas (con respecto del jugador 1)
    wins = 0
    draws = 0
    losses = 0
    alpha_ini = alpha
    epsilon_decay = 0.9999
    epsilon = 1.0
    #numero de episodios
    for episode in range(episodes):
        if episode%5000 == 0:
            print(episode*100/episodes, "%")

        #inicializamos el estado, el tablero y el turno
        state = INI
        tablero = tablero_inicial()
        turn = 0
        reward = 0

        # Factor de exploración que va decreciendo con los episodios
        epsilon *= epsilon_decay
        alpha = alpha_ini * (1 - episode / episodes)

        # En el principio de cada episodio, el agente comienza con todas las posibilidades de acción
        ACTIONS = [1,2,3,4,5,6,7,8,9]

        # Esto representa una jugada de cada uno
        while True:
            
            turn += 1
    
            ##########Jugador 1##########

            # Decidir la acción (explorar/explotar)
            if random.uniform(0, 1) < epsilon:
                action = random.choice(ACTIONS)
            else:
                maxim = q_table[state][ACTIONS[0]-1]
                action = ACTIONS[0]
                for ac in range(len(ACTIONS)-1):
                    if q_table[state][ACTIONS[ac+1]-1] > maxim:
                        maxim = q_table[state][ACTIONS[ac+1]-1]
                        action = ACTIONS[ac+1]

            # Actualizar el estado, el tablero y quitamos la acción de las posibles
            next_state = next_states(state, action, 1)
            tablero = next_tab(tablero, action, 1)
            ACTIONS.remove(action)

            # Comprobamos si hay ganador
            reward = checkresult(tablero) 

            # Actualizar la tabla Q de la jugada anterior
            if len(ACTIONS) == 0:
                best_next_action = 0
            else:
                valid_q_values = [q_table[next_state][a-1] for a in ACTIONS]
                best_next_action = np.min(valid_q_values)
            q_table[state][action-1] += alpha * (
                reward + gamma * best_next_action - q_table[state][action-1]
            )

            if reward > 0:
                wins += 1
                break
            elif turn == 9:
                draws += 1
                break

            # Fin turno Jugador 1
            state = next_state
            turn += 1

            ##########Jugador 2##########

            # Decidir la acción (explorar/explotar)
            if random.uniform(0, 1) < epsilon:
                action = random.choice(ACTIONS)
            else:
                minim = q_table[state][ACTIONS[0]-1]
                action = ACTIONS[0]
                for ac in range(len(ACTIONS)-1):
                    if q_table[state][ACTIONS[ac+1]-1] < minim:
                        minim = q_table[state][ACTIONS[ac+1]-1]
                        action = ACTIONS[ac+1]

            # Actualizar el estado, el tablero y quitamos la acción de las posibles
            next_state = next_states(state, action, 2)
            tablero = next_tab(tablero, action, -1)
            ACTIONS.remove(action) # Quitamos la casilla ya ocupada

            # Comprobamos si hay ganador
            reward = checkresult(tablero) # Comprobamos si hay ganador
            
            # Actualizar la tabla Q de la jugada anterior
            valid_q_values = [q_table[next_state][a-1] for a in ACTIONS]
            best_next_action = np.max(valid_q_values)
            q_table[state][action-1] += alpha * (
                reward + gamma * best_next_action - q_table[state][action-1]
            )

            if reward < 0:
                losses += 1
                break

            state = next_state

    return q_table, wins, draws, losses

# Función que muestra el ratio de victorias de Jug 1 y 2
def resultados(wins, draws, loses):
    print("Player 1: ", wins*100/EPISODES, "%")
    print("Draws: ", draws*100/EPISODES, "%")
    print("Player 2: ", loses*100/EPISODES, "%")   

#Función que muestra cuántos estados ha enfrentado la máquina
def mirar_estados(tabla):
    count = 0
    for state in range(STATES):
        if not all(x == 0 for x in tabla[state]):
            count += 1
    print("Se ha enfrentado la maquina a ", count, " estados.")

#Función que guarda la tabla Q en un .csv
def guardar_tabla(tabla):
    with open('tictactoe.csv', mode='w', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerows(tabla)

#---INICIO DE PROGRAMA PRINCIPAL---#

#Ejecución del algoritmo Q-Learning
q_table, wins, draws, loses = q_learning()

#Mostrar resultados
resultados(wins, draws, loses)
mirar_estados(q_table)
#Guardar tabla Q en un .csv
guardar_tabla(q_table)
