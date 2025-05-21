#################################################################################
# Autor: Miguel Ángel Luquín Guerrero
# Fecha de creación: 23/01/2025
# Última modificación: 19/05/2025
# Versión: 1.0
# Descripción: Programa que permite jugar al 3 en raya contra la máquina
#              que ha sido entrenada con el algoritmo Q-Learning. Así como probar a
#              la máquina contra un algoritmo aleatorio.
#              Para ello, se debe cargar la tabla Q desde el archivo tictactoe.csv
#              generado por el programa "tictactoe_train.py"
#################################################################################

import csv
import numpy as np
import random

# Parámetros del entorno
ROWS = 3
COLS = 3

####Funciones auxiliares####

# Leer la matriz desde el archivo CSV
def cargar_tabla():
    with open('tictactoe.csv', mode='r') as archivo:
        lector = csv.reader(archivo)
        q_table = [list(map(float, fila)) for fila in lector]
    return q_table


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
    if(player == -1):
        player = 2
    state += player*3**(9-action)
    return state

# Comprobamos si alguien ha ganado
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

# Funcion para cambiar de base n a base m
def convertir_base(num, base_n, base_m):
    num_decimal = 0
    potencia = 0
    while num > 0:
        digito = num % 10
        num_decimal += digito * (base_n ** potencia)
        num //= 10
        potencia += 1
    if num_decimal == 0:
        return 0
    resultado = 0
    multiplicador = 1
    while num_decimal > 0:
        resultado += (num_decimal % base_m) * multiplicador
        num_decimal //= base_m
        multiplicador *= 10
    return resultado
############################

# Función para imprimir el tablero
def imprimir_rejilla(numero):
    numero = convertir_base(numero,10,3)
    numero_str = str(numero).zfill(9)
    casillas = [int(digito) for digito in numero_str]

    for i in range(0, 9, 3):
        fila = casillas[i:i+3] 
        fila_mostrada = [' ' if x == 0 else 'X' if x == 1 else 'O' for x in fila]
        print(" | ".join(fila_mostrada))
        if i < 6:
            print("---------")
    print(" ")

# Función para probar el algoritmo Q-Learning contra uno aleatorio  
# jugando aleatoriamente como jugador 1 o 2
def jugar_contra_random(q_table):
    
    # Contadores de victorias
    win1 = 0
    win2 = 0
    GAMES = 1000000

    print("Iniciando juegos: 0.0%")
    for i in range(GAMES):
        if (i+1)%10000 == 0:
            print("Jugado un", (i+1)*100/GAMES, "%")

        # Elejigmos el turno de la máquina a mano
        jugada = np.random.choice([-1, 1])

        # Inicializamos el estado, el tablero y el turno
        ACTIONS = [1,2,3,4,5,6,7,8,9]
        tablero = tablero_inicial()
        turn = 0
        state = 0

        # Esto representa una jugada de cada uno
        while True: 
            
            turn += 1

            #Jugada del algoritmo Q-Learning cuando le toque
            if (turn%2 == 1 and jugada == 1):
                maxim = q_table[state][ACTIONS[0]-1]
                action = ACTIONS[0]
                for ac in range(len(ACTIONS)-1):
                    if q_table[state][ACTIONS[ac+1]-1] > maxim:
                        maxim = q_table[state][ACTIONS[ac+1]-1]
                        action = ACTIONS[ac+1]
                next_state = next_states(state, action, jugada)
                tablero = next_tab(tablero, action, jugada)
            elif (turn%2 == 0 and jugada == -1):
                minim = q_table[state][ACTIONS[0]-1]
                action = ACTIONS[0]
                for ac in range(len(ACTIONS)-1):
                    if q_table[state][ACTIONS[ac+1]-1] < minim:
                        minim = q_table[state][ACTIONS[ac+1]-1]
                        action = ACTIONS[ac+1]
                next_state = next_states(state, action, jugada)
                tablero = next_tab(tablero, action, jugada)

            #Jugada del algoritmo aleatorio
            else:
                action = random.choice(ACTIONS)
                next_state = next_states(state, action, -jugada)
                tablero = next_tab(tablero, action, -jugada)

            # Quitamos la casilla ya ocupada
            ACTIONS.remove(action) 

            # Comprobamos si hay ganador y gestionamos el marcador
            fin = checkresult(tablero) 
            if (fin > 0 and jugada == 1) or (fin < 0 and jugada == -1):
                win1 += 1
                break
            elif (fin < 0 and jugada == 1) or (fin > 0 and jugada == -1):
                win2 += 1
                break
            elif turn == 9:
                break

            state = next_state

    # Han terminado las partidas, mostramos los resultados
    print("Victorias QLearning: ", win1)
    print("Empates: ", GAMES-win1-win2)
    print("Victorias Aleatorio: ", win2)

# Función para jugar contra la máquina a través de la consola
# El usuario elige ser el jugador 1 o 2
def jugar_contra_maquina(q_table):
    # Inicializamos el tablero y el estado
    ACTIONS = [1,2,3,4,5,6,7,8,9]
    tablero = tablero_inicial()
    state = 0
    turn = 0

    # El usuario elige ser el jugador 1 o 2
    jugada = "0"
    while jugada != "1" and jugada != "2":
        jugada = input("¿Quieres ser el jugador 1 o el jugador 2? (1/2): ")
    #Bucle de juego
    while True:
        imprimir_rejilla(state)

        turn += 1

        if (turn%2 == 1 and jugada == "2"):
            print("Turno de la máquina")
            maxim = q_table[state][ACTIONS[0]-1]
            action = ACTIONS[0]
            for ac in range(len(ACTIONS)-1):
                if q_table[state][ACTIONS[ac+1]-1] > maxim:
                    maxim = q_table[state][ACTIONS[ac+1]-1]
                    action = ACTIONS[ac+1]
            next_state = next_states(state, action, 2)
            tablero = next_tab(tablero, action, -1)

        elif(turn%2 == 0 and jugada == "1"):
            print("Turno de la máquina")
            minim = q_table[state][ACTIONS[0]-1]
            action = ACTIONS[0]
            for ac in range(len(ACTIONS)-1):
                if q_table[state][ACTIONS[ac+1]-1] < minim:
                    minim = q_table[state][ACTIONS[ac+1]-1]
                    action = ACTIONS[ac+1]
            next_state = next_states(state, action, 2)
            tablero = next_tab(tablero, action, -1)

        else:
            print("Introduce la casilla en la que quieres jugar: (1 = arriba izqda, 2 arriba centro,... , 9 abajo derecha)")
            casilla = int(input("--> "))
            action = int(casilla)
            while action not in ACTIONS:
                print("Casilla incorrecta u ocupada")
                print("Introduce la casilla en la que quieres jugar: (1 = arriba izqda, 2 arriba centro,... , 9 abajo derecha)")
                casilla = int(input("--> "))
                action = int(casilla)
                continue
            next_state = next_states(state, action, 1)
            tablero = next_tab(tablero, action, 1)
        
        # Quitamos la casilla ya ocupada y actualizamos el estado
        ACTIONS.remove(action)
        state = next_state

        # Comprobamos si hay ganador y gestionamos el marcador
        fin = checkresult(tablero) 
        if fin > 0:
            imprimir_rejilla(state)
            print("¡Has ganado!")
            break
        elif fin < 0:
            imprimir_rejilla(state)
            print("¡Has perdido!")
            break
        elif turn == 9:
            imprimir_rejilla(state)
            print("¡Empate!")
            break

#Función principal
q_table = cargar_tabla()
terminate = False
while not terminate:
    print("¿Qué desea hacer? (Introduce el número de la opción)")
    print("1. Jugar contra la máquina")
    print("2. Probar la máquina contra un algoritmo aleatorio")
    print("3. Salir")
    opcion = input("--> ")
    if opcion == "1":
        jugar_contra_maquina(q_table)
    elif opcion == "2":
        jugar_contra_random(q_table)
    elif opcion == "3":
        print("Saliendo del programa...")
        terminate = True
