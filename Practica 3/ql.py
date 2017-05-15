import random
import numpy as np
import matplotlib.pyplot as plt


# Environment size
width = 5
height = 16

# Actions
num_actions = 4

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }


actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

# Discount factor
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension


def getState(y, x):                 # De coordenada a estado
    return y * width + x


def getStateCoord(state):          #De estado a coordenada
    return int(state / width), int(state % width)


def getActions(state):             # Devuelve acciones posibles dado un estado
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions


def getRndAction(state):            # Dado un estado coge las acciones posibles y elige una aleatoria
    return random.choice(getActions(state))


def getRndState():
    return random.randint(0, height * width - 1)


Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))


def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return

# Tareas
#   -Calcular promedio acciones antes de llegar al objetivo
#   -Modificar politica (greedy & e-greedy) para alternar entre exploracion y explotacion
#   -Calcular promedio acciones con nuevas politicas y comparar entre ellas

# greedy: si accion de Q en estado actual es >0 coger esa accion, sino accion random
# e-greedy: si accion de Q en estado actual es >0 coger esa accion, sino accion random. Salvo en 1 de cada
# % de veces (var epsilon) cogeremos accion random para obligar a explorar

def getGreedy(state):
    max_action = np.argmax(Q[state])
    return getRndAction(state) if Q[state][max_action] <= 0 else actions_list.keys()[actions_list.values().index(max_action)]       #Invierte action_list para devolver accion en lugar de valor

def getEGreedy(state, epsilon):
    rand = random.random()
    return getGreedy(state) if rand >= epsilon else getRndAction(state)

# Episodes

episodes = 100
total_actions_random = 0     # Variable numero acciones, incrementar cada nueva accion

#Random
for i in xrange(episodes):
    state = getRndState()       # Estado aleatorio
    while state != final_state:
        action = getRndAction(state)        # Accion aleatoria de ese estado
        total_actions_random += 1      # Atualizamos acciones
        y = getStateCoord(state)[0] + actions_vectors[action][0]    # Nueva coordenada Y del nuevo estado
        x = getStateCoord(state)[1] + actions_vectors[action][1]    # Nueva coordenada X del nuevo estado
        new_state = getState(y, x)      # Obtengo nuevo estado
        qlearning(state, actions_list[action], new_state)       # Actualizamos tabla
        state = new_state       # Actualizamos estado


#print Q
print "Acciones totales (Random)"
print total_actions_random
print "Promedio de acciones totales (Random)"
print float(total_actions_random)/float(episodes)


# Greedy

Q = np.zeros((height * width, num_actions))  # Limpieza tabla Q
total_actions_greedy = 0     # Variable numero acciones, incrementar cada nueva accion

for i in xrange(episodes):
    state = getRndState()       # Estado aleatorio
    while state != final_state:
        action = getGreedy(state)        # Accion aleatoria de ese estado
        total_actions_greedy += 1      # Atualizamos acciones
        y = getStateCoord(state)[0] + actions_vectors[action][0]    # Nueva coordenada Y del nuevo estado
        x = getStateCoord(state)[1] + actions_vectors[action][1]    # Nueva coordenada X del nuevo estado
        new_state = getState(y, x)      # Obtengo nuevo estado
        qlearning(state, actions_list[action], new_state)       # Actualizamos tabla
        state = new_state       # Actualizamos estado

#print Q
print "#################################################################"
print "Acciones totales (Greedy)"
print total_actions_greedy
print "Promedio de acciones totales (Greedy)"
print float(total_actions_greedy) / float(episodes)

# E-Greedy

Q = np.zeros((height * width, num_actions))  # Limpieza tabla Q
total_actions_egreedy = 0     # Variable numero acciones, incrementar cada nueva accion
epsilon = 0.9
for i in xrange(episodes):
    state = getRndState()       # Estado aleatorio
    while state != final_state:
        action = getEGreedy(state, epsilon)        # Accion aleatoria de ese estado
        total_actions_egreedy += 1      # Atualizamos acciones
        y = getStateCoord(state)[0] + actions_vectors[action][0]    # Nueva coordenada Y del nuevo estado
        x = getStateCoord(state)[1] + actions_vectors[action][1]    # Nueva coordenada X del nuevo estado
        new_state = getState(y, x)      # Obtengo nuevo estado
        qlearning(state, actions_list[action], new_state)       # Actualizamos tabla
        state = new_state       # Actualizamos estado

#print Q
print "#################################################################"
print "Acciones totales (E-Greedy)"
print total_actions_egreedy
print "Promedio de acciones totales (E-Greedy)"
print float(total_actions_egreedy) / float(episodes)

# Q matrix plot

s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

plt.show()
