import numpy as np
def choose_action(epsilon, q, state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(1, 5)
    else:
        action = np.argmax(q[state,:])
    return action