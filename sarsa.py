import numpy as np
from frozenlake import FrozenLake
import numpy as np
from chooseAction import choose_action
def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    # eta is the learning rate decay linearly eta[i] is the learning rate for episode i

    epsilon = np.linspace(epsilon, 0, max_episodes)
    # epsilon is decay linearly espilon[i] is the for episode i

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        action = choose_action(epsilon[i], q, s)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = choose_action(epsilon[i], q, next_state)
            q[s, action] += eta[i]* (reward + (gamma * q[next_state, next_action]-q[s, action]))
            s = next_state
            action = next_action


    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
if __name__ == '__main__':
    seed = 0
    # # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    print('## Sarsa')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    gamma = 0.9
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')
