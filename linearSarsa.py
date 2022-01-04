from LinearWrapper import LinearWrapper
from chooseAction import choose_action
from frozenlake import FrozenLake
import numpy as np
from chooseAction import choose_action
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)
        print(features.shape)
        # print("----->", q)
        # state = np.where(features == 1)
        # print("state = ", state)
        print(env.state)
    #     state = env.state # change this line to get current state:
    #     action = choose_action(epsilon[i], q, random_state)
    #     done = False
    #     while not done:
    #         next_state, reward, done = env.step(action)
    #         print("next_state", next_state)
    #         next_action = choose_action(epsilon[i], q, next_state)
            
    #         print("next action", next_action)
    #         delta  = reward + gamma*q[next_state, next_action] - q[state, action]
    #         for index in range(len(theta)):
    #             theta[index] = theta[index] + eta[i]*delta*features[state, action]
    #         state = next_state
    #         action = next_action
    # return theta

if __name__ =='__main__':
    seed = 0
    # max_episodes = 2000
    max_episodes = 1
    eta = 0.5
    epsilon = 0.5
    gamma = 0.9
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    linear_env = LinearWrapper(env)
    
    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    
    print('')