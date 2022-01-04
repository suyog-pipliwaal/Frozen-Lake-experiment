import numpy as np
from LinearWrapper import LinearWrapper
from chooseAction import choose_action
from frozenlake import FrozenLake
# def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
#     random_state = np.random.RandomState(seed)

#     eta = np.linspace(eta, 0, max_episodes)
#     epsilon = np.linspace(epsilon, 0, max_episodes)

#     theta = np.zeros(env.n_features)

#     for i in range(max_episodes):
#         features = env.reset()
#         state = random_state
#         for action in 

#     return theta
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:
        q = features.dot(theta)

        done = False
        while not done:
            #-- Finding the lineare greedy selection
            actions = range(env.n_actions)

            if random_state.rand() < epsilon[i]:
                action = random_state.choice(actions)
            else:
                #-- finding the maximum argument randomly 
                arg = np.argsort(q[actions])[::-1]
                n_tied = sum(np.isclose(q[actions], q[actions][arg[0]]))
                action = np.random.choice(arg[0:n_tied])
                action =  actions[action]
                
            
            next_features, r, done = env.step(action)
            delta = r - q[action]

            q = next_features.dot(theta)
            
            delta += gamma * max(q)
            
            theta += eta[i] * delta * features[action, :]
            
            features = next_features

    return theta

if __name__ =='__main__':
    seed = 0
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    gamma = 0.9
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    linear_env = LinearWrapper(env)

    print('## Q-Learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')
