import numpy as np

def policy_iteration(env, gamma, theta, max_iteration, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int) #Initialize the policy with zeros
    else:
        policy = np.array(policy, dtype=int) 

    n_policy_iteration = 0

    previous_policy = np.zeros(env.n_states)

    while max_iteration > n_policy_iteration:
        value = policy_evaluation(env, policy, gamma, theta, max_iteration)
        policy = policy_improvement(env, value, gamma)

        n_policy_iteration += 1

        if np.all(np.equal(previous_policy, policy)):
            break
        else:
            previous_policy = policy

    value = policy_evaluation(env, policy, gamma, theta, max_iteration)

    return policy, value