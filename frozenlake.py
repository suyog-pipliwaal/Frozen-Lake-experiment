import numpy as np
class FrozenLake(Environment):

    def __init__(self, lake, slip, max_steps, seed=None):

        lake =  [ ['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']
                ]

        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1

        self.action_probabilities = np.load('p.npy')
        
        # TODO:
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done

    #Method P returns the probability of going from one state to another    
    def p(self, next_state, state, action):
        # TODO:

        expcted_state = self.take_action(state, action)
        return expcted_state == next_state
    
    #The method r returns the expected reward in having transitioned from state to next state given action
    def r(self, next_state, state, action):
        # TODO:
        
        if state in self.goal_states:
           return 1

        if state in self.trap_states:
           return -1

        return 0

    #NEW METHOD take_action returns the coordinates of the new state after taking an action
    def take_action(self, state, action):
        
        if self.is_endstate(state):
            return state

        if state in self.absorbing_state:
            return self.endstate()

        state_coordinates = self.state_to_coordinates(state)
        action_coordinates = self.action_to_coordinates(action)

        transition_state_coordinates = [

            state_coordinates[0] + action_coordinates[0],
            state_coordinates[1] + action_coordinates[1]

        ]

        next_state = self.coordinates_to_state(transition_state_coordinates)
        return next_state if self.valid_coordinates(transition_state_coordinates) else state

    #NEW METHOD that avoids picking cooridinates that are out of environment
    def valid_coordinates(self, coordinates):

        if (coordinates[0] < 0) or (coordinates[0] >= self.width):
            return False

        if (coordinates[1] < 0) or (coordinates[1] >= self.height):
            return False

        coordinates_state = self.coordinates_to_state(coordinates)
        return coordinates_state not in self.wall_indexes

    #NEW METHOD transfroms the states to coordinates on the environment
    def state_to_coordinates(self,state):

        assert state < self.grid_size, "{0} not on grid".format(state)

        x_index = state % self.width
        y_index = (state - x_index) / self.width
        return [x_index, y_index]

    #NEW METHOD transfroms the states to coordinates on the environment
    def coordinaes_to_state(self, coordinates):
        return (coordinates[1] * self.width) + coordinates[0]

    #NEW METHOD transfroms the actions to coordinates on the environment
    def action_to_coordinates(self, action):

        if action == 0: #UP
           return [0, 1]

        if action == 1: #RIGHT
           return [1, 0]

        if action == 2: #DOWN
           return [0, -1]

        if action == 3: #LEFT
           return [-1, 0]

        return [0, 0]
        
    def is_endstate(self, state):
        return state == self.endstate()

    def endstate(self):
        return self.grid_size

    def set_renderer(self, renderer):
        self.renderer = renderer

   
   
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))

