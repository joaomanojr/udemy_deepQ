import numpy as np

class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end,
                 eps_dec):
        """ Agent init takes:
        --
        lr - alpha learning rate factor
        gamma - discount factor on MDP rewards
        n_actions - actions space dimension
        n_states - state space dimension
        eps_start - Epsilon Greedy initial value (exploration threshold)
        eps_end - Epsilon Greedy final value (must be > 0)
        eps_dec - Epsilon Greedy decrease factor
        
        """
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q ={}

        self.init_Q()
    
    def init_Q(self):
        """ Initialize our Tabular Learning Q dictionary
        --
        Indeed only final state must be zeroed - all initialized as zero for
        sake of simplicity.
        """
        for state in range(self.n_states):
            for action in range(self.n_actions):
                # note: use tuples as dictionary keys!
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        ''' Choose Epsilon Greedy action for a given state '''
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            # this is pure beauty: create a list comprehension getting all
            # actions' values considering past agent experience on this state!
            actions = np.array([self.Q[(state, a)] \
                                       for a in range(self.n_actions)])
            # author note: if more than one element have the maximum value it
            # returns always the one with lower index - can be improved with
            # a external custom function.
            action = np.argmax(actions)
        return action
    
    def decrement_epsilon(self):
        ''' Epsilon decrease function (linear) '''
        # Look: my beloved C ternary in python terms!
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon>self.eps_min\
                       else self.eps_min
    
    def learn(self, state, action, reward, state_):
        """ Off Policy (always Greedy) Learn function 
        --
        Here defined as plain Bellman equation, state_ is state'
        """
        actions = np.array([self.Q[state_, a] for a in range(self.n_actions)])
        a_max = np.argmax(actions)

        self.Q[(state, action)] += self.lr*(reward +
                                        self.gamma*self.Q[(state_, a_max)] -
                                        self.Q[(state, action)])

        self.decrement_epsilon()




            
