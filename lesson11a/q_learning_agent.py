import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input, 128)
        self.fc2 = nn.Linear(128, n_actions)

        # Author: self.parameters() from inherited class Module
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # Author: pytorch have different tensors for cuda/cpu devices
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        # Author: MSELoss will take care of activation for us...
        actions = self.fc2(layer1)

        return actions


class Agent():
    def __init__(self, lr, n_actions, gamma=0.95,
                 epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        """ Agent init takes:
        --
        lr - alpha learning rate factor
        input_dims - from our environment dimensions
        n_actions - actions space dimension
        gamma - discount factor on MDP rewards
        epsilon - Epsilon Greedy initial value (exploration threshold)
        eps_dec - Epsilon Greedy decrease factor
        eps_min - Epsilon Greedy minimum, final value (must be > 0)
        
        """
        self.lr = lr
        self.input_dims = 1
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        ''' Choose Epsilon Greedy action for a given state '''
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            # https://stackoverflow.com/questions/64192810/runtime-error-both-arguments-to-matmul-need-to-be-at-least-1d-but-they-are-0d
            actions = self.Q.forward(state.unsqueeze(dim=0))
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def decrement_epsilon(self):
        ''' Epsilon decrease function (linear) '''
        # Look: my beloved C ternary in python terms!
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
    
    def learn(self, state, action, reward, state_):
        """ Off Policy (always Greedy) Learn function 
        --
        Here defined as plain Bellman equation, state_ is state'
        """
        self.Q.optimizer.zero_grad()

        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states.unsqueeze(dim=0))[actions]
        q_next = self.Q.forward(states_.unsqueeze(dim=0)).max()

        q_target = reward + self.gamma*q_next

        # evaluate loss (cost) as the difference at better known and actual
        # action.
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        # Author: backpropagate cost and add a step on our optimizer.
        # These two calls are critical for learn loop.
        loss.backward()

        self.Q.optimizer.step()
        self.decrement_epsilon()
