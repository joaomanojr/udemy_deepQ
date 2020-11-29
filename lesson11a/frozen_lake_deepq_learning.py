
# import sys
# gym_dir='/home/joaomano/.local/lib/python3.6/site-packages'
# sys.path.append(gym_dir)

import gym
import matplotlib.pyplot as plt
import numpy as np
from q_learning_agent import Agent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    n_games = 10000
    scores = []
    win_pct_list = []

    agent = Agent(lr=0.0001, n_actions=4)

    for i in range(n_games):
        score = 0
        done = False
        # This looks like a context pointer to created environment, here we are
        # just initializing it - it returns our current state also;
        obs = env.reset()

        # Interact with environment until done (end or fall in a hole)
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        scores.append(score)

        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            print('episode', i, 'win pct %.2f' % win_pct,
                  'epsilon %.2f' % agent.epsilon)

    plt.plot(win_pct_list)
    plt.show()