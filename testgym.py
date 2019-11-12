'''
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
'''

import gym
env = gym.make('Breakout-ram-v0')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render('human')
env.close()