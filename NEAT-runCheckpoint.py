from __future__ import print_function
import os
import neat
import gym
import numpy as np
import pandas as pd
#import tensorflow as tf


def run(config_file):

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-497')

    pop = p.population
    net = []
    i = 0
    for id, genome in pop.items():
        net.append(neat.nn.FeedForwardNetwork.create(genome, config))

    env = gym.make('Breakout-ram-v0')
    timeoutVal = 200

    
    while True:
        fitness = 0
        countNoScore = timeoutVal
        observation = env.reset()
        observation, reward, done, info = env.step(1)
        while True:
            env.render()
            action = [0,0,0,0]

            for n in net:
                a = n.activate(observation/255)
                action[np.argmax(a)] += 1
            #print(action)
            observation, reward, done, info = env.step(np.argmax(action))

            fitness += reward
            if reward == 0:
                countNoScore -= 1
            else:
                countNoScore = timeoutVal
            if done or countNoScore==0:
                break
        print('fitness:{}'.format(fitness))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-breakout')
    run(config_path)