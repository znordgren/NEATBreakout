from __future__ import print_function
import os
import neat
import gym
import numpy as np


def eval_genomes(genomes, config):
    env = gym.make('Breakout-ram-v0')
    timeoutVal = 150
    for genome_id, genome in genomes:

        countNoScore = timeoutVal
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        observation = env.reset()
        while True:
            action = net.activate(observation/255)
            observation, reward, done, info = env.step(np.argmax(action))
            genome.fitness += reward
            if reward == 0:
                countNoScore -= 1
            else:
                countNoScore = timeoutVal
            if done or countNoScore==0:
                break

    env.close()



def run(config_file):

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=5,time_interval_seconds=10000))
    
    print('Starting Training')
    winner = p.run(eval_genomes, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make('Breakout-ram-v0')
    observation = env.reset()
    while True:
            action = net.activate(observation/255)
            observation, reward, done, info = env.step(np.argmax(action))
            genome.fitness += reward
            if done:
                break

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-breakout')
    run(config_path)
