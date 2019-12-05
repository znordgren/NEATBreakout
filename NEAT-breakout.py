'''
NEAT algorithm to play atari Breakout

requires neat-python to run which can be installed with 'pip install neat-python'
Also requires the openai gym which can be installed with 'pip install gym'

'''



from __future__ import print_function
import os
import neat
import gym
import numpy as np
import pandas as pd
#import tensorflow as tf

def eval_genomes(genomes, config):
    #ae = tf.keras.models.load_model('ae.h5')
    env = gym.make('Breakout-ram-v0')
    timeoutVal = 150
    observationArray = []
    observationIndex = [18,30,49,52,70,71,72,74,75,86,90,91,94,95,96,99,100,101,102,103,104,105,106,107,119,121,122]
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = [0,0,0]

        for iRun in range(0,len(fitness)):
            countNoScore = timeoutVal
            observation = env.reset()
            #observation, reward, done, info = env.step(1)
            while True:
                #latentObservations = ae.predict(np.array([observation])) # call autoencoder
                #print(latentObservations[0])
                #action = net.activate(latentObservations[0]/255)

                action = net.activate(observation[observationIndex]/255)
                observation, reward, done, info = env.step(np.argmax(action))
                #observationArray.append(observation)
                fitness[iRun] += reward
                if reward == 0:
                    countNoScore -= 1
                else:
                    countNoScore = timeoutVal
                if done or countNoScore==0:
                    break
        
        genome.fitness = np.mean(fitness)

    #df = pd.read_csv('observations.csv',index_col=0)
    #df = df.sample(frac=0.5,)
    #df = df.append(pd.DataFrame(observationArray))
    #df.to_csv('observations.csv')

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
