from __future__ import print_function
import os
import neat
import gym

def eval_genomes(genomes, config):
    env = gym.make('CartPole-v0')
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        observation = env.reset()
        for t in range(1000):
            action = net.activate(observation)
            observation, reward, done, info = env.step(action[0]>0.5)
            genome.fitness += reward/5
            if done:
                break
    env.close()



def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 70)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make('CartPole-v0')
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = winner_net.activate(observation)
        observation, reward, done, info = env.step(action[0]>0.5)
        if done:
            break

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-69')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-cartpole')
    run(config_path)
