from SnakeEnv import SnakeEnv2
import numpy as np
import pickle
import neat
import os
from math import tanh
from time import sleep
import visualize
import threading
import signal


def run_single_experiment():
    grid = (5, 5)
    area = grid[0] * grid[1]
    myenv = SnakeEnv(grid=grid)
    Q = np.array((np.zeros(area*area*4)))
    Q = np.reshape(Q, (area*area, 4))
    body_conf = {}

    while True:
        head_pos, fruit_pos = myenv.reset()
        t = 1
        observation = None
        while True:
            myenv.render()
            myenv.unwrapped.viewer.window.on_key_press = key_press

            state_idx = state_to_idx(head_pos, fruit_pos , grid, myenv.get_body_configuration())
            choices = Q[state_idx]
            # print choices
            choice = np.argmax(choices)
            observation, reward, done, info = myenv.step(choice)
            head_pos, fruit_pos = observation
            # updates the whole matrix slightly despite the fruit position
            # if the snake hit the wall, the fruit position is not important
            if reward > 0.0:
                Q[state_idx][choice] += reward
            else:
                for i in range(area):
                    ith_state = state_to_idx(head_pos, (i/grid[1], i%grid[1]), grid, myenv.get_body_configuration())
                    Q[ith_state][choice] += reward * (0.1 if ith_state != state_idx else 1.0)
            t += 1
            if done:
                # print "Episode finished after {} timesteps".format(t)
                myenv.close()
                break


def fitness(env, genome, config, render=False):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    head_pos, fruit_pos, dir, tiles_obs, tiles_to_fruit = env.reset()
    # updown = np.random.randint(2)
    # dir = np.array((updown, 0 if updown else 1)) * np.power(-1, np.random.randint(2))
    # print dir
    while True:
        if render:
            sleep(0.1)
            env.render()
        actions = net.activate([head_pos[0], head_pos[1],
                                fruit_pos[0], fruit_pos[1],
                                dir[0], dir[1],   # default direction
                                tiles_obs,
                                int(tiles_to_fruit)
                                ])
        action_taken = np.argmax(softmax(actions))
        obs, reward, done, info = env.step(action_taken)

        head_pos, fruit_pos, dir, tiles_obs, tiles_to_fruit = obs
        if done:
            genome.fitness = info['t']  # * tanh(turns / 1e3)
            # line_fun = lambda x: 10 - 0.5 * x
            # genome.fitness = 10*info['len'] - line_fun(info['t'])  # * tanh(turns / 1e3)
            return info['t']
            break


# softmax = lambda x : np.exp(x)/np.sum(np.exp(x))
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def run_timeout(signum, frame):
    print 'interrupted function'
    raise Exception("Too much time")


def eval_genomes(genomes, config):
    grid = (10, 10)
    myenv = SnakeEnv2(grid=np.array(grid), render=False)
    for genome_id, genome in genomes:
        # signal.signal(signal.SIGALRM, run_timeout)
        # signal.alarm(1)
        # try:
        genome.fitness = fitness(myenv, genome, config, render=False)
        # except:
            # print 'skipping genome {}'.format(genome_id)
            # genome.fitness = 0
        # print "Genome #{} fitness score = {}".format(genome_id, genome.fitness)
    # signal.alarm(0)
    myenv.close()

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes, 250)
    print "Best Genome {!s}".format(winner)
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    fitness(SnakeEnv2(grid=np.array((10, 10))), winner, config, render=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    

if __name__ == '__main__':
    localdir = os.path.dirname(__file__)
    config_path = os.path.join(localdir, 'neat_config')
    run(config_path)
