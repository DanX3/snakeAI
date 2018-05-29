from SnakeEnv import SnakeEnv2
import numpy as np
import pickle
import neat
import os
from math import tanh
from time import sleep
import visualize
import threading

GRID = (10, 10)
THREAD_COUNT = 8

def split_load(start, end, workers_count):
    count = end - start + 1
    loads = []
    for i in range(workers_count):
        load = count / workers_count + (i < count % workers_count)
        loads.append((start, start + load - 1))
        start += load
    return loads


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


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def thread_worker(genomes, config, start, end):
    print 'worker in ({}, {})'.format(start, end)
    myenv = SnakeEnv2(grid=np.array(GRID), render=False)
    for i in range(start, end+1):
        genome_id, genome = genomes[i]
        genome.fitness = fitness(myenv, genome, config, render=False)
        # print "Genome #{} fitness = {}".format(genome_id, genome.fitness)
    myenv.close()


def eval_genomes(genomes, config):
    threads = []
    loads = split_load(0, len(genomes)-1, THREAD_COUNT)
    for start, end in loads:
        new_thread = threading.Thread(target=thread_worker,
                                      args=(genomes, config, start, end))
        threads.append(new_thread)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    del threads[:]


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes, 20)
    print "Best Genome {!s}".format(winner)
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    fitness(SnakeEnv2(grid=np.array((10, 10))), winner, config, render=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    

if __name__ == '__main__':
    localdir = os.path.dirname(__file__)
    config_path = os.path.join(localdir, 'neat_config')
    run(config_path)
