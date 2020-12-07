########################################################################################################################
# IMPORTS
########################################################################################################################
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNet
import numpy as np
import torch
import gym.envs.box2d.car_racing
import csv


########################################################################################################################
# GET_OUTPUT
#   Inputs the values of obs into the NeuralNet net and returns the output of net.
########################################################################################################################
def get_output(obs, net):
    input_list = []
    x = []
    for o in obs:
        for j in o:
            for k in j:
                input_list.append(k)
    x = net.forward(torch.Tensor(input_list)).tolist()
    return x


def write_best_to_file(genetic, generation):
    best = genetic.nets[genetic.i1].get_flat_weights()
    filename = 'weights' + str(generation) + '.csv'
    file = open(filename, 'w')
    writer = csv.writer(file)
    for weight in best:
        writer.writerow([weight])


def read_file(net, generation):
    filename = 'weights' + str(generation) + '.csv'
    file = open(filename)
    reader = csv.reader(file)
    weights = []
    for row in reader:
        weights.append(float(row[0]))
    net.update_weights(weights)


if __name__ == '__main__':
    for gen in [3, 10, 20, 23]:
        best_net = NeuralNet(96*96*3, 50, 3)
        read_file(best_net, gen)
        env = gym.make('CarRacing-v0')
        reward = float(0)
        rewards = []
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = get_output(observation, best_net)
            observation, reward, done, _ = env.step(action)
        env.close()
    ga = GeneticAlgorithm(15, 96 * 96 * 3, 50, 3)
    for gen in range(100):
        for i in range(ga.population_size):
            reward = float(0)
            rewards = []
            observation = env.reset()
            done = False
            while not done:
                # env.render()
                action = get_output(observation, ga.nets[i])
                observation, reward, done, _ = env.step(action)
                rewards.append(reward)
            total_reward = np.sum(rewards) + 100
            print(total_reward)
            ga.nets[i].fitness = total_reward
            env.close()
        env.close()
        print("Processing generation ...")
        ga.choose_parents()
        ga.crossover()
        print("Mutating ...")
        ga.mutate()
        print("NEW GENERATION!")
        write_best_to_file(ga, gen)
