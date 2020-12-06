########################################################################################################################
# IMPORTS
########################################################################################################################
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNet
import numpy as np
import torch
import gym.envs.box2d.car_racing


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


if __name__ == '__main__':

    ga = GeneticAlgorithm(5, 96*96*3, 50, 3)
    env = gym.make('CarRacing-v0')
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
        total_reward = np.sum(rewards)
        print(total_reward)
        ga.nets[i].fitness = total_reward
    ga.choose_parents()
    ga.crossover()
