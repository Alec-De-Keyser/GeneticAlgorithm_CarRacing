from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNet
import numpy as np
import gym.envs.box2d.car_racing


def get_output(obs, net):
    print(obs)
    return obs


if __name__ == '__main__':

    ga = GeneticAlgorithm(10, 96*96*3, 50, 3)
    ga.init_population()
    env = gym.make('CarRacing-v0')
    reward = float(0)
    rewards = []
    observation = env.reset()
    done = False
    for i in ga.nets:
        x = get_output(observation, ga.nets[i])
#    while not done:

