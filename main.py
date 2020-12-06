from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNet
import numpy as np
import gym.envs.box2d.car_racing


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ga = GeneticAlgorithm(10, 96*96*3, 50, 3)
    ga.init_population()
    env = gym.make('CarRacing-v0')
