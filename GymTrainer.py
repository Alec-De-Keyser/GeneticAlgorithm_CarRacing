########################################################################################################################
# IMPORTS
########################################################################################################################
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNet
import numpy as np
import torch
import gym.envs.box2d.car_racing
import csv

class GymTrainer:

    ####################################################################################################################
    # GET_OUTPUT
    #   Inputs the values of obs into the NeuralNet net and returns the output of net.
    ####################################################################################################################
    def get_output(self, obs, net):
        input_list = []
        x = []
        for o in obs:
            for j in o:
                for k in j:
                    input_list.append(k)
        x = net.forward(torch.Tensor(input_list)).tolist()
        return x

    def write_best_to_file(self, genetic, generation):
        best = genetic.nets[genetic.i1].get_flat_weights()
        filename = 'weights' + str(generation) + '.csv'
        file = open(filename, 'w')
        writer = csv.writer(file)
        for weight in best:
            writer.writerow([weight])

    def read_file(self, net, generation):
        filename = 'weights' + str(generation) + '.csv'
        file = open(filename)
        reader = csv.reader(file)
        weights = []
        for row in reader:
            weights.append(float(row[0]))
        net.update_weights(weights)

    def train_gym(self):
        env = gym.make('CarRacing-v0')
        ga = GeneticAlgorithm(20, 96 * 96 * 3, 50, 3)
        for gen in range(200):
            for i in range(ga.population_size):
                reward = float(0)
                rewards = []
                observation = env.reset()
                done = False
                while not done:
                    # env.render()
                    action = self.get_output(observation, ga.nets[i])
                    observation, reward, done, _ = env.step(action)
                    rewards.append(reward)
                total_reward = np.sum(rewards) + 100
                print("car " + str(i) + ": " + str(total_reward))
                ga.nets[i].fitness = total_reward
                env.close()
            env.close()
            print("Processing generation ...")
            ga.choose_parents()
            ga.crossover()
            print("Mutating ...")
            ga.mutate()
            print("NEW GENERATION! Generation " + str(gen))
            self.write_best_to_file(ga, gen)

    def show_gym(self, generations):
        for gen in generations:
            best_net = NeuralNet(96*96*3, 50, 3)
            self.read_file(best_net, gen)
            env = gym.make('CarRacing-v0')
            observation = env.reset()
            done = False
            while not done:
                env.render()
                action = self.get_output(observation, best_net)
                observation, _, done, _ = env.step(action)
            env.close()
