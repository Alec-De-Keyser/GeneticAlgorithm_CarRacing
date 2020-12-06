from NeuralNet import NeuralNet
import numpy as np


class GeneticAlgorithm:

    def __init__(self, population_size, num_input_nodes, num_hidden_nodes, num_output_nodes):
        self.population_size = population_size
        self.nets = []
        self.i1 = 0
        self.i2 = 0
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes
        self.num_weights = (self.num_input_nodes * self.num_hidden_nodes) \
            + (self.num_hidden_nodes ** 2) * 2 \
            + (self.num_hidden_nodes * self.num_output_nodes)
        self.init_population()

    def init_population(self, ):
        for i in range(0, self.population_size):
            self.nets.append(NeuralNet(num_input_nodes=self.num_input_nodes, num_hidden_nodes=self.num_hidden_nodes,
                                       num_output_nodes=self.num_output_nodes))
        return self.nets

    def calc_fitness(self, reward):
        for i in range(0, self.population_size):
            self.nets[i].fitness = reward

    def set_distance_traveled(self, distances):
        for i in range(0, self.population_size):
            self.nets[i].distance = distances[i]

    def choose_parents(self):
        for j in range(2):
            for i in range(0, self.population_size):
                if self.nets[i].fitness >= self.nets[self.i1].fitness:
                    self.i1 = i
                elif self.nets[i].fitness >= self.nets[self.i2].fitness:
                    self.i2 = i

    def crossover(self):
        net1 = self.nets[self.i1]
        net2 = self.nets[self.i2]
        weights = np.array(self.num_weights)
        for w in range(0, self.num_weights):
            if w < self.num_weights / 5:
                weights[w] = net1[w].weights
            elif w < self.num_weights * 4 / 5:
                weights[w] = net2[w].weights

        for i in range(0, self.population_size):
            self.nets[i].update_weights(num_weights=self.num_weights, weights=weights)