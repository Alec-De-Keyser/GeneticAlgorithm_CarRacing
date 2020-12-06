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
            + (self.num_hidden_nodes ** 2) \
            + (self.num_hidden_nodes * self.num_output_nodes)
        self.init_population()

    def init_population(self, ):
        for i in range(0, self.population_size):
            self.nets.append(NeuralNet(num_input_nodes=self.num_input_nodes, num_hidden_nodes=self.num_hidden_nodes,
                                       num_output_nodes=self.num_output_nodes))
            random_weights = np.random.uniform(-1, 1, self.num_weights)
            self.nets[i].update_weights(random_weights)

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
        weights1 = net1.get_flat_weights()
        weights2 = net2.get_flat_weights()
        new_weights = []
        for w in range(self.num_weights):
            if w < self.num_weights * 1 / 10:
                new_weights.append(weights2[w])
            elif w < self.num_weights * 9 / 10:
                new_weights.append(weights1[w])
            else:
                new_weights.append(weights2[w])
        for net in self.nets:
            net.update_weights(new_weights)

