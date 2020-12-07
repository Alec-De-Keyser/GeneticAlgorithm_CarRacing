from NeuralNet import NeuralNet
import numpy as np


class GeneticAlgorithm:
    ####################################################################################################################
    # __INIT__
    #   Initialises instance of class GeneticAlgorithm.
    #   Makes the necessary variables and calls function init_population() to initialise the population.
    ####################################################################################################################
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

    ####################################################################################################################
    # INIT_POPULATION
    #   Makes an initial population, with uniformly distributed random weights.
    ####################################################################################################################
    def init_population(self):
        for i in range(0, self.population_size):
            self.nets.append(NeuralNet(num_input_nodes=self.num_input_nodes, num_hidden_nodes=self.num_hidden_nodes,
                                       num_output_nodes=self.num_output_nodes))
            random_weights = np.random.uniform(-1, 1, self.num_weights)
            self.nets[i].update_weights(random_weights)

        return self.nets

    ####################################################################################################################
    # CALC_FITNESS
    #   Calculates the fitness score (in the case of gym training, it simply receives and adds it to each net).
    #   TODO: for ROS implementation.
    ####################################################################################################################
    def calc_fitness(self, reward):
        for i in range(0, self.population_size):
            self.nets[i].fitness = reward

    ####################################################################################################################
    # CHOOSE_PARENTS
    #   Picks the two nets with the best fitness score and saves them in parameters self.i1 and self.i2
    #   Loops through the population twice. The first time getting the best, the second time getting the second best.
    ####################################################################################################################
    def choose_parents(self):
        for j in range(2):
            for i in range(0, self.population_size):
                if self.nets[i].fitness >= self.nets[self.i1].fitness:
                    self.i1 = i
                elif self.nets[i].fitness >= self.nets[self.i2].fitness:
                    self.i2 = i

    ####################################################################################################################
    # CROSSOVER
    #   Uses the flat weights (no tensors) of the two best NeuralNets to come up with a new set of weights.
    #       The new weights are 2/10th of the second best and 8/10th of the best NeuralNet. This way, the best
    #       is rewarded more than the second best. The parts of the second best NeuralNet are put at the ends of the
    #       new weights.
    #   TODO: implement more complicated version
    ####################################################################################################################
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

    ####################################################################################################################
    # MUTATE
    #   For a random 10% of the population (on average), gets the flat weights and changes a random number of them,
    #       at random places to random values.
    #   in then calls update_weights with these new weights on every net that has changed.
    ####################################################################################################################
    def mutate(self):
        for net in range(self.population_size):
            random_chance = np.random.uniform(0, 1)
            if random_chance > 0.1: # 10% of the cases
                random_number = np.random.uniform(0, self.num_weights / 3)
                weights = self.nets[net].get_flat_weights()
                for _ in range(int(random_number)):
                    random_value = np.random.uniform(-10, 10)
                    random_place = int(np.random.uniform(0, self.num_weights))
                    weights[random_place] = random_value
                self.nets[net].update_weights(weights)




