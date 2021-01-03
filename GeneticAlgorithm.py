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
        get_random = np.random.uniform
        population_size = self.population_size
        num_input_nodes = self.num_input_nodes
        num_hidden_nodes = self.num_hidden_nodes
        num_output_nodes = self.num_output_nodes
        num_weights = self.num_weights

        for i in range(0, population_size):
            self.nets.append(NeuralNet(num_input_nodes=num_input_nodes, num_hidden_nodes=num_hidden_nodes,
                                       num_output_nodes=num_output_nodes))
            random_weights = get_random(-1, 1, num_weights)
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
    #   TODO: keep complete gene of two best circulating
    ####################################################################################################################
    def crossover(self):
        net1 = self.nets[self.i1]
        net2 = self.nets[self.i2]
        weights1 = net1.get_flat_weights()
        weights2 = net2.get_flat_weights()
        new_weights1 = []
        new_weights2 = []
        num_weights = self.num_weights

        for w in range(num_weights):
            if w < num_weights * 1 / 10:
                new_weights1.append(weights2[w])
                new_weights2.append(weights1[w])
            elif w < num_weights * 9 / 10:
                new_weights1.append(weights1[w])
                new_weights2.append(weights2[w])
            else:
                new_weights1.append(weights2[w])
                new_weights2.append(weights1[w])

        population_size = self.population_size
        half_pop = population_size / 2
        i1 = self.i1
        i2 = self.i2

        for net in range(population_size):
            if net != i1 and net != i2:
                if net < half_pop:
                    self.nets[net].update_weights(new_weights1)
                else:
                    self.nets[net].update_weights(new_weights2)

    ####################################################################################################################
    # MUTATE
    #   For a random 10% of the population (on average), gets the flat weights and changes a random number of them,
    #       at random places to random values.
    #   it then calls update_weights with these new weights on every net that has changed.
    #   TODO: more efficient
    #   TODO: in function of the generation -> later generation, less mutations (same chance)
    ####################################################################################################################
    def mutate(self):
        population_size = self.population_size
        i1 = self.i1
        i2 = self.i2
        num_weights = self.num_weights
        get_random = np.random.uniform

        for net in range(population_size):
            random_chance = get_random(0, 1)
            if net != i1 and net != i2 and random_chance > 0.2:  # 80% of the cases
                #random_number = int(get_random(0, num_weights / 1000))
                weights = self.nets[net].get_flat_weights()
                #for _ in range(random_number):
                random_value = get_random(-10, 10) #in for
                random_place = int(get_random(0, num_weights)) #in for
                weights[random_place] = random_value #in for
                self.nets[net].update_weights(weights)





