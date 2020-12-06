########################################################################################################################
# IMPORTS
########################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    ####################################################################################################################
    # __INIT__
    #   Initialise instance of class NeuralNet
    #   Has 2 hidden layers
    #   Uses the same size for all hidden layers (for now)
    ####################################################################################################################
    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes):
        super(NeuralNet, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes
        # The following properties are only used in the genetic algorithm
        #   Delete them if you don't use them
        self.alive = 1
        self.distance = 0
        self.fitness = 0

        # define the hidden layers
        # creates a linear transformation module: xW + bxW + b
        #   where x = input, W = weight and b = bias
        # transformation between input and first hidden layer
        # this creates the weight and bias 'tensors' (I guess theta)
        self.hidden1 = nn.Linear(num_input_nodes, num_hidden_nodes)
        # transformation between first and second hidden layer
        self.hidden2 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        # transformation between second and third hidden layer
        self.hidden3 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        # define the output layers
        # transformation between last hidden layer and output
        self.output = nn.Linear(num_hidden_nodes, num_output_nodes)

    ####################################################################################################################
    # FORWARD
    #   Implements the forward propagation through the network
    #   Should be reimplemented if using this network for another reason.
    #   The current implementation designs a neural network that has no special features
    ####################################################################################################################
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x

    ####################################################################################################################
    # GET_WEIGHTS
    #   Returns the weights of all layers in a flat list
    ####################################################################################################################
    def get_weights(self):
        weights = []
        for w in self.hidden1.weight.data:
            weights.append(w.tolist())
        for w in self.hidden2.weight.data:
            weights.append(w.tolist())
        for w in self.output.weight.data:
            weights.append(w.tolist())
        flat_weights = []
        for i in weights:
            for j in i:
                flat_weights.append(j)
        return flat_weights

    ####################################################################################################################
    # UPDATE_WEIGHTS
    #   Updates weights, based on the input of the genetic algorithm
    #   The weights should be decided by the optimisation algorithm.
    #       In this case, a genetic algorithm is used
    ####################################################################################################################
    def update_weights(self, num_weights, weights):
        layer1_weights = self.num_input_nodes * self.num_hidden_nodes
        layer2_weights = self.num_hidden_nodes ** 2
        layer3_weights = layer2_weights
        for i in range(0, num_weights):
            if i < layer1_weights:
                self.hidden1[i].weights = weights[i]
            elif i < layer1_weights + layer2_weights:
                self.hidden2[i - layer1_weights].weights = weights[i]
            elif i < layer1_weights + layer2_weights + layer3_weights:
                self.hidden3[i - layer1_weights - layer2_weights] = weights[i]
            else:
                self.output[i - layer1_weights - layer2_weights - layer3_weights] = weights[i]


