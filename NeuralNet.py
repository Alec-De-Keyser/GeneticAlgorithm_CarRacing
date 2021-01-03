########################################################################################################################
# IMPORTS
########################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.fitness = 0

        # define the hidden layers
        # creates a linear transformation module: xW + bxW + b
        #   where x = input, W = weight and b = bias
        # transformation between input and first hidden layer
        # this creates the weight and bias 'tensors' (I guess theta)
        self.hidden1 = nn.Linear(num_input_nodes, num_hidden_nodes)
        # transformation between first and second hidden layer
        self.hidden2 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
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
    def get_flat_weights(self):
        weights = []
        hidden1_data = self.hidden1.weight.data
        hidden2_data = self.hidden2.weight.data
        output_data = self.output.weight.data
        for w in hidden1_data:
            for i in w:
                weights.append(i.tolist())
        for w in hidden2_data:
            for i in w:
                weights.append(i.tolist())
        for w in output_data:
            for i in w:
                weights.append(i.tolist())
        return weights

    ####################################################################################################################
    # UPDATE_WEIGHTS
    #   Updates weights, based on the input.
    #   The weights should be decided by the optimisation algorithm.
    #       In this case, a genetic algorithm is used.
    #   For loops because the weight.data variables need a single tensor of a particular shape.
    ####################################################################################################################
    def update_weights(self, weights):
        layer1_weights = self.num_input_nodes * self.num_hidden_nodes
        layer2_weights = self.num_hidden_nodes ** 2
        num_input_nodes = self.num_input_nodes
        num_hidden_nodes = self.num_hidden_nodes
        num_output_nodes = self.num_output_nodes
        hidden1_weights = []
        hidden2_weights = []
        output_weights = []

        for i in range(num_hidden_nodes):
            hidden1_weights.append(weights[i * num_input_nodes: i * num_input_nodes + num_input_nodes])

            hidden2_weights.append(weights[i * num_hidden_nodes + layer1_weights:
                                           i * num_hidden_nodes + layer1_weights + num_hidden_nodes])
        self.hidden1.weight.data = torch.Tensor(hidden1_weights)
        self.hidden2.weight.data = torch.Tensor(hidden2_weights)

        for i in range(num_output_nodes):
            output_weights.append(weights[i * num_hidden_nodes + layer1_weights + layer2_weights:
                                  i * num_hidden_nodes + layer1_weights + layer2_weights + num_hidden_nodes])
        self.output.weight.data = torch.Tensor(output_weights)


