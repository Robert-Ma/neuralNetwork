import numpy as np
from scipy.special import expit


# neural network  class definitoin
class neuralNetwork:
    # initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate,
                 wih=None, who=None, act_fun=None):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to
        # node j in the next layer
        # w11 w21 w31
        # w12 w22 w32
        # w13 w23 w33 etc
        # self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        if wih is None:
            self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                        (self.hnodes, self.inodes))
        if who is None:
            self.who = np.random.normal(0.0, pow(self.onodes, -0.5),
                                        (self.onodes, self.hnodes))
        if act_fun is None:
            # activation function is the sigmod function
            self.activation_function = lambda x: expit(x)

    def train(self, inputs_list, targets_list):
        """train the neural network"""

        # convert input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights,
        # recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # update the weights for the link between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))
        # TODO

    def predict(self, inputs_list):
        """predict the neural network"""

        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
