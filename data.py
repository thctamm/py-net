__author__ = 'Timothy H. C. Tamm'

import random
import math

random.seed()


# function for the NeuralNet output values
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Calculates the total error between expected outputs and actual outputs
def total_error(ideal, actual):
    if len(ideal) != len(actual):
        raise ValueError("Invalid input")
    sum = 0
    for i in range(len(ideal)):
        sum += 0.5*(ideal[i] - actual[i])**2
    return sum


# delta function for backpropagation
def delta(out, target):
    return -(target-out)*out*(1-out)


# Class of a neuron in the net.
class Neuron(object):
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = []
        self.output = 0
        for _ in range(num_inputs):
            if random.random() > 0.5:
                self.weights.append(random.random() * 0.05)
            else:
                self.weights.append(random.random() * -0.05)

    # sums the inputs weighted with the weights.
    def sum(self, inputs):
        S = 0.0
        for i, val in enumerate(inputs):
            S += val*self.weights[i]
        return S

    # given an array of weights, sets them as the Neuron's weights.
    def set_weights(self, weights):
        self.weights = weights


# Neuron Layer class. Holds all the neurons in that layer.
class NeuronLayer(object):
    def __init__(self, num_neurons, num_inputs, bias=0):
        self.num_neurons = num_neurons
        self.neurons = []
        self.bias = bias
        for _ in range(num_neurons):
            self.neurons.append(Neuron(num_inputs))


# Neural net class. holds all the neuron layers.
class NeuralNet(object):

    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_neurons_per_hidden_layer, biases=None, rate=0.5):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_hidden_layer = num_neurons_per_hidden_layer
        self.learning_rate = rate
        if biases is not None:
            if self.num_hidden_layers > 1:
                self.layers = [NeuronLayer(num_neurons_per_hidden_layer, num_inputs, biases[0])]
                self.layers += [NeuronLayer(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer, biases[i+1])
                                for i in range(num_hidden_layers - 1)]
                self.layers += [NeuronLayer(num_outputs, num_neurons_per_hidden_layer)]
            elif self.num_hidden_layers > 0:
                self.layers = [NeuronLayer(num_neurons_per_hidden_layer, num_inputs, biases[0])]
                self.layers += [NeuronLayer(num_outputs, num_neurons_per_hidden_layer, biases[1])]
            else:
                self.layers = [NeuronLayer(num_outputs, num_inputs, biases[0])]
        else:
            if self.num_hidden_layers > 1:
                self.layers = [NeuronLayer(num_neurons_per_hidden_layer, num_inputs)]
                self.layers += [NeuronLayer(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer)
                                for _ in range(num_hidden_layers - 1)]
                self.layers += [NeuronLayer(num_outputs, num_neurons_per_hidden_layer)]
            elif self.num_hidden_layers > 0:
                self.layers = [NeuronLayer(num_neurons_per_hidden_layer, num_inputs)]
                self.layers += [NeuronLayer(num_outputs, num_neurons_per_hidden_layer)]
            else:
                self.layers = [NeuronLayer(num_outputs, num_inputs)]

    # function for outputting all the weights in the net as a 1D array.
    def get_weights(self):
        out = []
        for l in self.layers:
            for n in l.neurons:
                for w in n.weights:
                    out.append(w)
        return out

    def get_net_state(self):
        for layer in self.layers:
            print("layer: ")
            for neuron in layer.neurons:
                print("neuron: ")
                print(neuron.weights)
                print(neuron.output)

    def calculate_error(self, data):
        error = 0
        for input, output in data:
            actual_out = self.forward_pass(input)
            error += total_error(output, actual_out)
        return error

    # outputs the total number of weights in the net
    def num_of_weights(self):
        if self.num_hidden_layers == 0:
            return self.num_outputs
        else:
            return self.num_outputs + (self.num_hidden_layers + 1) * self.num_hidden_layers

    # Given a 2D array of weights, starting with the first neuron in the first layer and
    # ending with the last neuron in the last array, sets the weights of the all the neurons in the net.
    def put_weights2d(self, weights):
        for i, layer in enumerate(self.layers):
            for n, neuron in enumerate(layer.neurons):
                neuron.set_weights(weights[i][n])

    def put_weights1d(self, weights):
        i = 0
        for l in self.layers:
            for n in l.neurons:
                for w in range(len(n.weights)):
                    n.weights[w] = weights[i]
                    i += 1

    # Given an input, calculates the output of the NN.
    def forward_pass(self, inputs):
        outputs = []
        for layer in self.layers:
            outputs = []
            for n in layer.neurons:
                out = sigmoid(n.sum(inputs) + layer.bias)
                n.output = out
                outputs.append(out)
            inputs = outputs
        return outputs

    # Backpropagation training. Only works with 1 hidden layer as of now
    def train(self, inputs, expected_outputs, debug=False):

        if self.num_hidden_layers != 1:
            raise ValueError("Backpropagation only works with 1 hidden layer")

        # run the inputs through the net
        original_out = self.forward_pass(inputs)

        # handle output layer
        output_deltas = [0 for _ in range(self.num_outputs)]
        for i in range(self.num_outputs):
            neuron = self.layers[1].neurons[i]
            output_deltas[i] = delta(neuron.output, expected_outputs[i])

        # Handle hidden layer
        hidden_deltas = [0 for _ in range(self.num_neurons_per_hidden_layer)]
        for i in range(self.num_neurons_per_hidden_layer):
            neuron = self.layers[0].neurons[i]
            change = 0
            for w in range(len(self.layers[1].neurons)):
                change += output_deltas[w] * self.layers[1].neurons[w].weights[i]
            hidden_deltas[i] = change * neuron.output * (1-neuron.output)

        # update output layer
        for n in range(self.num_outputs):
            for w in range(len(self.layers[1].neurons[n].weights)):
                self.layers[1].neurons[n].weights[w] -= self.learning_rate * output_deltas[n] * self.layers[0].neurons[w].output

        # update hidden layer
        for n in range(self.num_neurons_per_hidden_layer):
            for w in range(len(self.layers[0].neurons[n].weights)):
                self.layers[0].neurons[n].weights[w] -= self.learning_rate * hidden_deltas[n] * inputs[w]

        new_out = self.forward_pass(inputs)

        if debug:
            print("original error: " + str(total_error(expected_outputs, original_out)))
            print("new error: " + str(total_error(expected_outputs, new_out)))



