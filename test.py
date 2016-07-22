import random
import data as nn

__author__ = 'Timothy'


# THIS IS TESTING DATA!

# example 1:
def example1():

    num_inputs = 2
    num_outputs = 2
    num_hidden_layers = 1
    num_npl = 2
    biases = [.35, .60]

    net = nn.NeuralNet(num_inputs, num_outputs, num_hidden_layers, num_npl, biases)

    training_data = [[0.05, 0.1], [0.01, 0.99]]
    net.put_weights2d([[[.15, .20], [.25, .30]], [[.4, .45], [.5, .55]]])

    print("first outs: " + str(net.forward_pass(training_data[0])))
    for _ in range(1):
        net.train(training_data[0], training_data[1])
    print("new outs: " + str(net.forward_pass(training_data[0])))
    net.get_net_state()


# example 2: XOR gate
def example2():
    num_inputs = 2
    num_outputs = 1
    num_hidden_layers = 1
    num_npl = 5

    net = nn.NeuralNet(num_inputs, num_outputs, num_hidden_layers, num_npl)
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [1])
    ]
    for _ in range(10000):
        inp, out = random.choice(training_data)
        net.train(inp, out)
        #print(net.calculate_error(training_data))

    print("0, 0: " + str(net.forward_pass([0, 0])))
    print("0, 1: " + str(net.forward_pass([0, 1])))
    print("1, 0: " + str(net.forward_pass([1, 0])))
    print("1, 1: " + str(net.forward_pass([1, 1])))

example2()

