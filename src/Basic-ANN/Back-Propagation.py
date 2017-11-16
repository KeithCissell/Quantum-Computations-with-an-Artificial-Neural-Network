import math
import random
# import matplotlib.pyplot as plt


def initialize_weights(num_weights):
    weights = []

    rand_min = 0
    rand_max = 0.5

    for i in range(num_weights):
        rand = rand_min + ((rand_max - rand_min) * random.random())
        weights.append(rand)

    return weights

def build_neuron(num_inputs):
    neuron = {}
    neuron["weights"] = initialize_weights(num_inputs + 1)
    neuron["last_delta"] = [0.0] * (num_inputs + 1)
    neuron["deriv"] = [0.0] * (num_inputs + 1)
    return neuron

def build_network(hidden_layer_pattern, num_inputs):
    network = []

    # Add hidden layers
    num_layer_inputs = num_inputs

    for i in range(len(hidden_layer_pattern)):
        layer = []
        num_neurons = hidden_layer_pattern[i]
        for j in range(num_neurons):
            neuron = build_neuron(num_layer_inputs)
            layer.append(neuron)
        network.append(layer)
        num_layer_inputs = num_neurons

    # Build Output Layer
    output_layer = []
    for i in range(num_inputs):
        neuron = build_neuron(num_layer_inputs)
        output_layer.append(neuron)
    network.append(output_layer)

    return network

def print_network_topology(network):
    print("")
    print("NETWORK TOPOLOGY")
    print("Input Layer: ", len(network[-1]))

    for i in range(len(network) - 1):
        print("Layer", (i + 1), ": ", len(network[i]))

    print("OutputLayer: ", len(network[-1]))




if __name__ == "__main__":
    print("QUANTUM NETWORK COMENSING")

    my_hidden_layers = [2, 2]
    my_network = build_network(my_hidden_layers, 1)

    print_network_topology(my_network)
