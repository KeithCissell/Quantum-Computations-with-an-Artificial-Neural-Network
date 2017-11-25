import math
import random

from TrainingData import dataReader

# import matplotlib.pyplot as plt


def get_random(rand_min, rand_max):
    return rand_min + ((rand_max - rand_min) * random.random())


def train_network(network, training_data, iterations, num_inputs, learning_rate):
    print_network_topology(network)
    # print(training_data)
    # print("Iterations: ", iterations)
    # print("Inputs: ", num_inputs)
    for i in range(iterations):
        training_data_clone = training_data[:]
        for td in range(len(training_data)):
            # Grab training datum in random order
            datum_index = round(get_random(0, len(training_data_clone) - 1))
            datum = training_data_clone.pop(datum_index)

            training_inputs = datum[0 : (num_inputs)]
            training_outputs = datum[num_inputs : len(datum)]

            forward_propagate(network, training_inputs)
            output_layer = network[-1]

            back_propagate_error(network, training_outputs)
            calculate_error_derivatives_for_weights(network, training_inputs)
        update_weights(network, learning_rate)



def forward_propagate(network, inputs):
    for i in range(len(network)):
        layer = network[i]
        layer_inputs = []

        # Set layer_inputs
        if (i == 0):
            layer_inputs = inputs
        else:
            previous_layer = network[i - 1]
            for neuron in previous_layer:
                layer_inputs.append(neuron["output"])

        # Activate neurons
        for neuron in layer:
            neuron["activation"] = activate(neuron["weights"], layer_inputs)
            neuron["output"] = transfer(neuron["activation"])
            # print(neuron["activation"])
            # print(neuron["output"])
            # print()


def activate(weights, inputs, bias = 1.0):
    # for each qubit
    activation_sum = weights[-1] * bias
    for i in range(len(inputs)):
        activation_sum += weights[i] * inputs[i]
    return activation_sum



def transfer(activation):
    # Sigmoid
    return 1.0 / (1.0 + math.exp(-activation))

def transfer_derivative(output):
    # Sigmoid derivative
    return output * (1.0 - output)


def back_propagate_error(network, expected_outputs):
    for i in range(len(network)):
        index = len(network) - 1 - i
        layer = network[index]

        # Output Layer
        if (index == len(network) - 1):
            for j in range(len(layer)):
                neuron = layer[j]
                error = expected_outputs[j] - neuron["output"]
                neuron["delta"] = error * transfer_derivative(neuron["output"])
        # Hidden Layers
        else:
            next_layer = network[index + 1]
            for j in range(len(layer)):
                neuron = layer[j]
                error_sum = 0.0
                for next_neuron in next_layer:
                    error_sum += next_neuron["weights"][j] * next_neuron["delta"]
                neuron["delta"] = error_sum * transfer_derivative(neuron["output"])
                # print(neuron["delta"])
                # print()


def calculate_error_derivatives_for_weights(network, inputs):
    for i in range(len(network)):
        layer = network[i]
        layer_inputs = []

        # First Hidden Layer
        if (i == 0):
            layer_inputs = inputs
        else:
            previous_layer = network[i - 1]
            for neuron in previous_layer:
                layer_inputs.append(neuron["output"])

        # Calculate
        for neuron in layer:
            for j in range(len(layer_inputs)):
                signal = layer_inputs[j]
                neuron["deriv"][j] += neuron["delta"] * signal
            # Bias's weight
            neuron["deriv"][-1] += neuron["delta"] * 1.0
            # print(neuron["deriv"])
            # print()


def update_weights(network, learning_rate, mom=0.8):
    for layer in network:
        for neuron in layer:
            for i in range(len(neuron["weights"])):
                delta = (learning_rate * neuron["deriv"][i]) + (neuron["last_delta"][i] * mom)
                neuron["weights"][i] += delta
                neuron["last_delta"][i] = delta
                neuron["deriv"][i] = 0.0
                # print(neuron["weights"])
                # print()




def test_network(network, test_data, num_inputs):
    print("Testing Network...")
    for i in range(len(test_data)):
        print("Input", i+1)
        # Grab test datum in random order
        datum = test_data[i]
        test_inputs = datum[0 : (num_inputs)]
        test_outputs = datum[num_inputs : len(datum)]

        forward_propagate(network, test_inputs)
        output_layer = network[-1]

        for j in range(len(output_layer)):
            output = output_layer[j]["output"]
            percent_difference = abs(output - test_outputs[j]) * 100
            print("\tExpected: ", test_outputs[j])
            print("\tActual:   ", output)
            print("\tPercent Diff: ", percent_difference)
            print()
        print()




def initialize_weights(num_weights):
    weights = []
    rand_min = 0
    rand_max = 0.5
    for i in range(num_weights):
        rand = get_random(rand_min, rand_max)
        weights.append(rand)
    return weights


def build_neuron(num_inputs):
    neuron = {}
    neuron["activation"] = None
    neuron["delta"] = None
    neuron["deriv"] = [0.0] * (num_inputs + 1)
    neuron["last_delta"] = [0.0] * (num_inputs + 1)
    neuron["output"] = None
    neuron["weights"] = initialize_weights(num_inputs + 1)
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
    num_outputs = num_inputs
    for i in range(num_outputs):
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
    print()




if __name__ == "__main__":
    print("QUANTUM NETWORK COMENSING")

    # Build Network
    my_num_inputs = 2
    my_hidden_layers = [5]
    my_network = build_network(my_hidden_layers, my_num_inputs)

    #Training and Test Data

    rawData = dataReader.openFile("TrainingData/Input.csv")

    my_data_density = 5
    my_training_data = dataReader.createTrainingData(rawData, my_data_density, 'not')

    # print(my_training_data)

    my_test_data = my_training_data

    # my_training_data = [
    #     [0, 1, 1, 0],
    #     [0.92, 0.3919183588, 0.3919183588, 0.92],
    #     [0.82, 0.5723635209, 0.5723635209, 0.82],
    #     [0.72, 0.6939740629, 0.6939740629, 0.72],
    #     [0.62, 0.7846018098, 0.7846018098, 0.62],
    #     [0.52, 0.8541662602, 0.8541662602, 0.52],
    #     [0.42, 0.9075241044, 0.9075241044, 0.42],
    #     [0.32, 0.9474175426, 0.9474175426, 0.32],
    #     [0.22, 0.9754998719, 0.9754998719, 0.22],
    #     [0.12, 0.9927738917, 0.9927738917, 0.12],
    #     [0.02, 0.99979998, 0.99979998, 0.02],
    #     [1, 0, 0, 1]
    # ]

    # TRAIN ANN
    my_num_iterations = 5000
    my_learning_rate = 0.3
    train_network(my_network, my_training_data, my_num_iterations, my_num_inputs, my_learning_rate)
    test_network(my_network, my_test_data, my_num_inputs)
