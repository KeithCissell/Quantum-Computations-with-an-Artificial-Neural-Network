import math
import random

import dataReader

# import matplotlib.pyplot as plt


def get_random(rand_min, rand_max):
    return rand_min + ((rand_max - rand_min) * random.random())


# Translates range from [-1,1] to [0,1]
def shrink_range(values):
    translation = []
    for v in values:
        translation.append((v / 2) + 0.5)
    return translation


def train_network(network, training_data, iterations, num_inputs, bias, learning_rate, momentum):
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
            # transform data range [-1,1] --> [0,1]
            translated_datum = shrink_range(datum)

            training_inputs = translated_datum[0 : (num_inputs)]
            training_outputs = translated_datum[num_inputs : len(translated_datum)]

            forward_propagate(network, training_inputs, bias)
            output_layer = network[-1]

            back_propagate_error(network, training_outputs)
            calculate_error_derivatives_for_weights(network, training_inputs)
        update_weights(network, learning_rate, momentum)



def forward_propagate(network, inputs, bias):
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
            neuron["activation"] = activate(neuron["weights"], layer_inputs, bias)
            neuron["output"] = transfer(neuron["activation"])
            # print(neuron["activation"])
            # print(neuron["output"])
            # print()


def activate(weights, inputs, bias):
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


def update_weights(network, learning_rate, momentum):
    for layer in network:
        for neuron in layer:
            for i in range(len(neuron["weights"])):
                delta = (learning_rate * neuron["deriv"][i]) + (neuron["last_delta"][i] * momentum)
                neuron["weights"][i] += delta
                neuron["last_delta"][i] = delta
                neuron["deriv"][i] = 0.0
                # print(neuron["weights"])
                # print()




def test_network(network, test_data, num_inputs, bias, print_out):
    percent_difference_sum = 0
    count = 0
    if (print_out): print("Testing Network...")
    for i in range(len(test_data)):
        if (print_out): print("Input", i+1)
        # Grab test datum in random order
        datum = test_data[i]
        translated_datum = shrink_range(datum)

        test_inputs = translated_datum[0 : (num_inputs)]
        test_outputs = translated_datum[num_inputs : len(translated_datum)]

        forward_propagate(network, test_inputs, bias)
        output_layer = network[-1]

        for j in range(len(output_layer)):
            output = output_layer[j]["output"]
            percent_difference = abs(output - test_outputs[j]) * 100
            percent_difference_sum += percent_difference
            count += 1
            if (print_out):
                print("\tExpected: ", test_outputs[j])
                print("\tActual:   ", output)
                print("\tPercent Diff: ", percent_difference)
                print()
        if (print_out): print()
    average_accuracy = 100 - (percent_difference_sum / count)
    return average_accuracy




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

    # Network Training Inputs
    my_hidden_layers = [5] # Gene
    my_num_iterations = 1500 # Gene
    my_bias  = 1.0 # Gene
    my_learning_rate = 0.3 # Gene
    my_momentum = 0.8 # Gene
    my_data_gap = 50 # Gene

    # Data Sets
    notGateData = dataReader.openFile("TrainingData/NOTGate.csv")
    hadamardGateData = dataReader.openFile("TrainingData/HADAMARDGate.csv")

    # Training and Test Data
    my_training_data = dataReader.createTrainingData(hadamardGateData, my_data_gap)
    my_test_data = dataReader.createTrainingData(hadamardGateData, 1) # test on all available daata
    my_num_inputs = int(round(len(my_test_data[0]) / 2))

    # Build - Train - Test Network
    my_network = build_network(my_hidden_layers, my_num_inputs)
    train_network(my_network, my_training_data, my_num_iterations, my_num_inputs, my_bias, my_learning_rate, my_momentum)
    my_network_accuracy = test_network(my_network, my_test_data, my_num_inputs, my_bias, False)
    print("Average Network Accuracy:", my_network_accuracy, "%")
