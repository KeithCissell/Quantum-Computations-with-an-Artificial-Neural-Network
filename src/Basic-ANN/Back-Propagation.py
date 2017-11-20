import math
import random
# import matplotlib.pyplot as plt


def get_random(rand_min, rand_max):
    return rand_min + ((rand_max - rand_min) * random.random())


def train_network(network, training_data, iterations, num_inputs):
    # print_network_topology(network)
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
            output = network[-1][0]["outputs"]

            back_propagate_error(network, training_outputs)
            calculate_error_derivatives_for_weights(network, training_inputs)



def forward_propagate(network, inputs):
    for i in range(len(network)):
        layer = network[i]
        layer_inputs = []

        # Set layer_inputs
        if (i == 0):
            layer_inputs.append(inputs)
        else:
            previous_layer = network[i - 1]
            for neuron in previous_layer:
                layer_inputs.append(neuron["outputs"])

        # Activate neurons
        for neuron in layer:
            neuron["activation"] = activate(neuron["weights"], layer_inputs)
            neuron["outputs"] = transfer(neuron["activation"])
            # print(neuron["activation"])
            # print(neuron["outputs"])
            # print()


def activate(weights, inputs, bias = (1.0,1.0)):
    # for each qubit
    num_qubits = len(inputs[0])
    activation_set = []
    for q in range(num_qubits):
        alpha_sum = weights[-1][q] * bias[0]
        beta_sum = weights[-1][q] * bias[1]
        for i in range(len(inputs)):
            alpha_sum += weights[i][q] * inputs[i][q][0]
            beta_sum += weights[i][q] * inputs[i][q][1]
        activation_set.append((alpha_sum, beta_sum))
    return activation_set



def transfer(activation):
    # Sigmoid
    transfer = []
    for qubit in activation:
        alpha = 1.0 / (1.0 + math.exp(-qubit[0]))
        beta = 1.0 / (1.0 + math.exp(-qubit[1]))
        transfer.append((alpha, beta))
    return transfer

def transfer_derivative(output):
    # Sigmoid derivative
    return output * (1.0 - output)


def back_propagate_error(network, expected_outputs):
    for i in range(len(network)):
        index = len(network) - 1 - i
        layer = network[index]

        # Output Layer
        if (index == len(network) - 1):
            neuron = layer[0]
            for q in range(len(expected_outputs)):
                alpha_error = expected_outputs[q][0] - neuron["outputs"][q][0]
                beta_error = expected_outputs[q][1] - neuron["outputs"][q][1]

                alpha_delta = alpha_error * transfer_derivative(neuron["outputs"][q][0])
                beta_delta = beta_error * transfer_derivative(neuron["outputs"][q][1])
                neuron["delta"].append((alpha_delta, beta_delta))
        else:
            next_layer = network[index + 1]
            for j in range(len(layer)):
                neuron = layer[j]
                for q in range(len(expected_outputs)):
                    alpha_error_sum = 0.0
                    beta_error_sum = 0.0
                    for next_neuron in next_layer:
                        alpha_error_sum += next_neuron["weights"][j][q] * next_neuron["delta"][q][0]
                        beta_error_sum += next_neuron["weights"][j][q] * next_neuron["delta"][q][1]
                    alpha_delta = alpha_error_sum * transfer_derivative(neuron["outputs"][q][0])
                    beta_delta = beta_error_sum * transfer_derivative(neuron["outputs"][q][1])
                    neuron["delta"].append((alpha_delta, beta_delta))
                # print(neuron["delta"])
                # print()


def calculate_error_derivatives_for_weights(network, inputs):
    for i in range(len(network)):
        layer = network[i]
        layer_inputs = []

        # First Hidden Layer
        if (i == 0):
            layer_inputs.append(inputs)
        else:
            previous_layer = network[i - 1]
            for neuron in previous_layer:
                layer_inputs.append(neuron["outputs"])

        #
        for neuron in layer:
            print(neuron["deriv"])
            for j in range(len(layer_inputs)):
                qubit_set = layer_inputs[j]
                for q in range(len(qubit_set)):
                    alpha_signal = qubit_set[q][0]
                    beta_signal = qubit_set[q][1]
                    neuron["deriv"][j][q][0] += neuron["delta"][q][0] * alpha_signal
                    neuron["deriv"][j][q][1] += neuron["delta"][q][1] * beta_signal
            # Bias's weight
            for q in range(len(layer_inputs[0])):
                neuron["deriv"][-1][q][0] += neuron["delta"][q][0] * 1.0
                neuron["deriv"][-1][q][1] += neuron["delta"][q][1] * 1.0
            print(neuron["deriv"])
            print()




def initialize_weights(num_weights, num_outputs):
    weights = []
    rand_min = 0
    rand_max = 0.5
    for i in range(num_weights):
        weight_set = []
        for j in range(num_outputs):
            rand = get_random(rand_min, rand_max)
            weight_set.append(rand)
        weights.append(weight_set)
    return weights


def build_neuron(num_inputs, num_outputs):
    neuron = {}
    neuron["activation"] = None
    neuron["delta"] = [] * num_outputs
    neuron["deriv"] = [[[0.0, 0.0]] * num_outputs] * (num_inputs + 1)
    neuron["last_delta"] = [[[0.0, 0.0]] * num_outputs] * (num_inputs + 1)
    neuron["outputs"] = None
    neuron["weights"] = initialize_weights(num_inputs + 1, num_outputs)
    return neuron


def build_network(hidden_layer_pattern, num_inputs):
    network = []

    # Add hidden layers
    num_layer_inputs = num_inputs
    num_outputs = num_inputs

    for i in range(len(hidden_layer_pattern)):
        layer = []
        num_neurons = hidden_layer_pattern[i]
        for j in range(num_neurons):
            neuron = build_neuron(num_layer_inputs, num_outputs)
            layer.append(neuron)
        network.append(layer)
        num_layer_inputs = num_neurons

    # Build Output Layer
    output_layer = []
    # for i in range(num_outputs):
    neuron = build_neuron(num_layer_inputs, num_outputs)
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

    # Build Network
    my_num_inputs = 1
    my_hidden_layers = [2, 2]
    my_network = build_network(my_hidden_layers, my_num_inputs)
    print_network_topology(my_network)

    #Training Data
    my_training_data = [
        [(0,1), (1,0)],
        [(1,0), (0,1)]
    ]

    # TRAIN ANN
    my_num_iterations = 1
    train_network(my_network, my_training_data, my_num_iterations, my_num_inputs)
