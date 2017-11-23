import math
import random
# import matplotlib.pyplot as plt


def get_random(rand_min, rand_max):
    return rand_min + ((rand_max - rand_min) * random.random())


def train_network(network, training_data, iterations, num_inputs, learning_rate):
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
            outputs = network[-1][0]["outputs"]

            back_propagate_error(network, training_outputs)
            calculate_error_derivatives_for_weights(network, training_inputs)
        update_weights(network, learning_rate)



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
        alpha_sum = weights[-1][q][0] * bias[0]
        beta_sum = weights[-1][q][1] * bias[1]
        for i in range(len(inputs)):
            alpha_sum += weights[i][q][0] * inputs[i][q][0]
            beta_sum += weights[i][q][1] * inputs[i][q][1]
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
                total_delta = [0.0, 0.0]
                for q in range(len(expected_outputs)):
                    alpha_error_sum = 0.0
                    beta_error_sum = 0.0
                    for next_neuron in next_layer:
                        alpha_error_sum += next_neuron["weights"][j][q][0] * next_neuron["delta"][q][0]
                        beta_error_sum += next_neuron["weights"][j][q][1] * next_neuron["delta"][q][1]
                    alpha_delta = alpha_error_sum * transfer_derivative(neuron["outputs"][q][0])
                    beta_delta = beta_error_sum * transfer_derivative(neuron["outputs"][q][1])
                    neuron["delta"].append((alpha_delta, beta_delta))
                    # Accumulate to total delta
                    total_delta[0] += alpha_delta
                    total_delta[1] += beta_delta
                # Bia's Delta
                neuron["total_delta"] = total_delta
                # print(neuron["delta"])
                # print(neuron["total_delta"])
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

        # Calculate
        for neuron in layer:
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
            # print(neuron["deriv"])
            # print()


def update_weights(network, learning_rate, mom=0.8):
    for layer in network:
        for neuron in layer:
            for i in range(len(neuron["weights"])):
                for j in range(len(neuron["weights"][i])):
                    # handle bias vs. input weights
                    current_delta = neuron["last_total_delta"] if (j == (len(neuron["weights"][i]) - 1)) else neuron["last_delta"][i][j]
                    # adjust weights
                    alpha_delta = (learning_rate * neuron["deriv"][i][j][0]) + (current_delta[0] * mom)
                    beta_delta = (learning_rate * neuron["deriv"][i][j][1]) + (current_delta[1] * mom)
                    current_delta[0] = alpha_delta
                    current_delta[1] = beta_delta
                    neuron["weights"][i][j][0] += alpha_delta
                    neuron["weights"][i][j][1] += beta_delta
                    neuron["deriv"][i][j][0] = 0.0
                    neuron["deriv"][i][j][1] = 0.0
            # print(neuron["weights"])
            # print(neuron["last_total_delta"])
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
        outputs = network[-1][0]["outputs"]

        for q in range(len(outputs)):
            alpha_error = abs(outputs[q][0] - test_outputs[q][0]) * 100
            beta_error = abs(outputs[q][1] - test_outputs[q][1]) * 100
            print("Qubit", q+1)
            print("Alpha:")
            print("\tExpected:  ", test_outputs[q][0])
            print("\tActual:    ", outputs[q][0])
            print("\tError: ", alpha_error)
            print("Beta:")
            print("\tExpected:  ", test_outputs[q][1])
            print("\tActual:    ", outputs[q][1])
            print("\tError: ", beta_error)
            print()
        print()




def initialize_weights(num_weights, num_outputs):
    weights = []
    rand_min = 0
    rand_max = 0.5
    for i in range(num_weights):
        weight_set = []
        for j in range(num_outputs):
            randAlpha = get_random(rand_min, rand_max)
            randBeta = get_random(rand_min, rand_max)
            weight_set.append([randAlpha, randBeta])
        weights.append(weight_set)
    return weights


def build_neuron(num_inputs, num_outputs):
    neuron = {}
    neuron["activation"] = None
    neuron["delta"] = [] * num_outputs
    neuron["deriv"] = [[[0.0, 0.0]] * num_outputs] * (num_inputs + 1)
    neuron["last_delta"] = [[[0.0, 0.0]] * num_outputs] * (num_inputs + 1)
    neuron["outputs"] = None
    neuron["total_delta"] = None
    neuron["last_total_delta"] = [0.0, 0.0]
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
    print()




if __name__ == "__main__":
    print("QUANTUM NETWORK COMENSING")

    # Build Network
    my_num_inputs = 1
    my_hidden_layers = [2, 2]
    my_network = build_network(my_hidden_layers, my_num_inputs)
    print_network_topology(my_network)

    #Training and Test Data
    my_training_data = [
        [0, 1, 1, 0],
        [0.25, 0.75, 0.75, 0.25],
        [0.5, 0.5, 0.5, 0.5],
        [0.75, 0.25, 0.25, 0.75]
        [1, 0, 0, 1]
    ]
    my_test_data = my_training_data

    # TRAIN ANN
    my_num_iterations = 100
    my_learning_rate = 0.3
    train_network(my_network, my_training_data, my_num_iterations, my_num_inputs, my_learning_rate)
    test_network(my_network, my_test_data, my_num_inputs)
