
# coding: utf-8

# # Back-Propagation
#
# ### Implementation: Boolean Problems Using Minmax Model

# In[1]:

import math
import random
import matplotlib.pyplot as plt


# In[2]:

def random_vector(minmax):
    vector = list()
    for i in range(len(minmax)):
        rand = minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * random.random())
        vector.append(rand)
    return vector


# In[3]:

def initialize_weights(num_weights):
    minmax = list()
    for i in range(num_weights):
        minmax.append([0, 0.5])
    return random_vector(minmax)


# ## Log-Sigmoid Activation
# #### Activate
# An activation value is calculated by summing each input value times it's corresponding weight (plus a bias value)
# #### Transfer
# Runs the activation value through the log-sigmoid function to determine the neuron's output
# ![image.png](attachment:image.png)

# #### Transfer Derivative
# Returns the derivative of the log-sigmoid function which is used in error calculation during back-propagation
# ![image.png](attachment:image.png)

# In[4]:

def activate(weights, vector, bias=1.0):
    # initialize sum with output's weight * bias
    _sum = weights[-1] * bias
    for i in range(len(vector)):
        _sum += weights[i] * vector[i]
    return _sum


# In[5]:

def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))


# In[6]:

def transfer_derivative(output):
    return output * (1.0 - output)


# ## Forward Porpagation
#
# ### Inputs
# * network: Neural Network
# * vector: The problem set (does not include the solution)
#
# ### Process
# Forward propagation passes the vector (problem set) through each layer of the network and returns the overall output
# * __Hidden Layers__
#     * Each Neuron "activates" the vector
#     * The activation's value is then evaluated to produce an output
#
# * __Output Layer__
#     * Collects the output from every neuron in the network
#     * Neuron "activates" the collected outputs
#     * Produces an overall output for the network

# In[7]:

def forward_propagate(network, vector):
    for i in range(len(network)):
        layer = network[i]
        _input = None

        # Hidden Layers
        if (i != (len(network) - 1)):
            _input = vector

        # Output Layer
        else:
            hidden_layer_outputs = list()
            previous_layer = network[i - 1]
            for k in range(len(previous_layer)):
                hidden_layer_outputs.append(previous_layer[k]["output"])
            _input = hidden_layer_outputs

        # Activation and Output
        for neuron in layer:
            neuron["activation"] = activate(neuron["weights"], _input)
            neuron["output"] = transfer(neuron["activation"])

    # Return the overall output
    return network[-1][0]["output"] # Assumes one node for output layer


# ## Backward Propagation
# Backward propagation calculates the level error in each neuron's output
# ### Inputs
# * network: Neural Network
# * expected_output: Known solution to the problem
#
# ### Process
# * Output Layer
#     * Calculate error for the network based on the known solution
#     * Set "delta" to the error times the derivative of the log-sigmoid output
# * Hidden Layers
#     * Looks at each neuron in the layer
#         * Calculates the error attributed to that neuron based on how it effected the next layer of the network
#         * Each neuron in the next layer's corresponding wheight and "delta" add to the error value
#         * Set "delta" to the error times the derivative of the log-sigmoid output

# In[8]:

def backward_propagate_error(network, expected_output):
    for i in range(len(network)):
        index = len(network) - 1 - i
        layer = network[index]

        # Output Layer
        if (index == (len(network) - 1)):
            neuron = layer[0] # assume one node in output layer
            error = (expected_output - neuron["output"])
            neuron["delta"] = error * transfer_derivative(neuron["output"])

        # Hidden Layers
        else:
            next_layer = network[index + 1]
            for j in range(len(layer)):
                err_sum = 0.0
                neuron = layer[j]
                for next_neuron in next_layer:
                    err_sum += next_neuron["weights"][j] * next_neuron["delta"]
                neuron["delta"] = err_sum * transfer_derivative(neuron["output"])


# ## Adjusting Weights
#

# In[9]:

def calculate_error_derivatives_for_weights(network, vector):
    for i in range(len(network)):
        layer = network[i]
        _input = None

        # Hidden Layers
        if (i != (len(network) - 1)):
            _input = vector

        # Output Layer
        else:
            hidden_layer_outputs = list()
            previous_layer = network[i - 1]
            for k in range(len(previous_layer)):
                hidden_layer_outputs.append(previous_layer[k]["output"])
            _input = hidden_layer_outputs

        # Calculate error derivative for weights
        for neuron in layer:
            signal = None
            for k in range(len(_input)):
                signal = _input[k]
                neuron["deriv"][k] += neuron["delta"] * signal
            # Bias's weight
            neuron["deriv"][-1] += neuron["delta"] * 1.0


# In[10]:

def update_weights(network, learning_rate, mom=0.8):
    for layer in network:
        for neuron in layer:
            for i in range(len(neuron["weights"])):
                delta = (learning_rate * neuron["deriv"][i]) + (neuron["last_delta"][i] * mom)
                neuron["weights"][i] += delta
                neuron["last_delta"][i] = delta
                neuron["deriv"][i] = 0.0


# ## Supervised Learning

# In[11]:

def train_network(network, training_data, num_inputs, iterations, learning_rate, benchmark):
    correct = 0
    for epoch in range(iterations):
        for pattern in training_data:
            vector = list()
            for k in range(num_inputs):
                vector.append(float(pattern[k]))
            expected = pattern[-1]
            output = forward_propagate(network, vector)
            if (round(output) == expected):
                correct += 1
            backward_propagate_error(network, expected)
            calculate_error_derivatives_for_weights(network, vector)
        update_weights(network, learning_rate)

        # Collect data throught iterations
        if (((epoch + 1) % benchmark) == 0):
            print("> epoch = " + str(epoch+1) + ", Correct = " + str(correct / (benchmark * len(training_data))))
            correct = 0


# ## Testing

# In[12]:

def test_network(network, domain, num_inputs):
    correct = 0
    for pattern in domain:
        input_vector = list()
        for i in range(num_inputs):
            input_vector.append(float(pattern[i]))
        output = forward_propagate(network, input_vector)
        if (round(output) == pattern[-1]):
            correct += 1
    print("Finished test with a score of " + str(correct / len(domain)))
    return correct


# ## Network Creation

# In[13]:

def create_neuron(num_inputs):
    neuron = {}
    neuron["weights"] = initialize_weights(num_inputs + 1)
    neuron["last_delta"] = [0.0] * (num_inputs + 1)
    neuron["deriv"] = [0.0] * (num_inputs + 1)
    return neuron


# In[14]:

def build_network(layer_pattern, num_inputs):
    network = []

    #Build each layer of the network
    for i in range(len(layer_pattern)):
        num_nodes = layer_pattern[i]
        layer = []
        if (i == 0):
            for j in range(num_nodes):
                layer.append(create_neuron(num_inputs))
        else:
            for j in range(num_nodes):
                layer.append(create_neuron(len(network[i-1])))
        network.append(layer)

    # Create Output Node
    network.append([create_neuron(len(network[-1]))])

    return network


# # Train and Testing a Network

# In[15]:

def execute(network, training_data, num_inputs, iterations, learning_rate, benchmark):
    print("Topology: inputs = " + str(num_inputs) + "  layers = " + str(len(network)))
    train_network(network, training_data, num_inputs, iterations, learning_rate, benchmark)
    test_network(network, training_data, num_inputs)
    return network


""" ------------------------------------------------------------------------------------------


            CODE IMPLEMENTATIONS


-------------------------------------------------------------------------------------------"""


# # XOR Problem
# ![image.png](attachment:image.png)

# In[36]:

if __name__ == "__main__":
    # problem configuration
    xor = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
    inputs = 2

    # algorithm configuration
    learning_rate = 0.3
    hidden_layer_pattern = [20, 10]
    iterations = 1000
    benchmark = 100

    # execute the algorithm
    network = build_network(hidden_layer_pattern, inputs)
    execute(network, xor, inputs, iterations, learning_rate, benchmark)


# # XOR Single-Layer

# In[23]:

if __name__ == "__main__":
    # problem configuration
    xor = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
    inputs = 2

    # algorithm configuration
    learning_rate = 0.3
    hidden_layer_pattern = [1]
    iterations = 10000
    benchmark = 1000

    # execute the algorithm
    network = build_network(hidden_layer_pattern, inputs)
    execute(network, xor, inputs, iterations, learning_rate, benchmark)


# # Basic Truth Table
# ![image.png](attachment:image.png)

# In[25]:

if __name__ == "__main__":
    # problem configuration
    orAndNot = [[1,1,1,0],[1,1,0,1],[1,0,1,0],[0,1,1,0],[1,0,0,1],[0,1,0,1],[0,0,1,0],[0,0,0,0]]
    inputs = 3

    # algorithm configuration
    learning_rate = 0.3
    hidden_layer_pattern = [2, 2]
    iterations = 100
    benchmark = 10

    # execute the algorithm
    network = build_network(hidden_layer_pattern, inputs)
    execute(network, orAndNot, inputs, iterations, learning_rate, benchmark)


# # Training on Partial Data

# In[27]:

if __name__ == "__main__":
    # problem configuration
    orAndNot = [[1,1,1,0],[1,1,0,1],[1,0,1,0],[0,1,1,0],[1,0,0,1]] # [0,1,0,1],[0,0,1,0],[0,0,0,0]
    inputs = 3

    # algorithm configuration
    learning_rate = 0.3
    hidden_layer_pattern = [2, 2]
    iterations = 100
    benchmark = 10

    # execute the algorithm
    network = build_network(hidden_layer_pattern, inputs)
    execute(network, orAndNot, inputs, iterations, learning_rate, benchmark)

    # test on data it was not trained on
    test_set = [[0,1,0,1],[0,0,1,0],[0,0,0,0]]
    test_network(network, test_set, inputs)


# # XOR Partial

# In[29]:

if __name__ == "__main__":
    # problem configuration
    xor = [[0,0,0], [0,1,1], [1,0,1]] # [1,1,0]
    inputs = 2

    # algorithm configuration
    learning_rate = 0.3
    hidden_layer_pattern = [2, 2]
    iterations = 1000
    benchmark = 100

    # execute the algorithm
    network = build_network(hidden_layer_pattern, inputs)
    execute(network, xor, inputs, iterations, learning_rate, benchmark)

     # test on data it was not trained on
    test_set = [[1,1,0]]
    test_network(network, test_set, inputs)


# # XOR Complex
# ![image.png](attachment:image.png)

# In[33]:

if __name__ == "__main__":
    # problem configuration
    xorComplex = [[1,1,1,1],[1,1,0,0],[1,0,1,1],[0,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    inputs = 3

    # algorithm configuration
    learning_rate = 0.3
    hidden_layer_pattern = [2, 2]
    iterations = 2000
    benchmark = 200

    # execute the algorithm
    network = build_network(hidden_layer_pattern, inputs)
    execute(network, xorComplex, inputs, iterations, learning_rate, benchmark)


# # XOR Complex 2
# ![image.png](attachment:image.png)

# In[35]:

if __name__ == "__main__":
    # problem configuration
    xorComplex2 = [[1,1,1,1],[1,1,0,0],[1,0,1,1],[0,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    inputs = 3

    # algorithm configuration
    learning_rate = 0.3
    hidden_layer_pattern = [2, 2]
    iterations = 1000
    benchmark = 100

    # execute the algorithm
    network = build_network(hidden_layer_pattern, inputs)
    execute(network, xorComplex2, inputs, iterations, learning_rate, benchmark)


# In[ ]:
