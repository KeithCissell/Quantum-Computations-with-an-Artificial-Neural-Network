import math
import random

from DataRW import dataReader
import BackPropagation as BP



def optimize_ANN(data, iterations, pop_size, selection_rate, accuracy_weight, efficiency_weight, benchmark):
    population = init_population(pop_size)
    mutation_rate = 1.0

    for i in range(iterations):
        print("TESTING GENERATION: ", str(i+1), "/", str(iterations))
        print("Mutation Rate:", mutation_rate)
        for j in range(len(population)):
            # grab chromosome
            chromosome = population[j][0]
            print("Testing Chromosome: ", str(j+1), "/", str(len(population)))
            # build hidden layer pattern
            hidden_layers = [chromosome['hidden_layer1']['value']]
            # if (chromosome['hidden_layer1']['value'] != 0): hidden_layers.append(chromosome['hidden_layer1']['value'])
            # if (chromosome['hidden_layer2']['value'] != 0): hidden_layers.append(chromosome['hidden_layer2']['value'])
            # if (chromosome['hidden_layer3']['value'] != 0): hidden_layers.append(chromosome['hidden_layer3']['value'])
            # extract genes
            num_iterations = chromosome['iterations']['value']
            bias = chromosome['bias']['value']
            learning_rate = chromosome['learning_rate']['value']
            momentum = chromosome['momentum']['value']
            data_gap = chromosome['data_gap']['value']
            # Training and Test Data
            training_data = dataReader.createTrainingData(data, data_gap)
            test_data = dataReader.createTrainingData(data, 1) # test on all available daata
            num_inputs = int(round(len(test_data[0]) / 2))
            # calculate network efficiency
            layer_connections = 0
            layers = [num_inputs] + hidden_layers + [num_inputs]
            for k in range(1, len(layers)):
                layer_connections += layers[k-1] * layers[k]
            network_efficiency = 1 - (layer_connections / (num_inputs * 6 * num_inputs)) # inverse of: conections / highest connections possible
            population[j][2] = network_efficiency
            network_training_time = layer_connections * iterations * len(training_data)
            print("\tO(n) =", network_training_time)
            # Build - Train - Test Network
            network = BP.build_network(hidden_layers, num_inputs)
            BP.train_network(network, training_data, num_iterations, num_inputs, bias, learning_rate, momentum)
            network_accuracy = BP.test_network(network, test_data, num_inputs, bias, False)
            population[j][1] = network_accuracy
            # set score of chromosome
            score = (network_accuracy * accuracy_weight) + (network_efficiency * efficiency_weight)
            population[j][3] = score

        # print fittest chromosome at benchmark
        if (i % benchmark == 0):
            max_score = 0.0
            fittest_chromosome = None
            for j in range(len(population)):
                if (population[j][3] > max_score):
                    max_score = population[j][3]
                    fittest_chromosome = population[j]
            print("Fittest of Iteration", i+1)
            print()
            print_chromosome(fittest_chromosome[0])
            print()
            print("NETWORK ACCURACY:", fittest_chromosome[1])
            print("NETWORK EFFICIENCY:", fittest_chromosome[2])
            print("CHROMOSOME SCORE:", fittest_chromosome[3])
            print()

        # mutate population
        if (i+1 != len(population)):
            population = mutate_population(population, mutation_rate, selection_rate)
            mutation_x = (((i / iterations) * 10) + 1.0) # iteration falling in the range from [1, 11]
            mutation_rate = 1 / mutation_x # adjust 1 / x function
            if (mutation_x > 6): mutation_rate -= 0.08


def mutate_population(population, mutation_rate, selection_rate):
    num_parents = round(len(population) * selection_rate)
    if (num_parents < 1): num_parents = 1
    num_children = len(population) - num_parents
    parent_chromosomes = roulette_selection(population, num_parents)
    child_chromosomes = []
    for i in range(num_children):
        parent_index = i
        while (parent_index > (num_parents - 1)):
            parent_index -= (num_parents)
        parent_clone = clone_chromosome(parent_chromosomes[parent_index][0])
        new_chromosome = mutate_chromosome(parent_clone, mutation_rate)
        new_child = [new_chromosome, 0, 0, 0]
        child_chromosomes.append(new_child)
    new_population = parent_chromosomes + child_chromosomes
    return new_population


def roulette_selection(population, num_parents, elites=1):
    parents = []
    for i in range(elites):
        fittest_score = 0.0
        fittest_index = 0
        for j in range(len(population)):
            if (population[j][3] > fittest_score):
                fittest_score = population[j][3]
                fittest_index = j
        parent_chromosome = clone_chromosome(population[fittest_index][0])
        new_parent = [parent_chromosome, 0, 0, 0]
        parents.append(new_parent)
        del population[fittest_index]
    for i in range(num_parents - elites):
        fitness_sum = 0
        for chromosome in population:
            fitness_sum += chromosome[3]
        selection = random.uniform(0.0, fitness_sum)
        selection_sum = 0
        for j in range(len(population)):
            selection_sum += population[j][3]
            if (selection_sum > selection):
                parent_chromosome = clone_chromosome(population[j][0])
                new_parent = [parent_chromosome, 0, 0, 0]
                parents.append(new_parent)
                del population[j]
                break
    return parents



def mutate_chromosome(chromosome, mutation_rate):
    for key in chromosome:
        # extract gene info
        gene = chromosome[key]
        gValue = gene['value']
        gMin = gene['min']
        gMax = gene['max']
        gInt = gene['int']
        gRange = gMax - gMin
        # set mutation bounds
        searchMin = gValue - (gRange * mutation_rate)
        if (searchMin < gMin): searchMin = gMin
        searchMax = gValue + (gRange * mutation_rate)
        if (searchMax > gMax): searchMax = gMax
        # generate new val
        new_value = 0
        if (gInt):
            new_value = round(random.randint(math.floor(searchMin), math.ceil(searchMax)))
        else:
            adjustMin = round(searchMin * 10000)
            adjustMax = round(searchMax * 10000)
            new_value = random.randint(adjustMin, adjustMax) / 10000
        chromosome[key]['value'] = new_value
    return chromosome


def init_population(pop_size, guide=False):
    population = []
    if (guide): population.append([init_chromosome(), 0, 0, 0])
    for i in range(pop_size):
        new_chomosome = mutate_chromosome(init_chromosome(), 1.0)
        population.append([new_chomosome, 0, 0, 0])
    return population


def init_chromosome():
    chromosome = {}
    chromosome['hidden_layer1'] = {'value': 3, 'min': 2, 'max': 6, 'int': True}
    # chromosome['hidden_layer2'] = {'value': 3, 'min': 0, 'max': 6, 'int': True}
    # chromosome['hidden_layer3'] = {'value': 3, 'min': 0, 'max': 6, 'int': True}
    chromosome['iterations'] = {'value': 500, 'min': 50, 'max': 2000, 'int': True}
    chromosome['bias'] = {'value': 1.0, 'min': 0.1, 'max': 1.0, 'int': False}
    chromosome['learning_rate'] = {'value': 0.3, 'min': 0.1, 'max': 1.0, 'int': False}
    chromosome['momentum'] = {'value': 0.8, 'min': 0.1, 'max': 1.0, 'int': False}
    chromosome['data_gap'] = {'value': 50, 'min': 1, 'max': 100, 'int': True}
    return chromosome


def clone_chromosome(old_chromosome):
    new_chromosome = init_chromosome()
    for gene in new_chromosome:
        for value in new_chromosome[gene]:
            new_chromosome[gene][value] = old_chromosome[gene][value]
    return new_chromosome


def print_chromosome(chromosome):
    print("CHROMOSOME TOPOLOGY")
    print("Hidden Layer 1:", chromosome['hidden_layer1']['value'])
    # print("Hidden Layer 2:", chromosome['hidden_layer2']['value'])
    # print("Hidden Layer 3:", chromosome['hidden_layer3']['value'])
    print("Iterations:", chromosome['iterations']['value'])
    print("Bis:", chromosome['bias']['value'])
    print("Learning Rate:", chromosome['learning_rate']['value'])
    print("Momentum:", chromosome['momentum']['value'])
    print("Data Gap:", chromosome['data_gap']['value'])


if __name__ == "__main__":
    print("QUANTUM NETWORK OPTIMIZATION COMENSING")
    print()

    # Data Sets
    notGateData = dataReader.openFile("TrainingData/NOTGate.csv")
    hadamardGateData = dataReader.openFile("TrainingData/HADAMARDGate.csv")
    phaseFlipGateData = dataReader.openFile("TrainingData/PFLIPGate.csv")
    swapGateData = dataReader.openFile("TrainingData/SWAPGate.csv")
    cNotGateData = dataReader.openFile("TrainingData/CNOTGate.csv")
    cHadamardGateData = dataReader.openFile("TrainingData/CHADAMARDGate.csv")
    cPhaseFlipGateData = dataReader.openFile("TrainingData/CPFLIPGate.csv")
    toffolinGateData = dataReader.openFile("TrainingData/TOFFOLIGate.csv") #aka CCNOT Gate
    fredkinGateData = dataReader.openFile("TrainingData/FREDKINGate.csv") #aka CSWAP Gate

    # set GA params
    my_data = fredkinGateData
    my_iterations = 100
    my_pop_size = 20
    my_accuracy_weight = 1.0
    my_efficiency_weight = 0.0
    my_selection_rate = 0.5
    my_benchmark = 1

    # Build - Train - Test Network
    optimize_ANN(my_data, my_iterations, my_pop_size, my_selection_rate, my_accuracy_weight, my_efficiency_weight, my_benchmark)
