import numpy as np
from neuron import Neuron

# first create a configurable NN
NUM_NEURONS_PER_LAYER    = [784,   16,   16,   10]            # needs tweaking!
#                           ^ input layer   
#                                  ^ first hidden layer
#                                        ^ second hidden layer
#                                              ^ output layer

NUM_NEURON_LAYERS    = len(NUM_NEURONS_PER_LAYER)    # probably shouldn't ever be < 1
NUM_WEIGHTS_MATRICES = NUM_NEURON_LAYERS-1
OUTPUT_LAYER_IDX     = len(NUM_NEURONS_PER_LAYER)-1

# create empty structures
weight_matrices = []
neuron_list = []
bias_list   = []

# every layer of neurons has its own matrix to represent the weights for that layer
# there are NUM_NEURON_LAYERS-1 many weights matrices. So if there are four neuron layers
# there are therefore 3 separate weights matrices...

# so for n weights matrices, which corresponds to n+1 neuron layers...
# WM0 -> NUM_NEURONS_PER_LAYER[1] x NUM_NEURONS_PER_LAYER[0]
# WM1 -> NUM_NEURONS_PER_LAYER[2] x NUM_NEURONS_PER_LAYER[1]
# ...
# WMn -> NUM_NEURONS_PER_LAYER[n+1] x NUM_NEURONS_PER_LAYER[n]

for i in range(0, NUM_WEIGHTS_MATRICES):
    weight_matrices.append(np.zeros((NUM_NEURONS_PER_LAYER[i+1], NUM_NEURONS_PER_LAYER[i])))

# sanity check size of weights matrices
print('showing dimensions of weights matrices...')
for i in range(0, NUM_WEIGHTS_MATRICES):
    print(np.shape((weight_matrices[i])))

# now populate the neuron list of lists
for i in range(0, NUM_NEURON_LAYERS):
    neuron_layer = []
    for j in range(0, NUM_NEURONS_PER_LAYER[i]):
        neuron_layer.append(Neuron())
    neuron_list.append(neuron_layer)

# and lastly define the bias list of lists, one bias value for each neuron, not including the input layer
for i in range(1, NUM_WEIGHTS_MATRICES):
    bias_layer = []
    for j in range(0, NUM_NEURONS_PER_LAYER[i]):
        bias_layer.append(Neuron())
    bias_list.append(bias_layer)
