import numpy as np
# from neuron import Neuron

# first create a configurable NN
NUM_NEURONS_PER_LAYER    = [784,   128,  64,   10]
#                           ^ input layer   
#                                  ^ first hidden layer
#                                        ^ second hidden layer
#                                              ^ output layer

NUM_NEURON_LAYERS    = len(NUM_NEURONS_PER_LAYER)    # probably shouldn't ever be < 1
NUM_WEIGHTS_MATRICES = NUM_NEURON_LAYERS-1
OUTPUT_LAYER_IDX     = len(NUM_NEURONS_PER_LAYER)-1

def init_matrices(weight_matrices):
    # every layer of neurons has its own matrix to represent the weights for that layer
    # there are NUM_NEURON_LAYERS-1 many weights matrices. So if there are four neuron layers
    # there are therefore 3 separate weights matrices...

    # so for n weights matrices, which corresponds to n+1 neuron layers...
    # WM0 -> NUM_NEURONS_PER_LAYER[1] x NUM_NEURONS_PER_LAYER[0]
    # WM1 -> NUM_NEURONS_PER_LAYER[2] x NUM_NEURONS_PER_LAYER[1]
    # ...
    # WMn -> NUM_NEURONS_PER_LAYER[n+1] x NUM_NEURONS_PER_LAYER[n]

    for i in range(0, NUM_WEIGHTS_MATRICES):
        # random initialisation around 0
        weight_matrices.append(np.random.randn(NUM_NEURONS_PER_LAYER[i+1], NUM_NEURONS_PER_LAYER[i]) * 0.01)
    
    return weight_matrices

def init_neuron_list(neuron_list):
    # now populate the neuron list of lists
    for i in range(0, NUM_NEURON_LAYERS):
        neuron_layer = np.zeros(NUM_NEURONS_PER_LAYER[i])
        neuron_list.append(neuron_layer)
    
    return neuron_list

def init_bias_list(bias_list):
    # and lastly define the bias list of lists, one bias value for each neuron, not including the input layer
    for i in range(1, NUM_NEURON_LAYERS):
        bias_layer = np.zeros(NUM_NEURONS_PER_LAYER[i])
        bias_list.append(bias_layer)
    
    return bias_list

def feedforward(input_sample, weight_matrices, neuron_list, bias_list):
    # assign input_sample to input layer neurons
    neuron_list[0] = input_sample

    # apply the matrix * vector + bias operation, across all neuron layers
    for i in range(1, NUM_NEURON_LAYERS):
        neuron_list[i] = weight_matrices[i-1] @ neuron_list[i-1] + bias_list[i-1]
        # apply the activation function
        # TODO
    
    return neuron_list

if __name__ == '__main__':
    # create empty structures
    weight_matrices = []
    neuron_list     = []
    bias_list       = []

    weight_matrices = init_matrices(weight_matrices)
    neuron_list     = init_neuron_list(neuron_list)
    bias_list       = init_bias_list(bias_list)

    # print(weight_matrices)
    # print(neuron_list)
    # print(bias_list)

    # sanity check size of weights matrices
    print('showing dimensions of weights matrices...')
    for i in range(0, NUM_WEIGHTS_MATRICES):
        print(np.shape((weight_matrices[i])))

    # now to train the net...
    # - read in one training sample input image
    # - run feedforward()
    # ?- calcuate loss function
    # ?- run gradient descent
    # ???
    # - loop back to top, read in next image, repeat for whole training set


    ### debug
    # print('Neuron output layer before feedforward')
    # print(neuron_list[3])
    # # now take a sample and feed it forward through the NN, input_sample must be a np.array()
    # neuron_list = feedforward(np.random.randn(784) * 0.01, weight_matrices, neuron_list, bias_list)
    # print('Neuron output layer after feedforward')
    # print(neuron_list[3])
