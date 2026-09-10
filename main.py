import numpy as np
from neuron import Neuron

# first create a configurable NN
NUM_INPUT_SAMPLES  = 784
NUM_HIDDEN_LAYERS  = 2 # probably shouldn't ever be < 1
NUM_OUTPUT_SAMPLES = 10
HIDDEN_LAYER_NUM_NEURONS = [10, 8] # needs thinking
weight_matrix_hidden_layers = []

# every layer of neurons has its own matrix to represent the weights for that layer

# the input layer weights matrix has num_cols equal to NUM_INPUT_SAMPLES
# the input layer weights matrix has num_rows equal to HIDDEN_LAYER_NUM_NEURONS[0]
# -> HIDDEN_LAYER_NUM_NEURONS[0] x NUM_INPUT_SAMPLES
weight_matrix_input_layer = np.zeros((HIDDEN_LAYER_NUM_NEURONS[0], NUM_INPUT_SAMPLES))

# the first hidden layer weights matrix has num_cols equal to HIDDEN_LAYER_NUM_NEURONS[0]
# the first hidden layer weights matrix has num_rows equal to HIDDEN_LAYER_NUM_NEURONS[1]
# -> HIDDEN_LAYER_NUM_NEURONS[1] x HIDDEN_LAYER_NUM_NEURONS[0]
for i in range(0, NUM_HIDDEN_LAYERS-1):
    weight_matrix_hidden_layers.append(np.zeros((HIDDEN_LAYER_NUM_NEURONS[i+1], HIDDEN_LAYER_NUM_NEURONS[i])))

# the output layer weights matrix has num_cols equal to HIDDEN_LAYER_NUM_NEURONS[1]
# the output layer weights matrix has num_rows equal to NUM_OUTPUT_SAMPLES
weight_matrix_output_layer = np.zeros((NUM_OUTPUT_SAMPLES, HIDDEN_LAYER_NUM_NEURONS[-1]))

# check size of matrices
print('input layer matrix shape')
print(np.shape((weight_matrix_input_layer)))
print('')

print('hidden layer matrix shapes')
for i in range(0, NUM_HIDDEN_LAYERS-1):
    print(np.shape((weight_matrix_hidden_layers[i])))
    print('')

print('output layer matrix shape')
print(np.shape((weight_matrix_output_layer)))
print('')