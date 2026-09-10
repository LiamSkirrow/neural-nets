import numpy as np

# first create a configurable NN
NUM_INPUT_SAMPLES  = 784
NUM_HIDDEN_LAYERS  = 2
NUM_OUTPUT_SAMPLES = 10
HIDDEN_LAYER_NUM_NEURONS = [10, 10] # needs thinking

# every neuron has its own weights matrix

# the first hidden layer weights matrix has num_cols equal to NUM_INPUT_SAMPLES
# the first hidden layer weights matrix has num_rows equal to HIDDEN_LAYER_NUM_NEURONS[0]

# the second hidden layer weights matrix has num_cols equal to HIDDEN_LAYER_NUM_NEURONS[0]
# the second hidden layer weights matrix has num_rows equal to HIDDEN_LAYER_NUM_NEURONS[1]

# the third hidden layer weights matrix has num_cols equal to HIDDEN_LAYER_NUM_NEURONS[1]
# the third hidden layer weights matrix has num_rows equal to NUM_OUTPUT_SAMPLES

