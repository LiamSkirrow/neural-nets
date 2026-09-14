import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import struct

# first create a configurable NN
NUM_NEURONS_PER_LAYER    = [784,   128,  64,   10]
#                           ^ input layer   
#                                  ^ first hidden layer
#                                        ^ second hidden layer
#                                              ^ output layer

NUM_NEURON_LAYERS    = len(NUM_NEURONS_PER_LAYER)    # probably shouldn't ever be < 1
NUM_WEIGHTS_MATRICES = NUM_NEURON_LAYERS-1
OUTPUT_LAYER_IDX     = len(NUM_NEURONS_PER_LAYER)-1

def init_matrices(weight_matrices, weight_matrix_corrections):
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
        weight_matrix_corrections.append(np.zeros((NUM_NEURONS_PER_LAYER[i+1], NUM_NEURONS_PER_LAYER[i])))
    
    return weight_matrices, weight_matrix_corrections

def init_neuron_list(neuron_list):
    # now populate the neuron list of lists
    for i in range(0, NUM_NEURON_LAYERS):
        neuron_layer = np.zeros(NUM_NEURONS_PER_LAYER[i])
        neuron_list.append(neuron_layer)
    
    return neuron_list

def init_bias_list(bias_list, bias_list_corrections):
    # and lastly define the bias list of lists, one bias value for each neuron, not including the input layer
    for i in range(1, NUM_NEURON_LAYERS):
        bias_layer = np.zeros(NUM_NEURONS_PER_LAYER[i])
        bias_list.append(bias_layer)
        bias_list_corrections.append(bias_layer)
    
    return bias_list, bias_list_corrections

# apply an activation function, ReLU is implemented here, operates on a np array
def apply_activation(neuron_array):
    # implement ReLU
    neuron_array = np.maximum(0, neuron_array)

    return neuron_array

def feedforward(image, weight_matrices, neuron_list, bias_list):
    # assign latest MNIST image to input layer neurons
    neuron_list[0] = image

    # apply the matrix * vector + bias operation, across all neuron layers
    for i in range(1, NUM_NEURON_LAYERS):
        # apply the formula W.X + b
        neuron_list[i] = weight_matrices[i-1] @ neuron_list[i-1] + bias_list[i-1]
        # apply the activation function
        neuron_list[i] = apply_activation(neuron_list[i])
    
    return neuron_list

# first derivative of the activation function ReLU
def dSigma_dx(z_val):
    return (z_val > 0).astype(float)


def backprop(weight_matrices, weight_matrix_corrections, neuron_list, bias_list, bias_list_corrections, golden_output):
    
    # iterate over NN neuron layers in reverse order, don't include input layer
    for idx in range(NUM_NEURON_LAYERS-1, 0, -1):
        # print(idx)
        weight_matrix_curr = weight_matrices[idx-1]
        neuron_list_prev   = neuron_list[idx-1]
        neuron_list_curr   = neuron_list[idx]
        bias_list_curr     = bias_list[idx-1]
        
        # iterate over all connecting weights and biases per neuron
        for n in range(0, NUM_NEURONS_PER_LAYER[i-1]):
            # w_jk * a_(L-1) + b(L-1)
            z_L   = weight_matrix_curr @ neuron_list_prev + bias_list_curr
            # 
            dc_db = dSigma_dx(z_L) * 2.0 * (neuron_list_curr - golden_output)
            # 
            dc_dw = dc_db * neuron_list_prev

            # reshape dc_dw into a matrix for convenient subtraction later on
            weight_corrections = dc_dw.reshape(NUM_NEURONS_PER_LAYER[n], NUM_NEURONS_PER_LAYER[n-1])
            bias_corrections   = dc_db

        weight_matrix_corrections[idx] = weight_corrections
        bias_corrections[idx]          = bias_corrections

    # TODO return the corrections as np arrays??? or concat together into one big vector?
    return weight_corrections, bias_corrections

# process the MNIST images
def train_on_mnist_images(weight_matrices, weight_matrix_corrections, neuron_list, bias_list, bias_list_corrections, path):

    training_dataset_images = path + "/train-images-idx3-ubyte/train-images-idx3-ubyte"
    training_dataset_labels = path + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte"

    with open(training_dataset_images, "rb") as image_file, \
         open(training_dataset_labels, "rb") as label_file:
        # image header
        magic      = struct.unpack(">I", image_file.read(4))[0]
        num_images = struct.unpack(">I", image_file.read(4))[0]
        rows       = struct.unpack(">I", image_file.read(4))[0]
        cols       = struct.unpack(">I", image_file.read(4))[0]
        # label header
        label_magic = struct.unpack(">I", label_file.read(4))[0]
        num_labels  = struct.unpack(">I", label_file.read(4))[0]
        
        assert num_images == num_labels

        print('MNIST training set details:')
        print('Magic number: ' + str(magic))
        print('Num Images: '   + str(num_images))

        # iterate over all existing images
        for i in range(0, num_images):
            # read the successive images, plot
            image_data = image_file.read(rows * cols)
            image = np.frombuffer(image_data, dtype=np.uint8)
            image = image.reshape(rows, cols)
            # sanity checking plots
            # plt.imshow(image, cmap="gray")
            # plt.show()

            # flatten + normalise the data (normalised 1D array, ready for input layer of MLP)
            image = image.reshape(784)
            image = image.astype(np.float32) / 255.0

            # infer the image through the NN
            feedforward(image, weight_matrices, neuron_list, bias_list)

            # TODO: note, all the below should be simple function calls to keep this function as practical as possible

            # derive the label from the label data set
            # -> figure out the correct label for this image and create a 'y' vector/nparray that looks like [0,0...1...,0]
            sample_label  = label_file.read(1)[0]
            golden_output = np.zeros(NUM_NEURONS_PER_LAYER[-1])
            golden_output[sample_label] = 1

            # obtain the MSE, going (a^L -y)^2
            # (neuron_list[-1] - y)^2
            mean_sq_err = (neuron_list[-1] - golden_output)**2

            # backpropagate and obtain the corrections to the weights and biases
            weight_matrix_corrections, bias_list_corrections = backprop(weight_matrices, weight_matrix_corrections, neuron_list, bias_list, bias_list_corrections, golden_output)

            # apply the corrections to the weights and biases (negative of gradient)
            # TODO

            # plot the MSE (is this the same as the loss???), check if it's converging to a small value
            # TODO

            if(i % 100 == 0):
                print('iteration: ' + str(i))
            
            # print(golden_output)
            # input()

        return weight_matrices, neuron_list, bias_list


if __name__ == '__main__':
    # create empty structures
    weight_matrices   = []
    neuron_list       = []
    bias_list         = []
    weight_matrix_corrections = []  # the corrections to be applied during backprop
    bias_list_corrections     = []  # ^ ^ ^

    weight_matrices, weight_matrix_corrections = init_matrices(weight_matrices, weight_matrix_corrections)
    neuron_list                        = init_neuron_list(neuron_list)
    bias_list, bias_list_corrections           = init_bias_list(bias_list, bias_list_corrections)

    # sanity check size of weights matrices
    print('showing dimensions of weights matrices...')
    for i in range(0, NUM_WEIGHTS_MATRICES):
        print(np.shape((weight_matrices[i])))

    # begin training...
    # - open MNIST dataset
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print("Path to dataset files:", path)
    weight_matrix_corrections, bias_list_corrections = train_on_mnist_images(weight_matrices, weight_matrix_corrections, neuron_list, bias_list, bias_list_corrections, path)

    # take the negative, get direction of steepest descent
    weight_matrix_corrections = -weight_matrix_corrections
    bias_list_corrections

    # now ready for inference...
    # either select individual images from validation set, or run whole set statistics...
    # TODO

    ### debug
    # print('Neuron output layer before feedforward')
    # print(neuron_list[3])
    # # now take a sample and feed it forward through the NN, input_sample must be a np.array()
    # neuron_list = feedforward(np.random.randn(784) * 0.01, weight_matrices, neuron_list, bias_list)
    # print('Neuron output layer after feedforward')
    # print(neuron_list[3])

