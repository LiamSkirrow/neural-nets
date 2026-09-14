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
learning_rate        = 0.01

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
    delta = None

    # Work backwards: output layer -> first hidden layer
    for idx in range(NUM_NEURON_LAYERS - 1, 0, -1):
        W = weight_matrices[idx - 1]
        a_prev = neuron_list[idx - 1]
        a_curr = neuron_list[idx]
        b = bias_list[idx - 1]

        # Recalculate pre-activation value
        z = W @ a_prev + b

        if(idx == NUM_NEURON_LAYERS-1):
            # Output layer:
            # dC/dz = dC/da * da/dz
            delta = (2.0 * (a_curr - golden_output)* dSigma_dx(z))
        else:
            # Hidden layer:
            # propagate next layer's error backwards
            W_next = weight_matrices[idx]
            delta = ((W_next.T @ delta)* dSigma_dx(z))
        # dC/dW
        weight_matrix_corrections[idx-1] = np.outer(delta, a_prev)
        # dC/db
        bias_list_corrections[idx-1] = delta.copy()

    return weight_matrix_corrections, bias_list_corrections

# process the MNIST images
def train_on_mnist_images(weight_matrices, weight_matrix_corrections, neuron_list, bias_list, bias_list_corrections, path):

    training_dataset_images = path + "/train-images-idx3-ubyte/train-images-idx3-ubyte"
    training_dataset_labels = path + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte"

    # plot the MSE as we go! Hopefully it's converging?
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    mse_history = []

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
            image = image.reshape(rows * cols)
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

            # print('matrix dims')
            # for mat in weight_matrix_corrections:
            #     print(np.shape(mat))
            # print('bias dims')
            # for vec in bias_list_corrections:
            #     print(np.shape(vec))

            # apply the corrections to the weights and biases (negative of gradient)
            # take the negative, get direction of steepest descent
            for wm, wm_corr in zip(weight_matrices, weight_matrix_corrections):
                wm += learning_rate * wm_corr * -1.0
            for bl, bl_corr in zip(bias_list, bias_list_corrections):
                bl += learning_rate * bl_corr * -1.0

            if(i % 5000 == 0):
                # plot the MSE (is this the same as the loss???), check if it's converging to a small value
                print('iteration: ' + str(i))
                mse_history.append(sum(mean_sq_err))
                # Update plot
                line.set_data(range(len(mse_history)), mse_history)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)
                print('MSE:       ' + str(sum(mean_sq_err)))
            
            # print(golden_output)
            # input()

        plt.ioff()
        plt.show()

        return weight_matrices, bias_list


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
    # open MNIST dataset
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print("Path to dataset files:", path)
    weight_matrices, bias_list = train_on_mnist_images(weight_matrices, weight_matrix_corrections, neuron_list, bias_list, bias_list_corrections, path)

    # TODO
    # don't I need to do softmax() or something???

    # now ready for inference...
    # either select individual images from validation set, or run whole set statistics...
    # TODO
