# Neural Nets
This repo is my attempt at solidifying my own understanding of how neural networks actually operate at a low level of abstraction, basically ensuring I actually understand the math all the way down, for both inference and training.

### Classic Neural Net / Multi-Layer Perceptron
dir: `./mlp`

A vanilla NN to classify handwritten digits, based on the classic training set published by MNIST. Written in Python with help only from numpy for making the math a bit easier to implement accurately. Configurable number/dimension of neuron layers, backpropagation by hand. 

Complete for now, pretty consistently gets a correctness score of about 95-96%, although if you rerun it a few times it's easy to make it unstable. This is probably due to the randomisation of the initial weights just being the wrong choice. When this happens it'll score more like in the 70% range...

### Convolutional Neural Net (TODO)
dir: `./cnn`
An as-simple-as-possible example of a CNN, using either the Fashion-MNIST dataset or the CIFAR-10 car classification dataset.
