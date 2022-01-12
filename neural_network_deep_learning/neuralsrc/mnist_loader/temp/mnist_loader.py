"""
mnist loader

A library to load the MNIST image data.
"""

# Standard library
# This if for python 2.0s version
#import _pickle as cPickle
import  cPickle
import gzip
import sys

# This is for python 3.0s version
#import pickle
#import sys
#import gzip

sys.path.append('../../../data/')


# Third-party libraries
import numpy as np

def load_data():
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The "training_data" is returned as a tuple with two entries.
    The first entry with 50,000 entries. Each entry is, in turn,
    a numpy ndarray with 784=28*28 values, representing the 28*28
    pixels in a sigle MNIST image.

    The second entry in the "training_data" tuple is a numpy
    ndarray containing 50,000 entries. Those entries are just the
    digit values (0...9) for the corresponding images contained
    in the first entry of the tuple.

    The "validation_data" and "test_data" are similar, except each
    contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the "traning_data" a little.
    That's done in the wrapper function "load_data_wrapper()", see
    below.
    """
    # This is for the version python 2.0s
    f = gzip.open('../../../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    Return a tuple containing "(training_data, validation_data,
    test_data)". Based on "load_data", but the format is more
    convenient for use in our implementation of neural networks.

    In particular, "training_data" is a list containing 50,000
    2-tuples "(x,y)". "x" is a 784-dimensional numpy.ndarray
    containing the input image. "y" is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding
    to the correct digit for "x".

    "validation_data" and "test_data" are lists containing
    10,000 2-tuples "(x,y)". In each case, "x" is a 784-dimensional
    numpy.ndarray containing the input image, and "y" is the
    corresponding classification, i.e., the digital values (integers)
    corresponding to "x".

    Obviously, this means we're using slightly different formats for
    the training data and validation / test data. These formats
    turns out to be the most convenient for use in our neural
    network code.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784,1))
                      for x in tr_d[0]]
    training_results = [vectorized_result(y) for y
                       in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784,1))
                         for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a
    digit (0...9) into a corresponding desired output form
    the neural network.
    """
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def shared_dataset(data_xy):
    """
    Function that loads the dataset into shared variables.

    The reason we store our dataset in shared variable is to allow
    Theano to copy it into GPU (when code is run on GPU). Since
    copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy

# Main function
if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    print(len(training_data))
    print(len(validation_data))
    print(len(test_data))
