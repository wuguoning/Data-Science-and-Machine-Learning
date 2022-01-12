"""
===========================================
MNIST data load
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   Nov. 01, 2020
China University of Petroleum at Beijing

===========================================
"""
# import modules
import numpy as np
import matplotlib.pyplot as plt
import sys,os
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from functools import partial

# system path append
sys.path.append('../')

# import local modules
from neuralsrc.mnist_loader import MNIST_Loader
from neuralsrc.animation import SimpleAnimation
from neuralsrc.network import Network


if __name__ == "__main__":
    #===================================================
    # Process MNIST data

    # Load the MNIST dataset
    url_base = "http://yann.lecun.com/exdb/mnist/"
    key_file = {'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'}

    #dataset_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = "../data"
    save_file = dataset_dir + "/mnist.pkl"

    mnist_obj = MNIST_Loader(url_base, key_file, dataset_dir, save_file)
    training_data, validate_data, test_data = \
        mnist_obj.load_mnist(normalize=True, flatten=True, one_hot_label=True)

    # Test load data
    #print(np.shape(training_data[1][0]))
    #print(np.shape(training_data[1][1]))
    #print(len(training_data))


    #====================================================
    # new network
    sizes = [784, 100, 100, 100, 10]
    net = Network(sizes)

    # save test_data evaluate error
    save_file = '../data/test_h.txt'
    epoch = 50
    if os.path.exists(save_file):
        print('The file \'{}\' is exist!'.format(save_file))
        print('continue......')
    else:
        print("Preparing for training...\n")
        test_h = net.SGD(training_data, epoch, 10, 3.0, test_data)
        np.savetxt(save_file, test_h)


    #====================================================
    # Animation
    epoch_num = np.arange(epoch)
    test_accu = np.loadtxt(save_file)

    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim([0,55])
    ax.set_ylim([0.65,1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid()

    anim = SimpleAnimation(epoch_num, test_accu, label='Test Accuracy of SGD', ax = ax)
    ax.legend(loc='upper left')
    HTML=(anim.to_jshtml())
    plt.show()


