"""
===========================================
Compare  optimization methods
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   Nov.18, 2020
China University of Petroleum at Beijing

===========================================
"""
# import modules
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import pandas as pd
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from functools import partial

# system path append
sys.path.append('../')

# import local modules
from neuralsrc.mnist_loader import MNIST_Loader
from neuralsrc.animation import MultipleAnimation
from neuralsrc.network import Network5
from neuralsrc.network import QuadraticCost5
from neuralsrc.network import CrossEntropyCost5


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
    dataset_dir = "../data1"
    from_file = dataset_dir + "/mnist.pkl"

    mnist_obj = MNIST_Loader(url_base, key_file, dataset_dir, from_file)
    training_data, validate_data, test_data = \
        mnist_obj.load_mnist(normalize=True, flatten=True, one_hot_label=True)

    print("Successfully loaded data from {}\n".format(from_file))
    # Test load data
    #print(len(training_data))
    #print(len(validate_data))
    #print(len(test_data))


    #====================================================
    # new network
    sizes = [784, 30, 10]
    net = Network5(sizes, cost=CrossEntropyCost5)
    net.large_weight_initializer()

    # save compare results to pandas
    #save_file = '../data1/cross_entr_loss_optimizers_SGD.csv'
    #save_file = '../data1/cross_entr_loss_optimizers_RMSprop.csv'
    #save_file = '../data1/cross_entr_loss_optimizers_Adam.csv'
    save_file = '../data1/cross_entr_loss_optimizers_AdeMax.csv'

    epoch = 50
    epoch_size = 10
    learning_rate = 0.95

    if os.path.exists(save_file):
        print('The file \'{}\' is exist!'.format(save_file))
        print('continue......')
    else:
        print("Begin training...\n")
        keys = ['eval_cost', 'eval_accu', 'train_cost', 'train_accu', \
                'test_cost', 'test_accu']
        eval_cost,  eval_accu, train_cost,\
        train_accu, test_cost, test_accu = \
        net.SGD(training_data, epoch, epoch_size, learning_rate, lmbda=0.0,\
                evaluation_data=validate_data, \
                monitor_evaluation_cost=True,\
                monitor_evaluation_accuracy=True,\
                monitor_training_cost=True,\
                monitor_training_accuracy=True,\
                test_data=test_data,\
                monitor_test_cost=True,\
                monitor_test_accuracy=True,
                #optimizer="Adam")
                #optimizer="RMSprop")
                optimizer="AdaMax")
        vals = [eval_cost,  eval_accu, train_cost,\
                train_accu, test_cost, test_accu]

        data = {}
        for key, val in zip(keys, vals):
            data[key] = val
        df = pd.DataFrame(data)
        df.to_csv(save_file)
        print("Successfully saved data to {}.".format(save_file))



    ##====================================================
    ### Animation
    #epoch_num = np.arange(epoch)
    #data1 = pd.read_csv(save_file)
    #paths1 = []
    #labels1 = ['test_accu','train_accu','eval_accu']
    #data_len = 10000
    #for key in labels1:
    #    paths1.append(np.vstack((epoch_num, data1[key]/data_len)))
    #fig1, ax1 = plt.subplots(figsize=(16,9))
    #ax1.set_xlabel('Epoch')
    #ax1.set_ylabel('Accuracy')
    #ax1.set_xlim([0,400])
    #ax1.set_ylim([0.6,1.2])
    #ax1.grid()

    #anim = MultipleAnimation(*paths1, labels=labels1, ax = ax1)
    #ax1.legend(loc='upper left')
    #HTML=(anim.to_jshtml())
    #plt.show()


