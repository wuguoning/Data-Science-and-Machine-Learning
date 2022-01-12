"""
===========================================
MNIST data load
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   Nov,01, 2020
China University of Petroleum at Beijing
===========================================
"""
import urllib.request
import gzip
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


class MNIST_Loader(object):

    def __init__(self, url_base, key_file, dataset_dir, save_file):
        """
        Parameters:
          self.url_base: url link
          self.key_file: file names using a dictionary
          self.dataset_dir: files dir
          self.save_file: saved file name
        """

        self.url_base = url_base
        self.key_file = key_file
        self.dataset_dir = dataset_dir
        self.save_file = save_file

    def _download(self, file_name):
        file_path = self.dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(self.url_base + file_name, file_path)
        print("Done")

    def download_mnist(self):
        for v in self.key_file.values():
           self._download(v)

    def _load_label(self, file_name):
        file_path = self.dataset_dir + "/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print("Done")

        return labels

    def _load_img(self, file_name):
        img_size=784
        file_path = self.dataset_dir + "/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, img_size)

        return data

    def _convert_numpy(self):
        dataset = {}
        dataset['train_img'] =  self._load_img(self.key_file['train_img'])
        dataset['train_label'] = self._load_label(self.key_file['train_label'])
        dataset['test_img'] = self._load_img(self.key_file['test_img'])
        dataset['test_label'] = self._load_label(self.key_file['test_label'])

        return dataset

    def init_mnist(self):
        self.download_mnist()
        dataset = self._convert_numpy()
        print("Creating pickle file ...")
        with open(self.save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("Done")

    def _vectorized_result(self,j):
        """
        Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere. This is used to convert a
        digit (0...9) into a corresponding desired output form
        the neural network.
        """
        e = np.zeros((10,1))
        e[j] = 1.0
        return e

    def load_mnist(self, normalize=True, flatten=True, one_hot_label=False):
        """
        Parameters
        ----------
        normalize : Normalize the pixel values
        flatten : Flatten the images as one array
        one_hot_label : Encode the labels as a one-hot array

        Returns
        -------
        (Trainig Image, Training Label), (Test Image, Test Label)
        """
        if not os.path.exists(self.save_file):
            self.init_mnist()

        with open(self.save_file, 'rb') as f:
            dataset = pickle.load(f)

        if normalize:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0

        if not flatten:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
        else:
            for key in ('train_img', 'test_img'):
                dataset[key] = [x.reshape(784,1) for x in dataset[key]]

        validate_data = list(zip(dataset['train_img'][50000:60000], dataset['train_label'][50000:60000]))
        test_data = list(zip(dataset['test_img'], dataset['test_label']))

        if one_hot_label:
              dataset['train_label'] = [self._vectorized_result(x) for x in dataset['train_label']]

        training_data = list(zip(dataset['train_img'][0:50000], dataset['train_label'][0:50000]))

        return training_data, validate_data, test_data
