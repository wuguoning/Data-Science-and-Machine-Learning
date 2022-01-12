"""
===========================================
Generalized Stochastic Gradient Method and its Variants
Implementation with Tensorflow
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   Oct,28, 2020
China University of Petroleum at Beijing

===========================================
Reference:
    1. NiranjanKumar, Implementing different variants of Gradient
       Descent Optimization Algorithm in Python using Numpy
    2. SEBASTIAN RUDER, An overview of gradient descent
       optimization algorithms
    3. https://stackoverflow.com/questions/55552715/tensorflow-2-0-minimize-a-simple-function

===========================================
"""

# Import modules
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

class General_Stoch_Grad_Tensor(object):
    """
    Stochastic method based methods implementation
    with tensorflow
    """
    def __init__(self, w_init, b_init):
        """
        Parameters:
          w_init: w initial value
          b_init: b initial value
          self.w: tensorflow vairable
          self.b: tensorflow variable
          self.w_h: history of self.w
          self.b-h: history of self.b
        """
        self.w = tf.Variable(w_init, name='w', trainable=True, dtype=tf.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True, dtype=tf.float32)
        self.w_h = []
        self.b_h = []

    def Bealefun(self):
        return (1.5 - self.w + self.w*self.b)**2 + (2.25 - self.w + self.w*self.b**2)**2 + (2.625 - self.w + self.w*self.b**3)**2

    def fit(self, funname, algo, learning_rate = 0.1, tol=1E-8, max_iter=5000):
        """
        Parameters:
            tol: tolerance
            max_iter: max interation numbers
        """
        if algo == 'Adam':
            opt = tf.optimizers.Adam(learning_rate)
        elif algo == 'SGD':
            opt = tf.optimizers.SGD(learning_rate)
        elif algo == 'Adagrad':
            opt = tf.optimizers.Adagrad(learning_rate)
        elif algo == 'AdaDelta':
            opt = tf.optimizers.Adadelta(learning_rate)
        elif algo == 'RMSprop':
            opt = tf.optimizers.RMSprop(learning_rate)
        elif algo == 'AdaMax':
            opt = tf.optimizers.Adamax(learning_rate)


        err = np.Inf
        i = 0
        while err > tol:
            w_old = self.w.numpy()
            b_old = self.b.numpy()
            self.w_h.append(w_old)
            self.b_h.append(b_old)
            if funname == "Bealefun":
                train = opt.minimize(self.Bealefun, var_list=[self.w,self.b])
            err = np.abs(self.w - w_old) + np.abs(self.b - b_old)
            i = i + 1
            if i > max_iter:
                print(f'Stopping at max_iter={max_iter}')
                return
