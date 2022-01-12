# coding: utf-8
"""
Optimization Methods for Deep Learning.
Author:     齋騰康毅
Revised:    Gordon Woo
Email:      wuguoning@gmail.com
Department: China University of Petroleum at Beijing
Date:       Nov.09, 2020
"""
import numpy as np

#--------------------------------------------------
class SGD(object):
    """
    Stochastic Gradient Descent）
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


#--------------------------------------------------
class Momentum(object):
    """
    Momentum is a method that helps accelerate
    SGD in the relevant direction and dampens
    oscillations.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


#--------------------------------------------------
class Nesterov(object):
    """
    Nesterov Accelerated Gradient (NAG):
        is a way to give momentum term kind of
        prescience. We calculate the gradient not
        w.r.t to our current parameters \theta
        but w.r.t the approximate future position of
        our parameter.
    Reference:
        Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


#--------------------------------------------------
class AdaGrad(object):
    """
    Adadelta:
      Adadelta is an extension of Adagrad that seeks
      to reduce its aggressive, mononically decreasing
      learning rate. Instead of accumulating all past
      squared gradients, Adadelta restricts the window
      past gradients to some fixed size w.
    """

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


#--------------------------------------------------
class RMSprop(object):
    """
    RMSprop:
      RMSprop is an unpublished, adaptive learning rate
      method proposed by Geoff Hinton in his Coursera Class.
    """

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


#--------------------------------------------------
class Adam(object):
    """
    Adam:
      Adaptive Moment Estimation (Adam) is another method that
      computes adaptive learning rates for each parameters.
    Reference:
      Adam (http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

#--------------------------------------------------
class AdaMax(object):
    """
    AdaMax:
      AdaMax is a revised Adam method with p-norm
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, gamma = 0.02):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.h = None
        self.v = None

    def update(self, params, grads):
        if self.h == None:
            self.h, self.v = {}, {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] = (self.beta1*self.h[key] + (1 - self.beta1)*grads[key])/(1 - self.gamma)
            self.v[key] = max(self.beta2*self.v[key], grads[key]*grads[key])
            params[key] -= self.lr*self.h[key]/np.sqrt(self.v[key])

#--------------------------------------------------
class AdaDelta(object):
    """
    Adadelta:
     Adadelta is an extension of Adagrad that seeks
     to reduce its aggressive, mononically decreasing
     learning rate. Instead of accumulating all past
     squared gradients, Adadelta restricts the window
     past gradients to some fixed size w.
    """
    def __init__(self, lr=0.9, beta=0.1, gamma=0.97):
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.h = None
        self.v = None
        self.m = None

    def update(self, params, grads):
        if self.h == None:
            self.h, self.v, self.m  = {}, {}, {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                self.m[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.gamma*self.v[key] + (1 - self.gamma)*(grads[key]*grads[key])
            self.h[key] = np.sqrt(self.m[key] + 1e-7)*grads[key]/(np.sqrt(self.v[key] + 1e-7))
            self.m[key] = self.beta*self.m[key] + (1 - self.beta)*(self.h[key]*self.h[key])
            params[key] -= self.lr*self.h[key]

