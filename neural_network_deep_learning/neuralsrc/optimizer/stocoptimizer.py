# coding: utf-8
"""
Optimization Methods for Deep Learning.
Author:    Gordon Woo
Email:      wuguoning@gmail.com
Department: China University of Petroleum at Beijing
Date:       Nov.09, 2020
"""
import numpy as np

#--------------------------------------------------
class SGD1(object):
    """
    Stochastic Gradient Descent）
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        val = params - self.lr * grads

        return val


#--------------------------------------------------
class Momentum1(object):
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

        if np.all(self.v) is None:
            self.v = np.zeros_like(params)
        self.v = self.momentum*self.v - self.lr*grads
        val = params + self.v

        return val

#--------------------------------------------------
class Nesterov1(object):
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
        if np.all(self.v) is None:
            self.v = np.zeros_like(params)
        self.v *= self.momentum
        self.v -= self.lr * grads
        val = params + self.momentum * self.momentum * self.v
        val = val - (1 + self.momentum) * self.lr * grads
        return val

#--------------------------------------------------
class AdaGrad1(object):
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
        if np.all(self.h) == None:
            self.h = np.zeros_like(params)
        self.h += grads * grads
        val = params - self.lr * grads / np.sqrt(self.h + 1e-7)
        return val

#--------------------------------------------------
class RMSprop1(object):
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

        if np.all(self.h) == None:
            self.h = np.zeros_like(params)
        self.h *= self.decay_rate
        self.h += (1 - self.decay_rate) * grads * grads
        val = params - self.lr * grads / np.sqrt(self.h + 1e-7)
        return val


#--------------------------------------------------
class Adam1(object):
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
        if np.all(self.m) == None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        self.m += (1 - self.beta1) * (grads - self.m)
        self.v += (1 - self.beta2) * (grads**2 - self.v)
        val = params - lr_t * self.m / np.sqrt(self.v + 1e-7)
        return val


#--------------------------------------------------
class AdaMax1(object):
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

        if np.all(self.h) == None:
            self.h = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.h = (self.beta1*self.h + (1 - self.beta1)*grads)/(1 - self.gamma)
        self.v =np.maximum(self.beta2*self.v, grads*grads)
        val = params - self.lr*self.h/np.sqrt(self.v)
        return val


#--------------------------------------------------
class AdaDelta1(object):
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
        if np.all(self.h) == None:
            self.h = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.m = np.zeros_like(params)
        self.v = self.gamma*self.v + (1 - self.gamma)*(grads*grads)
        self.h = np.sqrt(self.m + 1e-7)*grads/np.sqrt(self.v + 1e-7)
        self.m = self.beta*self.m + (1 - self.beta)*(self.h*self.h)
        val = params - self.lr*self.h
        return val

