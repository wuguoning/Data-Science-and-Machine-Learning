# coding: utf-8
"""
Optimization Methods for Deep Learning.
Author:     Gordon Woo
Email:      wuguoning@gmail.com
Department: China University of Petroleum at Beijing
Date:       Nov.09, 2020
"""
import numpy as np

#--------------------------------------------------
class SGD2(object):
    """
    Stochastic Gradient Descent）
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads, batch_size):
        val = [w-self.lr*nw for w, nw in zip(params,grads)]

        return val


#--------------------------------------------------
class Momentum2(object):
    """
    Momentum is a method that helps accelerate
    SGD in the relevant direction and dampens
    oscillations.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads, batch_size):

        if np.all(self.v) is None:
            self.v = [np.zeros(w.shape) for w in params]

        self.v = [self.momentum*dv - self.lr/batch_size*dg for dv, dg in zip(self.v, grads)]
        val = [w+dv for w, dv in zip(params, self.v)]

        return val

#--------------------------------------------------
class Nesterov2(object):
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

    def update(self, params, grads, batch_size):
        if np.all(self.v) is None:
            self.v = [np.zeros(b.shape) for b in params]
        self.v = [self.momentum*dv for dv in self.v]
        self.v = [dv-self.lr/batch_size*dg for dv, dg in zip(self.v, grads)]
        val = [w+self.momentum*self.momentum*dv for w, dv in zip(params, self.v)]
        val = [w-(1+self.momentum)*self.lr/batch_size*dg for w, dg in zip(val, grads)]

        return val

#--------------------------------------------------
class AdaGrad2(object):
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

    def update(self, params, grads, batch_size):
        if np.all(self.h) == None:
            self.h = [np.zeros(b.shape) for b in params]
        self.h = [dh+dg*dg for dh, dg in zip(self.h, grads)]
        val = [w-self.lr/batch_size*dg/(np.sqrt(dh+1e-7)) \
               for w, dg, dh in zip(params, grads, self.h)]

        return val

#--------------------------------------------------
class RMSprop2(object):
    """
    RMSprop:
      RMSprop is an unpublished, adaptive learning rate
      method proposed by Geoff Hinton in his Coursera Class.
    """

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads, batch_size):

        if self.h == None:
            self.h = [np.zeros(b.shape) for b in params]
        self.h = [dh*self.decay_rate for dh in self.h]
        self.h = [dh+(1-self.decay_rate)*dg*dg for dh, dg in zip(self.h, grads)]
        val = [w-self.lr/batch_size*dg/np.sqrt(dh+1e7) \
               for w, dg, dh in zip(params, grads, self.h)]

        return val


#--------------------------------------------------
class Adam2(object):
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

    def update(self, params, grads, batch_size):
        if np.all(self.m) == None:
            self.m = [np.zeros(b.shape) for b in params]
            self.v = [np.zeros(b.shape) for b in params]
        self.iter += 1
        lr_t  = self.lr/batch_size * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        self.m = [dm+(1-self.beta1)*(dg-dm) for dm, dg in zip(self.m, grads)]
        self.v = [dv+(1-self.beta2)*(dg*dg-dv) for dv, dg in zip(self.v, grads)]
        val = [w-lr_t*dm/np.sqrt(dv+1e-7) for w, dm, dv in zip(params, self.m, self.v)]

        return val


#--------------------------------------------------
class AdaMax2(object):
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

    def update(self, params, grads, batch_size):

        if np.all(self.h) == None:
            self.h = [np.zeros(b.shape) for b in params]
            self.v = [np.zeros(b.shape) for b in params]
        self.h = [(self.beta1*dh + (1-self.beta1)*dg)/(1-self.gamma) for dh, dg in zip(self.h, grads) ]
        self.v = [np.maximum(self.beta2*dv, dg*dg) for dv, dg in zip(self.v, grads)]
        val = [w-self.lr/batch_size*dh/np.sqrt(dv) for w, dh, dv in zip(params, self.h, self.v)]

        return val


#--------------------------------------------------
class AdaDelta2(object):
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

    def update(self, params, grads, batch_size):
        if np.all(self.h) == None:
            self.h = [np.zeros(b.shape) for b in params]
            self.v = [np.zeros(b.shape) for b in params]
            self.m = [np.zeros(b.shape) for b in params]
        self.v = [self.gamma*dv+(1-self.gamma)*(dg*dg) for dv, dg in zip(self.v, grads)]
        self.h = [np.sqrt(dm+1e-7)*dg/np.sqrt(dv+1e-7) for dm, dg, dv in zip(self.m, grads, self.v)]
        self.m = [self.beta*dm+(1-self.beta)*(dh*dh) for dm, dh in zip(self.m, self.h)]
        val = [m-self.lr/batch_size*dh for m, dh in zip(params, self.h)]

        return val
