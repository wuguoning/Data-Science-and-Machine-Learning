"""
===========================================
Generalized Stochastic Gradient Method and its Variants
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   Oct,20, 2020
China University of Petroleum at Beijing

===========================================
Reference:
    1. NiranjanKumar, Implementing different variants of Gradient
       Descent Optimization Algorithm in Python using Numpy
    2. SEBASTIAN RUDER, An overview of gradient descent
       optimization algorithms
===========================================
"""

# Import modules
import numpy as np
import random

from autograd import elementwise_grad, value_and_grad

class General_Stochastic_Gradient_And_Variants(object):

    def __init__(self, w_init, b_init, fun):
        """
        Parameters:
            self.w: weight init
            self.b: bias init
            self.w_h: weight history
            self.b_h: bias history
            self.fun: the function
        """

        self.w = w_init
        self.b = b_init
        self.w_h = []
        self.b_h = []
        self.fun = fun
        self.X = None
        self.Y = None


    # gradient of the loss function about parameter w
    def grad_w(self, x, y):

        return elementwise_grad(self.fun, argnum=0)(x,y)


    # gradient of the loss function about parameter b
    def grad_b(self, x, y):

        return elementwise_grad(self.fun, argnum=1)(x,y)


    def fit(self, X, Y, algo, epochs=200, mini_batch_size=6):
        """
        Parameters:
            [self.X, self.Y]: training data
            algo: optimization algorithm
        """
        self.X = X
        self.Y = Y

        if algo == 'GD':
            eta = 0.5
            for i in range(epochs):
                data = list(zip(X, Y))
                random.shuffle(data)
                dw, db = 0, 0
                for x, y in data:
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                self.w -= eta * dw / X.shape[0]
                self.b -= eta * db / X.shape[0]
                self.append_log()

        elif algo == 'SGD':
            eta = 5
            for i in range(epochs):
                dw, db = 0, 0
                data = list(zip(X,Y))
                random.shuffle(data)
                for x, y in data:
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                    self.w -= eta * dw
                    self.b -= eta * db
                    self.append_log()

        elif algo == 'MiniBatch':
            for i in range(epochs):
                dw, db = 0, 0
                points_seen = 0
                data = list(zip(X,Y))
                random.shuffle(data)
                eta = 0.5
                for x, y in data:
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                    points_seen += 1
                    if points_seen % mini_batch_size == 0:
                        self.w -= eta * dw / mini_batch_size
                        self.b -= eta * db / mini_batch_size
                        self.append_log()
                        dw, db = 0, 0

        # Momentum is a method that helps accelerate
        #  SGD in the relevant direction and dampens
        #  oscillations.
        elif algo == 'Momentum':
            v_w, v_b = 0, 0
            eta = 0.1
            gamma = 0.9
            for i in range(epochs):
                dw, db = 0, 0
                data = list(zip(X,Y))
                random.shuffle(data)
                for x, y in data:
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                v_w = gamma * v_w + eta * dw
                v_b = gamma * v_b + eta * db
                self.w = self.w - v_w
                self.b = self.b - v_b
                self.append_log()

        # Nesterov Accelerated Gradient (NAG):
        #  is a way to give momentum term kind of
        #  prescience. We calculate the gradient not
        #  w.r.t to our current parameters \theta
        #  but w.r.t the approximate future position of
        #  our parameter.
        elif algo == 'NAG':
            v_w, v_b = 0, 0
            eta = 0.1
            gamma = 0.9
            for i in range(epochs):
                dw, db = 0, 0
                data = list(zip(X,Y))
                random.shuffle(data)
                v_w = gamma * v_w
                v_b = gamma * v_b
                for x, y in data:
                    dw += self.grad_w(self.w - v_w, self.b - v_b)
                    db += self.grad_b(self.w - v_w, self.b - v_b)
                v_w = v_w + eta * dw
                v_b = v_b + eta * db
                self.w = self.w - v_w
                self.b = self.b - v_b
                self.append_log()

        # Adagrad:
        #  It adapts the learning rate to the parameters,
        #  performing smaller updates (i.e. low learning rate)
        #  for parameters associated with frequently
        #  occruing features, and large updates (i.e. high
        #  learning rates) for parameters associated with
        #  infrequent features. For this reason, it is well
        #  suited for dealing with sparse data.
        elif algo == 'Adagrad':
            eps = 1E-5
            dw, db = 0, 0
            v_w, v_b = 0, 0
            eta = 0.2
            for i in range(epochs):
                data = list(zip(X, Y))
                random.shuffle(data)
                for x, y in data:
                    dw = self.grad_w(self.w, self.b)
                    db = self.grad_b(self.w, self.b)
                    v_w += dw*dw
                    v_b += db*db
                    self.w = self.w - eta/(np.sqrt(v_w) + eps)*dw
                    self.b = self.b - eta/(np.sqrt(v_b) + eps)*db
                    self.append_log()
        # Adadelta:
        #  Adadelta is an extension of Adagrad that seeks
        #  to reduce its aggressive, mononically decreasing
        #  learning rate. Instead of accumulating all past
        #  squared gradients, Adadelta restricts the window
        #  past gradients to some fixed size w.
        elif algo == 'AdaDelta':
            e_g2, e_x2 = 0, 0
            dw, db = 0, 0
            v_w, v_b = 0, 0
            eta = 5.
            eps = 1E-05
            gamma = 0.9
            beta = 0.9
            for i in range(epochs):
                data = list(zip(X, Y))
                random.shuffle(data)
                for x, y in data:
                    dw = self.grad_w(self.w, self.b)
                    db = self.grad_b(self.w, self.b)
                    e_g2 = gamma*e_g2 + (1 - gamma)*(dw*dw + db*db)
                    v_w = np.sqrt(e_x2 + eps)*dw/(np.sqrt(e_g2 + eps))
                    v_b = np.sqrt(e_x2 + eps)*db/(np.sqrt(e_g2 + eps))
                    e_x2 = beta*e_x2 + (1 - beta)*(v_w*v_w + v_b*v_b)
                    self.w = self.w - eta*v_w
                    self.b = self.b - eta*v_b
                    self.append_log()

        # RMSprop:
        #   RMSprop is an unpublished, adaptive learning rate
        #   method proposed by Geoff Hinton in his Coursera Class.
        elif algo == 'RMSprop':
            e_g2 = 0
            dw, db = 0, 0
            v_w, v_b = 0, 0
            eps = 1E-05
            gamma = 0.9
            eta = 0.2

            for i in range(epochs):
                data = list(zip(X, Y))
                random.shuffle(data)
                for x, y in data:
                    dw = self.grad_w(self.w, self.b)
                    db = self.grad_b(self.w, self.b)
                    e_g2 = gamma*e_g2 + (1 - gamma)*(dw*dw + db*db)
                    v_w = eta/(np.sqrt(e_g2 + eps))*dw
                    v_b = eta/(np.sqrt(e_g2 + eps))*db
                    self.w = self.w - v_w
                    self.b = self.b - v_b
                    self.append_log()

        # Adam:
        #   Adaptive Moment Estimation (Adam) is another method that
        #   computes adaptive learning rates for each parameters.
        elif algo == 'Adam':
            beta1, beta2 = 0.9, 0.999
            gamma1, gamma2 = 0.02, 0.01
            v_w, v_b = 0, 0
            v_t = 0
            eta = 0.2
            eps = 1E-5
            niter = 0
            for i in range(epochs):
                data = list(zip(X, Y))
                random.shuffle(data)
                for x, y in data:
                    dw = self.grad_w(self.w, self.b)
                    db = self.grad_b(self.w, self.b)
                    niter += 1
                    v_w = (beta1*v_w + (1 - beta1)*dw)/(1 - gamma1)
                    v_b = (beta1*v_b + (1 - beta1)*db)/(1 - gamma2)
                    v_t = beta2*v_t + (1 - beta2)*(dw*dw + db*db)
                    v_t = v_t/(1 - np.power(0.6, niter))
                    self.w = self.w - eta/(np.sqrt(v_t) + eps)*v_w
                    self.b = self.b - eta/(np.sqrt(v_t) + eps)*v_b
                    self.append_log()

        # AdaMax:
        #   AdaMax is a revised Adam method with p-norm
        elif algo == 'AdaMax':
            beta1, beta2 = 0.9, 0.999
            gamma1, gamma2 = 0.02, 0.01
            v_w, v_b = 0, 0
            u_t = 0
            eps = 1E-5
            eta = 0.3
            for i in range(epochs):
                data = list(zip(X, Y))
                random.shuffle(data)
                for x, y in data:
                    dw = self.grad_w(self.w, self.b)
                    db = self.grad_b(self.w, self.b)
                    v_w = (beta1*v_w + (1 - beta1)*dw)/(1 - gamma1)
                    v_b = (beta1*v_b + (1 - beta1)*db)/(1 - gamma1)
                    u_t = np.max((beta2*u_t, np.sqrt(dw*dw + db*db)))
                    self.w = self.w - eta/(np.sqrt(u_t))*v_w
                    self.b = self.b - eta/(np.sqrt(u_t))*v_b
                    self.append_log()

        # AMSGrad:
        #   AMSGrad that uses the maximum of past squared
        #   gradients v_t rather than exponential average
        #   to update the parameters.
        elif algo == 'AMSGrad':
            beta1, beta2 = 0.9, 0.9
            eta = 0.5
            eps = 1E-5
            v_w, v_b = 0, 0
            v_t, u_t = 0, 0
            for i in range(epochs):
                data = list(zip(X, Y))
                random.shuffle(data)
                for x, y in data:
                    dw = self.grad_w(self.w, self.b)
                    db = self.grad_b(self.w, self.b)
                    v_w = beta1*v_w + (1 - beta1)*dw
                    v_b = beta1*v_b + (1 - beta1)*db
                    v_t = beta2*v_t + (1 - beta2)*(dw*dw + db*db)
                    u_t = np.max((v_t, u_t))
                    self.w = self.w - eta/(np.sqrt(u_t) + eps)*v_w
                    self.b = self.b - eta/(np.sqrt(u_t) + eps)*v_b
                    self.append_log()

     #logging
    def append_log(self):
        self.w_h.append(self.w)
        self.b_h.append(self.b)
