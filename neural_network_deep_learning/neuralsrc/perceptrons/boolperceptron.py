"""
A module to implement the perception algorithm.
            |---
            | 1, w*x + b > 0,
f(w*x + b) =|
            | 0, w*x + b <=0.
            |---
"""

# Third-party libraries
import numpy as np

class BoolPerceptron(object):

    def boolnand(self, input_data):
        weights = np.array([-2, -2])
        bias = 4
        return(signfun(np.dot(weights, input_data) + bias))

    def booland(self, input_data):
        weights = np.array([2, 2])
        bias = -3
        return(signfun(np.dot(weights, input_data) + bias))

    def boolor(self, input_data):
        weights = np.array([2, 2])
        bias = -1
        return(signfun(np.dot(weights, input_data) + bias))


# Miscellaneous functions
def signfun(z):
    """sign function"""
    if z > 0:
        return 1
    else:
        return 0
