"""
MODULE TO DEFINE ACTIVATION FUNCTIONS

This module defines:
- the logsig function and its derivative
- the tansig function and its derivative

Date: January 2017
Author: Mareva BRIXY
"""

# coding: utf8
from __future__ import unicode_literals

import math
import numpy as np

class ActivationFunction:
    """ This class defines an activation function: its expression and its derivative. """

    def __init__(self, function, derivative):
        """ The arguments 'function' and 'derivative' correspond to a function of one variable.

        Arguments:
        function                Python method               activation function
        derivative              Python method               first derivative of activation function

        Modifed attributes:
        function                Python method               activation function
        derivative              Python method               first derivative of activation function
        """

        self.function = np.vectorize(function)
        self.derivative = np.vectorize(derivative)


# Definition of the function logsig and tansig and their derivatives
def logsig_fct(x):
    return 1 / (1 + math.exp(-x))

def tansig_fct(x):
    return 2 / (1 + math.exp(- 2 * x)) - 1

def logsig_deriv(x):
    return math.exp(-x) / ((1 + math.exp(-x)) * (1 + math.exp(-x)))

def tansig_deriv(x):
    return (4 * math.exp(- 2 * x)) / ((1 + math.exp(- 2 * x)) * (1 + math.exp(- 2 * x)))

# Definition of the objets Logsig and Tansig
logsig = ActivationFunction(logsig_fct, logsig_deriv)
tansig = ActivationFunction(tansig_fct, tansig_deriv)
