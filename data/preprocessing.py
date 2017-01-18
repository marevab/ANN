"""
MODULE FOR DATA PREPROCESSING

This module defines some useful functions in order to preprocess data (before training).

Date: January 2017
Author: Mareva BRIXY
"""

# coding: utf8
from __future__ import unicode_literals

import math
import numpy as np

def label_to_vector_form(labels, C):
    """ This function enables to define a list of labels like [4,1,2,0] into a vector form,
    such as:
    [4, 1, 2, 0] -> [[0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]

    This function is useful during the training phase of a neural network
    whose output neurons correspond to the different classes offered by the training set.

    Let N be the size of the list of labels.

    Arguments:
    labels                  N x 1 array                 labels
    C                       int                         number of categories

    Arguments:
    labels_vector           N x C array                 labels in a vector form
    """

    N = len(labels)
    x = range(N)
    y = labels
    labels_vector = np.zeros([N, C])
    xy = zip(x,y)

    for i, coord in enumerate(xy):
        labels_vector[coord] = 1

    return labels_vector


def norma_by_max(data):
    """ This function enables to normalize a dataset between 0 and 1.
    All values are divided by the maximum value that can be taken.

    We assume that:
    - there are only positive values in the dataset
    - each covariate reach the same maximum

    Arguments:
    data                    N x P array                 dataset of N samples of P features

    Arguments:
    norm_data               N x P array                 normalized dataset
    """

    norm_data = data / np.max(np.max(data))
    return norm_data

