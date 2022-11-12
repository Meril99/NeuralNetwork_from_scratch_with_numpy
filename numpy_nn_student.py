#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: XXX and YYY
"""

import struct
import numpy as np
import time
import matplotlib.pyplot as plt


# provided function for reading idx files
def read_idx(filename):
    '''Reads an idx file and returns an ndarray'''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Task 1: reading the MNIST files into Python ndarrays
        
# Task 2: visualize a few bitmap images
        
# Task 3: input pre-preprocessing    

# Task 4: output pre-processing
        
# Task 5-6: creating and initializing matrices of weights
        
# Task 7: defining functions sigmoid, softmax, and sigmoid'
        
# Task 8-9: forward pass
        
# Task 10: backpropagation
        
# Task 11: weight updates
        
# Task 12: computing error on test data
        
# Task 13: error with initial weights
        
# Task 14-15: training

# Task 16-18: batch training
