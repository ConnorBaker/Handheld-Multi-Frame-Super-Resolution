# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:22:15 2022

@author: jamyl
"""
import time
import math

import numpy as np
from numba import float32, cuda
import torch as th


DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
EPSILON_DIV = 1e-10

DEFAULT_THREADS = 16


def getTime(currentTime, labelName, printTime=True, spaceSize=50):
	'''Print the elapsed time since currentTime. Return the new current time.'''
	if printTime:
		print(labelName, ' ' * (spaceSize - len(labelName)), ': ', round((time.perf_counter() - currentTime) * 1000, 2), 'milliseconds')
	return time.perf_counter()

def isTypeInt(array):
	'''Check if the type of a numpy array is an int type.'''
	return array.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.uint, np.int64]


def getSigned(array):
	'''Return the same array, casted into a signed equivalent type.'''
	# Check if it's an unsigned dtype
	dt = array.dtype
	if dt == np.uint8:
		return array.astype(np.int16)
	if dt == np.uint16:
		return array.astype(np.int32)
	if dt == np.uint32:
		return array.astype(np.int64)
	if dt == np.uint64:
		return array.astype(np.int64)

	# Otherwise, the array is already signed, no need to cast it
	return array


def mse(im1, im2):
    return np.linalg.norm(im1 - im2) / np.prod(im1.shape)


def divide(num, den):
    """
    Performs num = num/den (in place), via CuPy. ``num``/``den`` may be numba
    device arrays, torch tensors or cupy arrays (zero-copy through the CUDA
    array interface) -- numba kernels cannot JIT on this hardware.
    """
    import cupy as cp
    assert num.shape == den.shape
    n = cp.asarray(num)
    cp.divide(n, cp.asarray(den), out=n)

        
def add(A, B):
    """
    Performs A += B (in place), via CuPy. ``A``/``B`` may be numba device
    arrays, torch tensors or cupy arrays (zero-copy through the CUDA array
    interface) -- numba kernels cannot JIT on this hardware.
    """
    import cupy as cp
    assert A.shape == B.shape
    a = cp.asarray(A)
    a += cp.asarray(B)

    
def round_iso(iso):
    n = round(math.log2(iso/100))
    rounded_iso = 100 * (2**n)
    return int(rounded_iso)


def timer(func, enabled, start_s=None, end_s=None, spaceSize=50):
    def wrapper(*args, **kwargs):
        cuda.synchronize()
        t1 = time.perf_counter()
        if start_s is not None:
            print(start_s)
        
        out = func(*args, **kwargs)
        
        cuda.synchronize()

        if end_s is not None:
            print(end_s, ' ' * (spaceSize - len(end_s)), ': ', round((time.perf_counter() - t1) * 1000, 2), 'milliseconds')
        
        return out
    if enabled:
        return wrapper
    else:
        return func


    
    
