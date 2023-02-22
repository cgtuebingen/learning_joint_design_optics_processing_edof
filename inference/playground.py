#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:56:28 2020

@author: Jieen Chen

This script is for experimenting the functions.
"""

from scipy import special
import numpy as np
import matplotlib.pyplot as plt

n = 30
x = np.arange(-n/2,n/2)

[X,Y] = np.meshgrid(x, x)
r = 0.05*np.sqrt(np.multiply(X,X) + np.multiply(Y,Y))
z = np.ones(r.shape)

# bessel function
squared_b = scipy.special.jv(2*np.pi*r,z)
b = np.multiply(squared_b, squared_b)

b = b/sum(b)

plt.imshow(b)

print(b.max())