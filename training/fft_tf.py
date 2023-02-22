# -*- coding: utf-8 -*-
# Author: Andreas Engelhardt
# Fast Fourier Transform utilities for Tensorflow

import numpy as np
import tensorflow as tf


# fft shift 2D version based on https://github.com/vsitzmann/deepoptics/blob/master/src/layers/optics.py
# These enable GPU usage since there is no GPU kernel for roll.
def tf_fftshift2d(a_tensor):
    input_shape = a_tensor.shape.as_list()
    if len(input_shape) > 2:
        s = 1
        t = 3
    else:
        s = 0
        t = 2
    new_tensor = a_tensor
    for axis in range(s, t):
        split = (input_shape[axis] + 1)//2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def tf_ifftshift2d(a_tensor):
    input_shape = a_tensor.shape.as_list()
    if len(input_shape) > 2:
        s = 1
        t = 3
    else:
        s = 0
        t = 2
    new_tensor = a_tensor
    for axis in range(s, t):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


# Fftshift equivalent for tensorflow, shifts the values so that zero frequency is in center, but CPU only.
def fftshift_batch(f, res):
    fshift = f
    for dim in range(0, CHANNELS + 1):
        fshift = tf.manip.roll(fshift, shift=int(res / 2), axis=dim + 1)

    return fshift


# performs convolution in fourier space using tensorflow ops,
# assuming real inputs of type float32, outputs a real tensor of type float32
def tf_conv2d(x, k, x_size, k_size):
    csize = x_size + k_size  # - 1
    # padding kernel
    k = tf.pad(k, tf.constant([[int((csize - k_size) / 2), int((csize - k_size) / 2)],
                               [int((csize - k_size) / 2), int((csize - k_size) / 2)]]), "CONSTANT")

    # expanding kernel for batch size
    k = tf_fftshift2d(k)
    k = tf.expand_dims(k, 0)
    # padding image
    x_padded = tf.pad(x, tf.constant([[0, 0], [int((csize - x_size) / 2), int((csize - x_size) / 2)],
                                      [int((csize - x_size) / 2), int((csize - x_size) / 2)]]), "SYMMETRIC")
    k_f = tf.fft2d(tf.complex(k, 0.0))
    # k_f = tf.complex(k, 0.0)
    x_f = tf.fft2d(tf.complex(x_padded, 0.0))

    # operation
    # multiply support broadcasting which enables support of memory efficient computation of batches
    conv = tf.multiply(x_f, k_f)
    conv_r = tf.real(tf.ifft2d(conv))
    # remove padding
    img_pad = (csize - x_size)/2
    conv_r = conv_r[:, img_pad:-img_pad, img_pad:-img_pad]

    return conv_r
