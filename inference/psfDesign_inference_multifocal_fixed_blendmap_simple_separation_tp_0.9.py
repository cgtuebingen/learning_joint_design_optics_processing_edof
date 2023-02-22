#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:42:15 2020

@author: Jieen Chen

This is the corresponding inference script for the simple view separation script.
The input of this script is:
    I_o--a full-view (wide-angle view) image.
The expected outputs are:
    I_s--intermediate image, the view multiplexed image
    I_w--wide angle view image
    I_n--narrow angle view image

For debug purpose, the input is the same with the training data. 
And the exepcted contents of I_w and I_n are the output of the training network.
"""
import matplotlib

matplotlib.use('Agg')

import numpy as np
from tensorpack import *
from tensorpack.utils.viz import *
import argparse
from tensorpack.tfutils.summary import add_moving_summary
import tensorflow as tf
# import tensorflow.contrib
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.utils.argtools import shape2d, shape4d
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.globvars import GlobalNS as use_global_argument
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu
import matplotlib.pyplot as plt
import os
import load_data
import cv2
import reconstruction
from fft_tf import fftshift2d_tf, ifftshift2d_tf, conv_fft2
from psfDesign_train_multifocal_fixed_blendmap_simple_separation_tp_0.9 

# ------------------------------------
# Global Settings
# ------------------------------------

BATCH = 1         # number of sequences per update
SHAPE = 4800  # dimension simulation grid
INSHAPE = SHAPE // 24  # dimension of input shape if different from shape
FEATURE_FACTOR = 1  # scaling factor for feature number in reconstruction CNN (a factor of 2 cuts filters in half))

# ZERNIKE_MODES = 2        # used bands of Zernike polynomials
SENSOR_RES = SHAPE  # Resolution of sensor (one dimension) assuming a square sensor, so actual resolution is square
# SENSOR_PITCH = 8.0e-6  # Pixel distance in m - using the pitch of the SLM pixels here for now
SLM_RES = [2400, 1350]  # Resolution of SLM, gets padded to a square of the first dimension,
L_SLM = 15.36e-3  # length of longer size of SLM in m, assuming 16:9 aspect for SLM here
ASPECT_FACTOR = 0.5625  # factor to represent aspect ratio

PAD = 200       # padding for propagations, experimentally evaluated for depth range and grid spacings

F_NUMBER = 5.6  # aperture opening 1/stops
# L_SENSOR = 15.15e-3  # length of one size of camera sensor in m, assuming square of longer sides (if non-sqaure)
L_SENSOR = 3.45e-6 * 4096     # length of one size of camera sensor in m
FOCAL_LENGTH = .035  # focal length in m of the fixed lens element
FOCUS_DISTANCE = -1  # distance of focus plane from pupil of optical element in m, -1 for infinity

# distance between optical elements and image plane (z)
if FOCUS_DISTANCE == -1:
    Z_DISTANCE = FOCAL_LENGTH
else:
    Z_DISTANCE = np.abs((FOCUS_DISTANCE * FOCAL_LENGTH) / (FOCUS_DISTANCE - FOCAL_LENGTH))

Z2_DISTANCE = FOCAL_LENGTH  # distance between SLM and lens in m - this is an approximated, measured value

# diameter of aperture (region of SLM)
APERTURE = FOCAL_LENGTH / F_NUMBER  # effective aperture of the lens (diameter of entrance pupil)
# circular aperture is 8.64e-3 (L_SLM), rectangulat 12.219e-3

N_LAYERS = 8  # number of convolution layers in decoder network, must be multiple of 2, added one
# additional layer for output so it's currently N +1 conv layers
CHANNELS = 1  # image channels used, e.g. 1 for greyscale, 3 for rgb
WAVELENGTHS = [5.2e-7]  # wavelengths corresponding to channels in m
CURVATURE_APPROX = True  # whether to use fresnel approximation of incident spherical wavefront, default is off
RECT_APERTURE = 1
SLM_CROP = False
RND_WVL = False  # a wavelength is chosen from the list randomly to enable multi spectral training at no extra memory cost
# strength of random perspective transform applied to psf, off if 0, adequate values are between 0.01 and 0.2
RND_TRANSFORM = 0
# factor by which an unmodulated version of the psf is blended into the final psf to simulate zeroth order diffraction
# peak
ZEROTH_ORDER = 0
# DISTANCES = [100, 1, 0.8, 0.67, 0.58, 0.5, 0.40]  # distances d of object to optical system - relevant for EDOF applications
# DISTANCES = [10.0, 4.5, 2.5, 1.8, 1.4, 1.0, 0.9, 0.7, 0.6, 0.5]
# DISTANCES = [50.0, 10.0, 4.5, 2.5, 1.8, 1.5, 1.2, 0.9, 0.7, 0.6, 0.5]
DISTANCES = [100.0]

# set a fixed log dir outside of repository
# LOGDIR = '/graphics/scratch/engelhar/train_logs/model_slm45k_multifocal_blendmap_unet5'
LOGDIR = '../train_logs/model_slm2k_2proptest_fixed_blendmap_simple_separation'

# path to training data lmdb
TRAIN_PATH = '/graphics/scratch/engelhar/datasets/google-highres_train_80k.lmdb'
# TRAIN_PATH = '/home/andy/Projects/datasets/google-images/google-highres_train_80k.lmdb'

# path to testing data lmdb
# VAL_PATH = '/home/andy/Projects/datasets/google-images/google-highres_val_800.lmdb'
VAL_PATH = '/graphics/scratch/engelhar/datasets/google-highres_val_800.lmdb'

# path to data for export as lmdb
# TEST_PATH = '/home/andy/Projects/datasets/google-images/google-highres_val_800.lmdb'
TEST_PATH = '/graphics/scratch/engelhar/datasets/google-highres_val_800.lmdb'

# path to write the export files after training to
# EXPORT_PATH = '/graphics/scratch/engelhar/test_data/unet5_slm45k_multifocal_blendmap'
EXPORT_PATH = '../test_data/plainCNN_slm2k_2proptest_fixed_blendmap_simple_separation'
# maximum number of outputted during test callback, set to -1 if all in dataflow are to be outputted
MAX_TEST = 16

if len(WAVELENGTHS) != CHANNELS:
    print('Elements in list of wavelengths does not match the number of channels! On Purpose?...')


# ------------------------------------
# Utilities
# ------------------------------------

# PSNR copied from https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/tfutils/symbolic_functions.py
def psnr(prediction, ground_truth, maxp=None, name='psnr'):
    """`Peek Signal to Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.
    Args:
        prediction: a :class:`tf.Tensor` representing the prediction signal.
        ground_truth: another :class:`tf.Tensor` with the same shape.
        maxp: maximum possible pixel value of the image (255 in in 8bit images)
    Returns:
        A scalar tensor representing the PSNR.
    """

    maxp = float(maxp)

    def log10(x):
        with tf.name_scope("log10"):
            numerator = tf.log(x)
            denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

    mse = tf.reduce_mean(tf.square(prediction - ground_truth))
    if maxp is None:
        psnr = tf.multiply(log10(mse), -10., name=name)
    else:
        psnr = tf.multiply(log10(mse), -10.)
        psnr = tf.add(tf.multiply(20., log10(maxp)), psnr, name=name)

    return psnr


