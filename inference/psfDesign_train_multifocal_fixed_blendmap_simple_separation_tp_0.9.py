#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:34:14 2020

@author: Jieen Chen

This script trains a simple view separation network for testing and debugging.
The intermediate image is an equal-weighted sum of two views. 
The resolution is set to be 100x100 for training efficiency.
The cost function has two components: 1. data fitting term 2. view relation term.
The weight for the view-relation term is 1e-3.
The number of epoch is 100. Each epoch has 1000 iterations. 
The decoder is unet5.
PSF simulation is not considered. 
The recorded images are:
    I_s--intermediate image
    I_w_gt--ground truth wide angle view
    I_n_gt--ground truth narrow view
    I_w--wide-angle view output
    I_n--narrow view output

A corresponding inference script is developed to use the training. 
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


# creates a specified number of random perspective transforms that can be used with tf.contrib.image.transform
# inspired by https://github.com/aleju/imgaug
def create_perspective_transforms(strength, shape, number=20):
    transforms = []
    translations = []
    print('creating perspective transforms...')

    # add identity transform
    transforms.append(np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32))
    translations.append(np.float32([0, 0]))

    for i in range(0, number):
        rnd_pert = np.random.normal(0, .5, 8) * strength * shape[0] - (strength * shape[0]) / 2.0

        # set up first point set - mutated four corners
        src_pts = np.zeros((4, 2), dtype=np.float32)

        src_pts[0] = [0 + rnd_pert[0], 0 + rnd_pert[1]]
        src_pts[1] = [shape[0] + rnd_pert[2], 0 + rnd_pert[3]]
        src_pts[2] = [shape[0] + rnd_pert[4], shape[1] + rnd_pert[5]]
        src_pts[3] = [0 + rnd_pert[6], shape[1] + rnd_pert[7]]

        # determine dimensions of transformed image
        nwidth = np.maximum(np.abs(src_pts[0][0] - src_pts[1][0]), np.abs(src_pts[2][0] - src_pts[3][0]))
        nheight = np.maximum(np.abs(src_pts[0][1] - src_pts[3][1]), np.abs(src_pts[1][1] - src_pts[2][1]))

        # second point set, transformed image corners
        dst = np.array([[0, 0],
                        [nwidth - 1, 0],
                        [nwidth - 1, nheight - 1],
                        [0, nheight - 1]], dtype=np.float32)

        # use opencv to calculate perspective transform (closed form solution)
        # we map from destination to source points
        tm = cv2.getPerspectiveTransform(dst, src_pts)

        # compensate for shift of center point
        # center point in destination coordinates
        center_point = np.array([[[shape[0] / 2, shape[1] / 2]]], dtype=np.float32)

        # get ransformed center point
        ncenter = cv2.perspectiveTransform(center_point, tm)

        # create translation matrix
        translation = np.float32([(ncenter[0, 0, 0] - center_point[0, 0, 0]), (ncenter[0, 0, 1] - center_point[0, 0, 1])])

        translations.append(translation)

        tm = tm.flatten()

        # store parameters in format suitable for tensorflow's image transform (8 parameters as flat vector)
        transforms.append(tm[0:8])

    return transforms, translations


# todo: split into get transform and appy transform for multispectral version
# based on a set of precomputed perspective transforms, perturbate an image
def random_perturbation(image_tensor, strength, name, number=32):
    image_shape = image_tensor.shape.as_list()
    # make sure tensor has 4 dims
    if len(image_shape) < 4:
        height = image_shape[0]
        width = image_shape[1]
        if len(image_shape) > 2:
            image_tensor = tf.reshape(image_tensor, [1, height, width, image_shape[2]])
        else:
            image_tensor = tf.reshape(image_tensor, [1, height, width, 1])
    else:
        height = image_shape[1]
        width = image_shape[2]
    # create transforms and save in variable
    tmptf, tmptl = create_perspective_transforms(strength, [height, width], number)
    tmptf = np.float32(tmptf)
    tmptl = np.float32(tmptl)
    transforms = tf.get_variable(name + '_transforms', initializer=tf.constant_initializer(tmptf),
                                               shape=[len(tmptf), 8], dtype=tf.float32, trainable=False)
    translations = tf.get_variable(name + '_translations', initializer=tf.constant_initializer(tmptl),
                                               shape=[len(tmptl), 2], dtype=tf.float32, trainable=False)

    t_idx = tf.random_uniform([2], 0.0, 1.0, dtype=tf.float32)
    select = tf.cast(t_idx[0] * (number - 1), tf.int32)
    transform = tf.gather(transforms, select, name='gather_transform')
    translation = tf.gather(translations, select, name='gather_translation')
    image_transformed = tf.contrib.image.transform(image_tensor, transform, name=name)
    image_transformed = tf.contrib.image.translate(image_transformed, translation)
    image_transformed = tf.image.resize_image_with_crop_or_pad(image_transformed, height, width)
    # only perturbate half of the time
    output = tf.where(t_idx[1] > 0, image_transformed, image_tensor)

    # convert back to input format
    if len(image_shape) < 4:
        if len(image_shape) > 2:
            output = output[0, :, :, :]
        else:
            output = output[0, :, :, 0]
    return output


# encode image data assumed to be linear with SRGB 'gamma' transfer function, color primaries remain untouched
# output is in range 0 to 255 float
# https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_(%22gamma%22)
def linear2srgb(im):
    im_srgb = im
    alpha = 0.055
    im_srgb[im <= 0.0031308] *= 12.92
    bool_mask = im > 0.0031308
    im_srgb[bool_mask] = im_srgb[bool_mask] ** (1.0 / 2.4) * (1.0 + alpha) - alpha

    return im_srgb * 255.0


# using tensorflow ops
def tf_linear2srgb(im):
    # im_srgb = im
    alpha = 0.055
    mask = im <= 0.0031308
    im_tmp = tf.where(mask, im * 12.92, im, name='cond1_linear2srgb')
    im_tmp2 = tf.pow(im, 1.0 / 2.4) * (1.0 + alpha) - alpha
    im_srgb = tf.where(tf.logical_not(mask), im_tmp2, im_tmp, name='cond2_linear2srgb')

    return im_srgb * 255.0


# creates a circle in coordinate grid x, y with diameter d
def circ(x, y, d):
    r = np.sqrt(x * x + y * y)
    circle = np.array(r < d / 2.0, dtype=float)
    circle[r == d / 2.0] = 0.5
    return circle


# creates a circle like circ() above but with tensorflow ops, no anti-aliasing
def tf_circ(x, y, d):
    r = tf.sqrt(x * x + y * y)
    circle = tf.cast(r <= d / 2.0, dtype=tf.float32)
    # mask = r == d/2
    # circle = tf.where(mask, tf.constant(0.5, shape=[SLM_RES, SLM_RES], dtype=tf.float32), circle, name='cond2_circle')
    return circle


# creates a rectangular coordinate grid from vector x and side length d
def tf_rect(x, y, d1, d2):
    x = tf.abs(x)
    y = tf.abs(y)
    y1 = tf.cast(x <= d1 / 2.0, dtype=tf.float32)
    y2 = tf.cast(y <= d2 / 2.0, dtype=tf.float32)
    return tf.multiply(y1, y2)


# convert cartesian to polar coordinates (numpy arrays only)
def cart2pol(x, y):
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return theta, rho


# computes the complex exponential with the help of Euler's formula since there is no cuda kernel for it in (older) tf
def tf_exp_complex(phase, name):
    return tf.complex(tf.cos(phase), tf.sin(phase), name=name)


# method to downsample a 2D tensor by a factor using area interpolation (averaging)
def area_downsample(image, factor, name):
    factor = int(factor)
    output = tf.nn.avg_pool(image, [1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)
    return output


# upsample a tensor by a factor of two - the reverse of area downsample
def upsample_nearest(image, factor, name):
    # extend to four dimensional tensor
    img_shape = image.shape.as_list()
    if len(img_shape) < 4:
        image = tf.reshape(image, [1, img_shape[0], img_shape[1], 1])
        target_res_x = factor * img_shape[0]
        target_res_y = factor * img_shape[1]
    else:
        target_res_x = factor * img_shape[1]
        target_res_y = factor * img_shape[2]
    # use convolutional layer for upsampling with constant kernel, transposed
    # image_big = tf.nn.conv2d_transpose(image, tf.ones([factor, factor, 1, 1]),
    #                                    output_shape=[1, target_res_x, target_res_y, 1],
    #                                    strides=[1, factor, factor, 1],
    #                                    padding='SAME',
    #                                    name=name)
    image_big = tf.image.resize_nearest_neighbor(image, [target_res_x, target_res_y], align_corners=True, name=name)
    if len(img_shape) < 4:
        image_big = image_big[0, :, :, 0]
    return image_big


# compute modulo on GPU
def modulo_tf(a, b):
    return a - tf.floor_div(a, b) * b


# propagates field Uin along distance z using fresnel approximation (near field),
# ds discretization size (delta parameter), res is resolution of system
def ang_fresnel_propagation(u_in, z, ds, wavelength, res):
    c = WAVELENGTHS.index(wavelength)
    # padding to reduce artifacts, quarter resolution seems to be enough
    delta = ds
    n_pad = res // 4
    n = res + 2 * n_pad
    delta_freq = 1 / (delta * n)
    # delta_freq = 1 / L_SLM
    coords_freq = tf.linspace((- n / 2.0) * delta_freq, (n / 2.0 - 1) * delta_freq, n,
                              name='linspace_freqCoords_c%s' % c)
    x, y = tf.meshgrid(coords_freq, coords_freq, name='meshgrid_freqCoordinates_c%s' % c)
    # shift coordinates
    x = ifftshift(x, n)
    y = ifftshift(y, n)
    squared_sum_freq = x * x + y * y

    u_in_padded = tf.pad(u_in, [[n_pad, n_pad], [n_pad, n_pad]])
    # ifftshift on coordinates instead of doing it later on tensors
    # squared_sum_freq = ifftshift(squared_sum_freq)
    # phase shift for distance z
    phi_z = np.pi * wavelength * -1.0 * squared_sum_freq * z
    h = tf_exp_complex(phi_z, 'exp_phase_z_c%s' % c)
    # Fourier transform of input field
    # uin_spec = fftshift(tf.fft2d(fftshift(u_in_padded, n), name='fft2d_uin_c%s' % c), n)
    uin_spec = tf.fft2d(u_in_padded, name='fft2d_uin_c%s' % c)
    # uin_spec = tf.multiply(uin_spec, delta * delta, name='mult_delta2_c%s' % c)

    # multiply fields
    uout_spec = tf.multiply(uin_spec, h, name='mult_h_c%s' % c)

    # inverse Fourier transform to get field on sensor
    delta_inv = 1 / delta
    u_out = tf.ifft2d(uout_spec, name='ifft2d_uout_c%s' % c)
    # u_out = ifftshift(tf.ifft2d(ifftshift(uout_spec, n), name='ifft2d_uout_c%s' % c), n) * \
    #       (delta_inv * delta_inv)
    u_out = u_out[n_pad:-n_pad, n_pad:-n_pad]
    return u_out


# angular spectrum propagation with variable grid spacings according to Schmidt
# u_in: input field, z: distance, ds,do: grid spacings source and observation plane, r1sq: source coordinates squared,
# res: resolution of grid (one dimension, square assumed), window: window function of size (res,res) to be appöoed on
# source field (anti-aliasing quick fix)
def ang_fresnel_propagation_schmidt(u_in, z, ds, do, r1sq, wavelength, res):
    c = WAVELENGTHS.index(wavelength)
    wvl = tf.convert_to_tensor(wavelength, dtype=tf.float32)
    # padding to reduce artifacts, quarter resolution seems to be enough
    delta1 = ds
    delta2 = do
    n = res
    n_pad = 200
    n_padded = n + 2 * n_pad

    # wavenumber k
    k = 2 * np.pi / wvl

    # source coordinates are represented by r1sq

    # spatial frequencies of source plane
    delta_freq = 1 / (delta1 * n_padded)
    coordsf = tf.linspace(-n_padded // 2 * delta_freq, (n_padded // 2 - 1) * delta_freq, n_padded,
                          name='linspace_freqCoords_c%s' % c)
    fx1, fy1 = tf.meshgrid(coordsf, coordsf, name='meshgrid_freqCoordinates_c%s' % c)
    fsq = fx1 * fx1 + fy1 * fy1

    # scaling parameter
    m = delta2 / delta1

    # observation plane coordinates
    coords2 = tf.linspace(-n // 2 * delta2, (n // 2 - 1) * delta2, n,
                          name='linspace_r2Coords_c%s' % c)
    x2, y2 = tf.meshgrid(coords2, coords2, name='meshgrid_r2Coordinates_c%s' % c)
    r2sq = x2 * x2 + y2 * y2

    # quadratic phase factors
    q1 = tf_exp_complex(k / 2. * (1. - m) / z * r1sq, name='phase_q1_c%s' % c)
    q2 = tf_exp_complex(np.pi * wavelength * -1.0 * z / m * fsq, name='phase_q2_c%s' % c)
    q3 = tf_exp_complex(k / 2. * (m - 1.) / (m * z) * r2sq, name='phase_q3_c%s' % c)

    # window to avoid anti-aliasing, Todo: improve!
    # u_in = u_in * tf.complex(window, window)

    # compute output field at observation plane
    u = tf.multiply(u_in / m, q1, name='mult_q1_c%s' % c)
    # pad field for convolution
    u_padded = tf.pad(u, [[n_pad, n_pad], [n_pad, n_pad]], name='pad_u_c%s' % c)
    # perform fft
    u_fft = fftshift2d_tf(tf.fft2d(fftshift2d_tf(u_padded), name='fft_as_c%s' % c)) * (delta1 * delta1)
    u = u_fft * q2
    # transform back
    factor = n_padded * delta_freq
    u = ifftshift2d_tf(tf.ifft2d(ifftshift2d_tf(u), name='ifft_as_c%s' % c)) * (factor * factor)

    # remove padding and multiply third phase factor
    u_out = u[n_pad:-n_pad, n_pad:-n_pad] * q3
    # u_out = u * q3
    # u_out = u * q3

    return u_out


# padded version of ang_spec_prop assuming u_in has a padding of "padding"
def ang_fresnel_propagation_schmidt_padded(u_in, z, ds, do, r1sq, wavelength, res, padding):
    c = WAVELENGTHS.index(wavelength)
    wvl = tf.convert_to_tensor(wavelength, dtype=tf.float32)
    # padding to reduce artifacts, quarter resolution seems to be enough
    delta1 = ds
    delta2 = do
    n = res
    n_pad = padding
    n_padded = n + 2 * n_pad

    # wavenumber k
    k = 2 * np.pi / wvl

    # source coordinates are represented by r1sq

    # spatial frequencies of source plane
    delta_freq = 1 / (delta1 * n_padded)
    coordsf = tf.linspace(-n_padded // 2 * delta_freq, (n_padded // 2 - 1) * delta_freq, n_padded,
                          name='linspace_freqCoords_c%s' % c)
    fx1, fy1 = tf.meshgrid(coordsf, coordsf, name='meshgrid_freqCoordinates_c%s' % c)
    fsq = fx1 * fx1 + fy1 * fy1

    # scaling parameter
    m = delta2 / delta1

    # observation plane coordinates
    coords2 = tf.linspace(-n_padded // 2 * delta2, (n_padded // 2 - 1) * delta2, n_padded,
                          name='linspace_r2Coords_c%s' % c)
    x2, y2 = tf.meshgrid(coords2, coords2, name='meshgrid_r2Coordinates_c%s' % c)
    r2sq = x2 * x2 + y2 * y2

    # quadratic phase factors
    q1 = tf_exp_complex(k / 2. * (1. - m) / z * r1sq, name='phase_q1_c%s' % c)
    q2 = tf_exp_complex(np.pi * wavelength * -1.0 * z / m * fsq, name='phase_q2_c%s' % c)
    q3 = tf_exp_complex(k / 2. * (m - 1.) / (m * z) * r2sq, name='phase_q3_c%s' % c)

    # compute output field at observation plane
    u = tf.multiply(u_in / m, q1, name='mult_q1_c%s' % c)
    # perform fft
    u_fft = fftshift2d_tf(tf.fft2d(fftshift2d_tf(u), name='fft_as_c%s' % c)) * (delta1 * delta1)
    u = u_fft * q2
    # transform back
    factor = n_padded * delta_freq
    u = ifftshift2d_tf(tf.ifft2d(ifftshift2d_tf(u), name='ifft_as_c%s' % c)) * (factor * factor)

    # remove padding and multiply third phase factor
    # u_out = u[n_pad:-n_pad, n_pad:-n_pad] * q3
    u_out = u * q3

    return u_out


def reverse_view_relation(prediction_wide_input):
    """This method reverses the view relation: cubic upscale to 2048, then crop the central square with INSHAPE dimension
    for the narrow view"""
    # wide_size = tf.shape(prediction_wide_input)
    # wide_size = tf.cast(wide_size, tf.float32)
    # wide_resize_mat = tf.constant([1,2048/INSHAPE, 2048/INSHAPE,1])
    # wide_size = tf.multiply(wide_size, wide_resize_mat)
    # wide_size = tf.cast(wide_size, tf.int32)
        
    # prediction_wide_2D = prediction_wide[0,:,:,0]
    # prediction_wide_2D = tf.reshape(prediction_wide_2D, [INSHAPE,INSHAPE])
    expected_original_from_wide = tf.image.resize_bicubic(prediction_wide_input, (INSHAPE*2, INSHAPE*2)) # estimate the original view from the wide view
    crop_ratio = 0.5
    expected_long_from_wide = tf.image.central_crop(expected_original_from_wide, crop_ratio) # estimate the narrow view from the original view
    
    return expected_long_from_wide

# ------------------------------------
# Model
# ------------------------------------

class Model(ModelDesc):

    # define inputs, incoming and expected are 3 channel rgb each, mosaics are binary masks for each submosaic
    def inputs(self):
        return [tf.TensorSpec((None, INSHAPE, INSHAPE, CHANNELS), tf.float32, 'expected_wide'),
                tf.TensorSpec((None, INSHAPE, INSHAPE, CHANNELS), tf.float32, 'expected_long'),
                tf.TensorSpec((None,), tf.float32, 'distance')]

    @auto_reuse_variable_scope
    def _encoder(self, incoming_wide, incoming_long, depth):
        """Each incoming image is encoded into an image on the sensor simulating the image formation process in the
            camera. Assuming monochromatic light for now.
        """

        with tf.name_scope('optics_sim'):
            # Variable representing optical system
            # basically a mask that blends between the two optical configurations (wide fov and double fov)
            init = tf.constant(0.5, tf.float32)
            blend_mask_factor = tf.get_variable('a_blendmask', initializer=init)
            # tf.add_to_collection("SLM_coefficients", blend_mask_factor)

            # output slm pattern to tensorboard
            blend_mask = tf.multiply(blend_mask_factor, tf.ones([INSHAPE, INSHAPE], tf.float32))
            viz_phi = tf.expand_dims(blend_mask, -1)
            viz_phi = tf.expand_dims(viz_phi, 0)
            viz_phi = viz_phi * 255.0
            viz_slm = tf.cast(tf.clip_by_value(viz_phi, 0, 255), tf.uint8, name='slmviz')
            tf.summary.image('SLM_pattern', viz_slm, max_outputs=1)

            # add uniform noise to SLM mask (to simulate tolerances), according to paper sec. 3.2
            unoise = tf.random_uniform(shape=tf.shape(blend_mask), minval=-.01, maxval=0.01, dtype=tf.float32,
                                       name='uniformnoise_SLM')
            phi_slm = blend_mask + unoise

            # crop the variable grid to the aspect ratio of the SLM and apply aperture mask
            if SLM_CROP:
                x_length = APERTURE
                y_length = APERTURE * ASPECT_FACTOR
                crop_aspect = tf_rect(x_prime, y_prime, x_length, y_length)
                crop = crop_aspect  # * aperture

            # define available range as 0 to 1 which corresponds to a range of 2pi, centered aroung 0
            phi_slm = tf.multiply(phi_slm, 2 * np.pi, name='shift_range') - np.pi

            # distance d is randomly drawn from the predefined list of distances, one for each batch
            # at inference time we will check if there is a distance given
            if get_current_tower_context().is_training:
                d_idx = tf.random_uniform([1], 0.0, 1.0, dtype=tf.float32)
                d_idx = tf.cast(d_idx * (len(DISTANCES) - 1), tf.int32, name='cast_distanceIdx')
                distance_tensor = tf.convert_to_tensor(DISTANCES)
                distance = tf.gather(distance_tensor, d_idx, name='gather_d')
            else:
                d_idx = tf.random_uniform([1], 0.0, 1.0, dtype=tf.float32)
                d_idx = tf.cast(d_idx * (len(DISTANCES) - 1), tf.int32, name='cast_distanceIdx')
                distance_tensor = tf.convert_to_tensor(DISTANCES)
                distance = tf.cond(tf.equal(depth[0], 0.0), lambda: tf.gather(distance_tensor, d_idx, name='gather_d'),
                                   lambda: depth[0], name='cond_distance')
                # distance = tf.Print(distance, [distance])
            # tell tf the shape of the distance in case it got lost during the conditional
            distance.set_shape(1)
            # tensorboard output
            add_moving_summary(distance[0])

            # create encoded image as mix of the two field of views
            blend_mask = tf.reshape(blend_mask, [1, INSHAPE, INSHAPE, 1])
            coded_image = tf.add((1.0 - blend_mask) * incoming_wide, blend_mask * incoming_long, name='add_codedImage')

        # add gaussian noise to output (to simulate sensor noise), according to paper sec. 3.1.3, eq. 6
        gnoise = tf.random_normal(shape=tf.shape(coded_image), mean=0.0, stddev=tf.random_uniform([1], 0.0, 0.012,
                                                                                                  dtype=tf.float32),
                                  dtype=tf.float32, name='gaussian_noise')
        activations = coded_image + gnoise

        return activations

    @auto_reuse_variable_scope
    def _decoder(self, enc_activations):
        """Decode into final image pair (2 times the CHANNELs output)
            Decode with wide convolutional network, architecture based on u-net by Ronneberger et al.
        """
        # deactivated
        # adding one iteration of wiener deconvolution to the input of neural network, learning noise parameter
        # gamma = tf.get_variable('gamma_decoder', [1, 1], initializer=tf.zeros_initializer)
        # dir_deconv = wiener_deconvolution_tf(enc_activations[:, :, :, 0:CHANNELS],
        #                                      enc_activations[0, :, :, CHANNELS], gamma)
        # enc_activations = ConcatWith('concat_dirDeconv', enc_activations, [dir_deconv], 3)

        # enc_activations = enc_activations * 2.0 - 1.0
        with tf.name_scope('decoder'):
            dec_out = reconstruction.unet5(enc_activations, 2 * CHANNELS, FEATURE_FACTOR)
            return dec_out

    def build_graph(self, *inputs):
        expected_wide, expected_long, depth = inputs
        # [0, 255] -> [-1, 1]
        # create zernike polynomials deactivated
        # print 'generating zernike matrix...'
        # z = tf.convert_to_tensor(synthesize_zernike(SLM_RES, 2, ZERNIKE_MODES), dtype=tf.float32)
        # connect encoder decoder
        activations = self._encoder(expected_wide, expected_long, depth)
        prediction = self._decoder(activations)
        prediction_wide = prediction[:, :, :, 0:CHANNELS]
        prediction_long = prediction[:, :, :, CHANNELS:]
        # downres test
        # expected = expected[:, 15:497, 15:497, :]
        # incoming = incoming[:, 15:497, 15:497, :]
        # compute loss: mean-squared-error as reconstruction error
        
        # compute the expected narrow view from the expected wide view, to enforce relation between the narrow view and the wide view
        # prediction_wide_input = tf.reshape(prediction_wide, [INSHAPE, INSHAPE])
        # prediction_wide_input = tf.Print(prediction_wide_input, [tf.shape(prediction_wide_input)])
        # expected_long_from_wide = reverse_view_relation(prediction_wide_input)
        prediction_long_from_wide = reverse_view_relation(prediction_wide)
        viz_prediction_long_from_wide = prediction_long_from_wide * 255.0
        viz_prediction_long_from_wide = tf.cast(tf.clip_by_value(viz_prediction_long_from_wide, 0, 255), tf.uint8, name='viz_expected_long_from_wide')
        tf.summary.image('expected_long_view_from_wide', viz_prediction_long_from_wide, max_outputs=1)
       
        # data fitting
        cost = tf.add(tf.reduce_mean(tf.squared_difference(prediction_wide, expected_wide), name="mse_wide"),
                    tf.reduce_mean(tf.squared_difference(prediction_long, expected_long), name="mse_long"), name='mse_data_fitting')
        
        # regularizer of inter-view relation, part 2: predicted narrow view vs expected narrow view
        inter_view_weight = tf.get_variable('inter_view_weight', initializer=float(1e-3), trainable=False)
        cost = tf.add(inter_view_weight * tf.reduce_mean(tf.squared_difference(expected_long, prediction_long_from_wide), name="mse_long_from_wide_expected"), 
                      (1-inter_view_weight) * cost, name='mse')

        # just for TensorBoard track peak-signal-to-noise ratio
        with tf.name_scope('psnr'):
            # [-1, 1] -> [0, 255]
            # prediction_scaled = 128.0 * (1.0 + prediction)
            prediction_scaled = tf_linear2srgb(prediction)
            # expected_scaled = 128.0 * (1.0 + expected)
            expected_wide_scaled = tf_linear2srgb(expected_wide)
            expected_long_scaled = tf_linear2srgb(expected_long)
            # incoming_scaled = 128.0 * (1.0 + incoming)
            # incoming_scaled = tf_linear2srgb(incoming)

            prediction_psnr = tf.reduce_mean(psnr(prediction_scaled[:, :, :, 0:CHANNELS],
                                                  expected_wide_scaled, maxp=255., name="prediction_wide") +
                                             psnr(prediction_scaled[:, :, :, CHANNELS:],
                                                  expected_long_scaled, maxp=255., name="prediction_wide"),
                                             name='prediction')
            # incoming_psnr = psnr(incoming_scaled, expected_scaled, maxp=255., name="incoming")
            prediction_ssim = tf.reduce_mean(tf.reduce_mean(tf.image.ssim(prediction_scaled[:, :, :, 0:CHANNELS],
                                                            expected_wide_scaled, max_val=255), name='ssim_wide') +
                                             tf.reduce_mean(tf.image.ssim(prediction_scaled[:, :, :, CHANNELS:],
                                                                          expected_long_scaled, max_val=255),
                                                            name='ssim_long'), name='ssim')
            # psnr_improvement = tf.divide(prediction_psnr, incoming_psnr, name="improvement")

        add_moving_summary(prediction_psnr, prediction_ssim)

        # use tensorboard for visualization
        with tf.name_scope("visualization"):
            # psf_viz = activations[:, :, :, CHANNELS:]
            # psf_viz = psf_viz / tf.reduce_max(psf_viz)
            # psf_viz = tf.tile(psf_viz, [1, 1, 1, CHANNELS], name='tile_psf_viz')
            # difference = tf.abs(prediction - expected)
            # difference = tf.divide(difference, tf.reduce_max(difference), name='normalize_difference')
            viz = (tf.concat([expected_wide, prediction[:, :, :, 0:CHANNELS], expected_long,
                              prediction[:, :, :, CHANNELS:(CHANNELS*2)],
                              activations[:, :, :, 0:CHANNELS]], 2)) * 255.0
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            pred_export = tf.clip_by_value(prediction * 255.0, 0, 255)
            pred_export = tf.cast(pred_export, tf.uint8, name='export_viz')
#        tf.summary.image('expected_wide, estimate_wide, expected_long, estimate_long, encoderOut', viz,
#                         max_outputs=max(3, BATCH))
        tf.summary.image('expected_wide', viz[:,:,:INSHAPE,:], max_outputs=BATCH)
        tf.summary.image('estimate_wide', viz[:,:,INSHAPE:2*INSHAPE,:], max_outputs=BATCH)
        tf.summary.image('expected_long', viz[:,:,2*INSHAPE:3*INSHAPE,:], max_outputs=BATCH)
        tf.summary.image('estimate_long', viz[:,:,3*INSHAPE:4*INSHAPE,:], max_outputs=BATCH)
        tf.summary.image('encoderOut', viz[:,:,4*INSHAPE:,:], max_outputs=BATCH)
        return cost

    # ADAM optimizer with settings according to ??
    def optimizer(self):
        # see paper sec 2.3
        lr = tf.get_variable('learning_rate', initializer=float(5e-5), trainable=False)
        tf.summary.scalar('learning_rate' + '-summary', lr)
        return tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # momentum optimizer with settings from paper sec 3.2
        # lr = symbolic_functions.get_scalar_var('learning_rate', 5e-3, summary=True)
        # return tf.train.MomentumOptimizer(lr, 0.5, name='MomentumOPtimizer', use_nesterov=True)


# ----------------------------------------
# DATA
# ----------------------------------------

# adds poisson noise to the image that simulates a low light noisy image (few photons), peak = 1 is really noisy
def add_poisson_noise(img, peak):
    return np.random.poisson(img * 255.0 * peak) / peak / 255

#TODO:
#def downsample(original_view):
#    """This method downsamples the original view to be the narrow view."""
#    # parameters: original_view resolution: 2000x2000
#    # original view angle: 50x50 degree 
#    # cropped view angle: 20x20 degree 
#    # narrow view resolution: 800x800 
#    crop_resolution = np.array([800, 800])
#    # crop original view 
#    center_ind = original_view.shape()/2
#    narrow_view = original_view[center_ind[0]-crop_resolution[0]/2:center_ind[0]+crop_resolution[0]/2, 
#                                center_ind[1]-crop_resolution[1]/2:center_ind[1]+crop_resolution[1]/2]
#    return narrow_view

# class for Poisson Noise with a peak value drawn from uniform random distribution between 0 and max_peak
# Todo: make augmentor work
class PoissonNoise(imgaug.ImageAugmentor):
    # pick random stdev from a given set of sigmas
    def __init__(self, max_peak=0.1):
        super(PoissonNoise, self).__init__()
        self.peak_size = np.maximum(max_peak, 0.2)
        self._init(locals())

    def _get_augment_params(self, img):
        return self.rng.uniform(*img.shape)

    def _augment(self, img):
        peak = self.rng() * self.peak_size

        return np.clip(np.random.poisson(img * 255.0 * peak) / peak / 255, 0, 255)


def get_data(subset='train'):
    assert subset in ['train', 'test', 'export']
    isTrain = (subset == 'train')

    # load dataset using load_data
    # load data from lmdb file
    if isTrain:
        data_path = TRAIN_PATH
    elif subset == 'export':
        data_path = TEST_PATH
    else:
        data_path = VAL_PATH

    ds = load_data.loaddata(data_path)
    if isTrain:
        # for training, we augment the data by adding Poisson noise, do random crop and rotation and flips
        augmentors = [
            # imgaug.Brightness(30, clip=False),
            # imgaug.Contrast((0.4, 1.5)),
            imgaug.ResizeShortestEdge(2048, cv2.INTER_CUBIC),
            imgaug.RandomCrop([int(INSHAPE*2), int(INSHAPE*2)]),
            imgaug.Flip(horiz=True),
            imgaug.Flip(vert=True),
            imgaug.Rotation(max_deg=180.0, center_range=(0.5, 0.5), step_deg=90.0)
        ]
    else:
        # for testing, we just use the same data over and over
        augmentors = [
            imgaug.ResizeShortestEdge(2048, cv2.INTER_CUBIC),
            imgaug.CenterCrop([int(INSHAPE*2), int(INSHAPE*2)]),
        ]
    # augment all data
    ds = AugmentImageComponents(ds, augmentors, index=[0])

    # for non noise case, just copy datapoint, add distance
    if subset == 'export':
        ds = MapData(ds, lambda dp: [dp[0], dp[0]])
    else:
        ds = MapData(ds, lambda dp: [dp[0], dp[0], 0.])
    # ds = AugmentImageComponents(ds, [PoissonNoise(max_peak=0.2)], index=0)
    # resize expected datapoints to resemble a wide fov and a zoomed in view (double magnitude
    ds = AugmentImageComponent(ds, [imgaug.ResizeShortestEdge(INSHAPE, cv2.INTER_AREA)], index=0)
    ds = AugmentImageComponent(ds, [imgaug.CenterCrop([INSHAPE, INSHAPE])], index=1)
    # create batches
    ds = PrintData(ds)
    ds = BatchData(ds, BATCH)
    # or use PrefetchData instead (PrefetchDataZMQ is faster!)
    if subset == 'export':
        return ds
    else:
        ds = PrefetchDataZMQ(ds, 8 if isTrain else 6, hwm=30)

        return ds


# ----------------------------------------
# CONFIG
# ----------------------------------------


# Callback to clip the coefficients to the positive domain
class ClipCallback(Callback):
    def _setup_graph(self):
        varlist = tf.get_collection("SLM_coefficients")
        ops = []
        for v in varlist:
            n = v.op.name
            logger.info("Clip {}".format(n))
            ops.append(tf.assign(v, tf.clip_by_value(v, 0.0, 1.0)))
        self._op = tf.group(*ops, name='clip')

    def _trigger_step(self):
        self._op.run()


# Callback shuffle variables around
class ShuffleVariable(Callback):
    def _setup_graph(self):
        varlist = tf.get_collection("SLM_coefficients")
        ops = []
        for v in varlist:
            n = v.op.name
            logger.info("Shuffle {}".format(n))
            ops.append(tf.assign(v, fftshift(v, SLM_RES)))
        self._op = tf.group(*ops, name='clip')

    def _trigger(self):
        print('shuffling tensor')
        self._op.run()


# Callback to save final SLM pattern to disk
class OutputSLM(Callback):
    def _setup_graph(self):
        self.varlist = tf.get_collection("SLM_coefficients")

    def _after_train(self):
        for var in self.varlist:
            pattern = self.trainer.sess.run(var)
            output = "./image_SLM45k_blendmask_multifocal_unet.png"
            pattern = np.uint8(np.clip(pattern * 255, 0, 255))
            cv2.imwrite(output, pattern)


# Callback to write out test data, triggered after training
class TestDataExport(Callback):
    def __init__(self):
        # setup data flow for test
        self.test_ds = get_data('export')
        self.test_ds.reset_state()
        self.distances = [50, 5, 3.8, 2, 1, 0.8, 0.75, 0.5, 0.45]

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(['incoming', 'expected_wide', 'expected_long', 'distance'],
                                               ['visualization/viz', 'psnr/prediction'])

    def _trigger(self):
        print('Generating test data...')
        idx = 0
        psnr_accum = np.zeros(len(self.distances))
        for input, expected_w, expected_l in self.test_ds.get_data():
            # for idx in range(0, MAX_TEST):
            #    input, expected = self.test_ds.__iter__()
            idx_d = 0
            for d in self.distances:
                outputs = self.pred(input, expected_w, expected_l, (float(d),))
                im = outputs[0]  # * 256.0
                # write image to disk
                dst = os.path.join(EXPORT_PATH, 'd_%s' % d)
                if not os.path.exists(dst):
                    os.makedirs(dst)
                output = "/".join([dst, "image_%s.png" % idx])
                cv2.imwrite(output, im[0])
                psnr_accum[idx_d] = psnr_accum[idx_d] + outputs[1]
                idx_d += 1
            if 0 < MAX_TEST <= idx:
                break
            idx += 1
        psnr_accum = psnr_accum / float(MAX_TEST)
        # create and save psnr plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.xticks(list(range(0, len(self.distances))), self.distances)
        ax.bar(list(range(0, len(self.distances))), psnr_accum, color='c')
        ax.set_xlabel('Distances (depth)')
        ax.set_ylabel('average psnr')
        fig.savefig(EXPORT_PATH + '/psnr_distance_plot.pdf', bbox_inches='tight')
        plt.close(fig)
        # write out psnr values
        with open(EXPORT_PATH + '/psnr.txt', 'w') as f:
            f.write(str(psnr_accum)[1:-1])


def get_config():
    # using a fixed logging dir?
    if LOGDIR:
        logger.set_logger_dir(LOGDIR)
    else:
        logger.auto_set_dir()
    dataset_train = get_data('train')
    dataset_val = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ClipCallback(),
            SessionRunTimeout(15000),
            # during validation-phase on test data
            InferenceRunner(dataset_val, [ScalarStats(['mse:0', 'psnr/prediction:0', 'psnr/ssim:0'])]),
            # using a cyclic learning rate here which is helpful since we don't use batchnorm
            ScheduledHyperParamSetter('learning_rate', [(15, 2e-4), (60, 1e-4), (95, 1e-4)]),
            # export test data every 20 epochs
            # PeriodicTrigger(TestDataExport(), every_k_epochs=50)
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            # OutputFilters(),
            ProgressBar(['mse:0', 'psnr/prediction:0']),
            MergeAllSummaries(),
            OutputSLM()
        ],
        model=Model(),
        steps_per_epoch=100,
        max_epoch=50,
        starting_epoch=1
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()
    # use_global_argument(args)
    return args


if __name__ == '__main__':
    """
    # train from scratch
    python psfDesign_train.py --gpu 0
    # re-start previous checkpoint
    python psfDesign_train.py --gpu 0 --load ../model/train_log/psfDesign....../checkpoint
    """   
    args = get_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    launch_train_with_config(
        config, # SimpleTrainer()) 
        SyncMultiGPUTrainer(max(get_num_gpu(), 1)))
