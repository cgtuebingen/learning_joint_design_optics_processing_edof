#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Andreas Engelhardt
# Framework for Simulation of SLM driven computational camera applications and reconstruction v0.1
# roughly based on Sitzmann et al. "End-to-end Optimization of Optics and Image Processing for
# Achromatic Extended Depth of Field and Super-resolution Imaging"
# http://www.computationalimaging.org/publications/end-to-end-optimization-of-optics-and-image-processing-for-achromatic-extended-depth-of-field-and-super-resolution-imaging/

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
from tensorpack.utils.gpu import get_nr_gpu
# import lucyrichardson_deconvolution
import reconstruction
import cv2
import os

"""This script performs inference on a given image and model checkpoint. Application: EDOF
usage: psfDesign_infer.py --input <path/to/input/image.jpg> --output <output/image.jpg> --model <path/to/modeldir> 
--d <distance in m>"""

# ------------------------------------
# Global Settings
# ------------------------------------

BATCH = 1                # number of sequences per update
SHAPE = 4600  # dimension simulation grid
INSHAPE = SHAPE // 5  # dimension of input shape if different from shape
FEATURE_FACTOR = 1  # scaling factor for feature number in reconstruction CNN (a factor of 2 cuts filters in half))

# ZERNIKE_MODES = 2        # used bands of Zernike polynomials
SENSOR_RES = SHAPE  # Resolution of sensor (one dimension) assuming a square sensor, so actual resolution is square
# SENSOR_PITCH = 8.0e-6  # Pixel distance in m - using the pitch of the SLM pixels here for now
SLM_RES = [2300, 1294]  # Resolution of SLM, gets padded to a square of the first dimension,
L_SLM = 15.36e-3  # length of longer size of SLM in m, assuming 16:9 aspect for SLM here
ASPECT_FACTOR = 0.5625  # factor to represent aspect ratio

PAD = 200       # padding for propagations, experimentally evaluated for depth range and grid spacings

F_NUMBER = 5.6  # aperture opening 1/stops
L_SENSOR = 3.45e-6 * 4096  # length of one size of camera sensor in m, assuming square of longer sides (if non-sqaure)
FOCAL_LENGTH = .1  # focal length in m of the fixed lens element
FOCUS_DISTANCE = 2.5  # distance of focus plane from pupil of optical element in m, -1 for infinity

# distance between optical elements and image plane (z)
if FOCUS_DISTANCE == -1:
    Z_DISTANCE = FOCAL_LENGTH
else:
    Z_DISTANCE = np.abs((FOCUS_DISTANCE * FOCAL_LENGTH) / (FOCUS_DISTANCE - FOCAL_LENGTH))

Z2_DISTANCE = 0.147  # distance between SLM and lens in m - this is an approximated, measured value

# diameter of aperture (region of SLM)
APERTURE = FOCAL_LENGTH / F_NUMBER  # effective aperture of the lens (diameter of entrance pupil)
# circular aperture is 8.64e-3 (L_SLM), rectangular 12.219e-3

N_LAYERS = 8  # number of convolution layers in decoder network, must be multple of 2, added one
# additional layer for output so it's currently N +1 conv layers
CHANNELS = 1  # image channels used, e.g. 1 for greyscale, 3 for rgb
WAVELENGTHS = [5.2e-7]  # wavelengths corresponding to channels in m
CURVATURE_APPROX = False  # whether to use fresnel approximation of incident spherical wavefront, default is on
RECT_APERTURE = False
SLM_CROP = True
RND_WVL = False  # a wavelength is chosen from the list randomly to enable multi spectral training at no extra memory cost
# strength of random perspective transform applied to psf, off if 0, adequate values are between 0.01 and 0.2
RND_TRANSFORM = 0.0
# factor by which an unmodulated version of the psf is blended into the final psf to simulate zeroth order difftaction
# peak
ZEROTH_ORDER = 0.00

# set a fixed log dir outside of repository
LOGDIR = './../model_test/'


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
    print 'creating perspective transforms...'

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
    im_srgb = im.astype(np.float32)
    alpha = 0.055
    im_srgb[im <= 0.0031308] *= 12.92
    bool_mask = im > 0.0031308
    im_srgb[bool_mask] = im_srgb[bool_mask] ** (1.0 / 2.4) * (1.0 + alpha) - alpha

    return im_srgb * 255.0


def srgb2linear(im):
    im = im.astype(np.float32)
    im /= 255.0
    im_linear = im
    alpha = 0.055
    im_linear[im <= 0.04045] *= 0.077399381
    bool_mask = im > 0.04045
    im_linear[bool_mask] = ((im_linear[bool_mask] + alpha) / (1 + alpha)) ** 2.4

    return np.clip(im_linear, 0, 1)


# crops input image to the center square of 'shape' pixels
def center_crop_image(image, shape):
    dims = image.shape
    h0 = int((dims[0] - shape) * 0.5)
    w0 = int((dims[1] - shape) * 0.5)
    # center = [int(dims[0] * 0.5), int(dims[1] * 0.5)]
    return image[h0:h0+shape, w0:w0+shape]


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
    circle = np.array(r < d/2.0, dtype=float)
    circle[r == d/2.0] = 0.5
    return circle


# creates a circle like circ() above but with tensorflow ops, no anti-aliasing
def tf_circ(x, y, d):
    r = tf.sqrt(x * x + y * y)
    circle = tf.cast(r <= d/2.0, dtype=tf.float32)
    # mask = r == d/2
    # circle = tf.where(mask, tf.constant(0.5, shape=[SLM_RES, SLM_RES], dtype=tf.float32), circle, name='cond2_circle')
    return circle


    # creates a rectangular coordinate grid from vector x and side length d
def tf_rect(x, y, d1, d2):
    x = tf.abs(x)
    y = tf.abs(y)
    y1 = tf.cast(x <= d1/2.0, dtype=tf.float32)
    y2 = tf.cast(y <= d2/2.0, dtype=tf.float32)
    return tf.multiply(y1, y2)


# convert cartesian to polar coordinates (numpy arrays only)
def cart2pol(x, y):
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return theta, rho


# performs convolution in fourier space using tensorflow ops, Todo: register as layer?
# assuming real inputs of type float32, outputs a real tensor of type float32
def conv_fft2(x, k, x_size, k_size):
    csize = x_size + k_size  # - 1
    # padding kernel
    k = tf.pad(k, tf.constant([[int((csize - k_size) / 2), int((csize - k_size) / 2)],
                               [int((csize - k_size) / 2), int((csize - k_size) / 2)]]), "CONSTANT")

    # expanding kernel for batch size
    k = fftshift2d_tf(k)
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


# computes the complex exponential with the help of Euler's formula since there is no cuda kernel for it in (older) tf
def tf_exp_complex(phase, name):
    return tf.complex(tf.cos(phase), tf.sin(phase), name=name)


# method to downsample a 2D tensor by a factor using area interpolation (averaging)
def area_downsample(image, factor, name):
    factor = int(factor)
    output = tf.nn.avg_pool(image, [1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)
    return output


# upsample a tensor by a factor of two - the reverse of area downsample
def upsample_neareast(image, factor, name):
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


# fft shift 2D version based on https://github.com/vsitzmann/deepoptics/blob/master/src/layers/optics.py
# these enable GPU usage since there is no GPU kernel for roll
def fftshift2d_tf(a_tensor):
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


def ifftshift2d_tf(a_tensor):
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


# fftshift equivalent for tensorflow, shifts the values so that zero frequency is in center
def fftshift_batch(f, res):
    fshift = f
    for dim in range(0, CHANNELS + 1):
        fshift = tf.manip.roll(fshift, shift=int(res / 2), axis=dim + 1)

    return fshift


def fftshift(f, res):
    fshift = f
    for dim in range(0, CHANNELS + 1):
        fshift = tf.manip.roll(fshift, shift=int(res / 2), axis=dim)

    return fshift


# ifftshift equivalent for tensorflow, shifts the values so that zero frequency is in center
def ifftshift(f, res):
    fshift = f
    for dim in reversed(xrange(CHANNELS + 1)):
        fshift = tf.manip.roll(fshift, shift=int(res / 2), axis=dim)

    return fshift


# ifftshift equivalent for tensorflow, shifts the values so that zero frequency is in center
def ifftshift_batch(f, res):
    fshift = f
    for dim in reversed(xrange(CHANNELS + 1)):
        fshift = tf.manip.roll(fshift, shift=int(res / 2), axis=dim + 1)

    return fshift


# compute modulo on GPU
def modulo_tf(a, b):
    return a - tf.floor_div(a, b) * b


# calculates a no-reference sharpness metric based on frequency analysis
# according to https://www.sciencedirect.com/science/article/pii/S1877705813016007
def fm(im):
    total = im.shape[0] * im.shape[1]
    fm = 0.0
    for i in range(CHANNELS):
        f_im = np.fft.fftshift(np.fft.fft2(im[:, :, i]))
        f_im = np.abs(f_im)
        m = np.max(f_im)
        number = np.count_nonzero(f_im > m / 1000)
        fm = fm + float(number)/total
        print fm
    fm = np.mean(fm)
    return fm * 10.0


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
    #u_out = ifftshift(tf.ifft2d(ifftshift(uout_spec, n), name='ifft2d_uout_c%s' % c), n) * \
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

    # window to avoid anti-aliasing, Todo: improve!
    # u_in = u_in * tf.complex(window, window)

    # compute output field at observation plane
    u = tf.multiply(u_in / m, q1, name='mult_q1_c%s' % c)
    # pad field for convolution
    # u_padded = pad2d_gpu(u, n_pad)
    # u_padded = tf.pad(u, [[n_pad, n_pad], [n_pad, n_pad]], 'CONSTANT', name='pad_u_c%s' % c)
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


# get arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to image')
    parser.add_argument('--output', help='path to result (to be written)')
    parser.add_argument('--model', help='path to model directory')
    parser.add_argument('--d', help='depth when testing EDOF')
    args = parser.parse_args()
    return args


# Model description
def _encoder(incoming, depth):
    """Each incoming image is encoded into an image on the sensor simulating the image formation process in the
        camera. Assuming monochromatic light for now.
    """

    with tf.name_scope('optics_sim'):
        # aperture coordinates, center point is slightly shifted since dimensions are multiple of 2
        # sample rate
        padding = PAD
        delta = L_SLM / float(SHAPE)
        # extending the dimensions as a form of padding
        n = SHAPE + 2 * padding
        ext_length = L_SLM + delta * (padding * 2)
        delta = ext_length / float(n)
        coords = tf.linspace(float(- n / 2), float(n / 2 - 1), n)
        coords = tf.multiply(coords, delta, name='mult_coords1_delta')
        x, y = tf.meshgrid(coords, coords, name='meshgrid_apertureCoordinates')

        # phase term for propagation between SLM and lens
        squared_sum = x * x + y * y

        delta1 = APERTURE / float(SHAPE)
        ext_length2 = APERTURE + delta1 * (padding * 2)
        delta1 = ext_length2 / float(n)
        delta_inv = 1 / delta
        x_prime = x * delta_inv * delta1
        y_prime = y * delta_inv * delta1

        # phase term for propagation between optical system and image plane, distance z
        squared_sum1 = x_prime * x_prime + y_prime * y_prime

        # ::fresnel propagation for distance z between optical element and sensor, using angular spectrum method::
        delta2 = L_SENSOR / SHAPE
        ext_length3 = L_SENSOR + delta2 * (2 * padding)
        delta2 = ext_length3 / float(n)

        # create aperture grid
        if RECT_APERTURE:
            aperture = tf_rect(x_prime, y_prime, APERTURE, APERTURE)
        else:
            aperture = tf_circ(x_prime, y_prime, APERTURE)

        # Optical Element::SLM
        # phase term for SLM in range [0, 2pi], generated from Zernike polynomials
        # phi_slm = tf.get_variable('a_slm', initializer=tf.random_uniform((SLM_RES, SLM_RES), 0, 1))
        # data format is HW!
        phi_slm = tf.get_variable('a_slm', [SLM_RES[1], SLM_RES[0]], initializer=tf.zeros_initializer)
        tf.add_to_collection("SLM_coefficients", phi_slm)

        slm_viz = phi_slm

        # add uniform noise to SLM mask (to simulate tolerances), according to paper sec. 3.2
        unoise = tf.random_uniform(shape=tf.shape(phi_slm), minval=-.01, maxval=0.01, dtype=tf.float32,
                                   name='uniformnoise_SLM')
        # phi_slm = phi_slm + unoise

        # crop the variable grid to the aspect ratio of the SLM and apply aperture mask
        if SLM_CROP:
            x_length = APERTURE
            y_length = APERTURE * 0.5625
            crop_aspect = tf_rect(x_prime, y_prime, x_length, y_length)
            crop = crop_aspect  # * aperture

        # define available range as 0 to 1 which corresponds to a range of 2pi
        phi_slm = tf.multiply(phi_slm, 2 * np.pi, name='shift_range') - np.pi

        # not necessary
        # phi_slm = tf.mod(phi_slm, 2*np.pi, name='mod_2pi')

        distance = tf.convert_to_tensor(depth)

        # pad and scale
        pad = int((SLM_RES[0] - SLM_RES[1]) * 0.5) + padding // 2
        phi_slm = tf.pad(phi_slm, tf.constant([[pad, pad], [padding // 2, padding // 2]]), "CONSTANT")
        phi_slm = upsample_neareast(phi_slm, 2, 'upsample_slm')

        # upsample incoming image to SHAPE x SHAPE
        if INSHAPE < SHAPE:
            incoming = upsample_neareast(incoming, 5, 'upsample_incoming')

        # wavelength dependent operations...
        coded_image = []
        psfs = []
        # loop over all wavelengths - if too many we can do a stochastic version instead?
        for c in range(0, CHANNELS):

            # phase term for propagation from (virtual) object point source(s) to optical system, distance d
            # paraxial approximation of spherical wave emitted by a point source at distance z from optical element
            # choose a wavelength randomly
            if RND_WVL:
                wvl_idx = tf.random_uniform([1], 0.0, 1.0, dtype=tf.float32)
                wvl_idx = tf.cast(wvl_idx * (len(WAVELENGTHS) - 1), tf.int32, name='cast_wvlIdx')
                wavelengths_tensor = tf.convert_to_tensor(WAVELENGTHS)
                wvl = tf.gather(wavelengths_tensor, wvl_idx, name='gather_wvl')
            else:
                wvl = tf.convert_to_tensor(WAVELENGTHS[c], dtype=tf.float32, name='wavelength_c%s' % c)

            if CURVATURE_APPROX:
                phi_d = np.pi / (wvl * distance) * squared_sum
                field_o = tf_exp_complex(phi_d, 'exp_phase_o_c%s' % c)
            else:
                # to prevent overflow / loss of precision for long distances and short wavelength
                # process in float64 and work with modulo 2pi with lower precision afterwards
                squared_sum2 = tf.cast(squared_sum, dtype=tf.float64)
                distance = tf.cast(distance, dtype=tf.float64)
                r = tf.sqrt(squared_sum2 + distance * distance, name='sqrt_r_c%s' % c)
                k = tf.cast(2 * np.pi / wvl, dtype=tf.float64)
                phi_d = modulo_tf(k * r, 2. * np.pi)
                phi_d = tf.cast(phi_d, dtype=tf.float32)
                field_o = tf_exp_complex(phi_d, 'exp_phase_o_c%s' % c)

            # crop the variable grid to the aspect ratio of the SLM and apply aperture mask
            if SLM_CROP:
                crop = tf.complex(crop, crop)
                field_o = tf.multiply(field_o, crop, name='mult_field_0_crop_c%s' % c)

            # OpticalElement::SLM
            # if zeroth order diffraction peak is to be simulated, create an unmodulated field as well
            if ZEROTH_ORDER > 0:
                field_unmodulated = field_o
                field_o = tf_exp_complex(phi_slm, 'exp_phase_o_c%s' % c) * field_o
            else:
                field_o = tf_exp_complex(phi_slm, 'exp_phase_o_c%s' % c) * field_o

            # ::Propagation between SLM and lens::
            field_o = ang_fresnel_propagation_schmidt_padded(field_o, Z2_DISTANCE, delta, delta1, squared_sum,
                                                             WAVELENGTHS[c], SHAPE, padding)

            # OpticalElement::lens
            # phase term for fixed thin lens right behind ŜLM
            phi_lens = -np.pi / (wvl * tf.convert_to_tensor(FOCAL_LENGTH, tf.float32)) * squared_sum1
            field_o = tf_exp_complex(phi_lens, 'exp_phase_lens_c%s' % c) * field_o

            # effect of aperture
            pupil_function = tf.multiply(tf.complex(aperture, 0.0, name='complex_aperture_c%s' % c), field_o,
                                         name='mult_pupilFunction_c%s' % c)

            # variable grid sizes
            uout = ang_fresnel_propagation_schmidt_padded(pupil_function, Z_DISTANCE, delta1, delta2,
                                                          squared_sum1,
                                                          WAVELENGTHS[c], SHAPE, padding)
            # remove padding
            uout = uout[padding:-padding, padding:-padding]

            # psf is squared magnitude of amplitude spread function, which is the field
            # of the point source on sensor
            uout = tf.abs(uout, name='magnitude_h_c%s' % c)
            psf = tf.square(uout, name='hSquared_c%s' % c)

            if ZEROTH_ORDER > 0:
                field_unmodulated = ang_fresnel_propagation_schmidt_padded(field_unmodulated, Z2_DISTANCE, delta,
                                                                           delta1, squared_sum, WAVELENGTHS[c],
                                                                           SHAPE, padding)
                field_unmodulated = tf_exp_complex(phi_lens, 'exp_phase_lens_c%s' % c) * field_unmodulated
                field_unmodulated = tf.multiply(tf.complex(aperture, 0.0, name='complex_aperture_c%s' % c),
                                                field_unmodulated, name='mult_pupilFunction_c%s' % c)
                uout_unmodulated = ang_fresnel_propagation_schmidt_padded(field_unmodulated, Z_DISTANCE, delta1,
                                                                          delta2, squared_sum1, WAVELENGTHS[c],
                                                                          SHAPE, padding)
                uout_unmodulated = uout_unmodulated[padding:-padding, padding:-padding]
                uout_unmodulated = tf.abs(uout_unmodulated, name='magnitude_h_c%s' % c)
                psf_unmodulated = tf.square(uout_unmodulated, name='hSquared_c%s' % c)
                psf = ZEROTH_ORDER * psf_unmodulated + (1. - ZEROTH_ORDER) * psf

            # convolution - image formation model as convolution with shift invariant psf: I_enc = I_gt x psf
            # Todo: account for wavelengths in phi_slm
            # prepare psf as kernel - reshape
            # DEBUG Output
            # psf = tf.Print(psf, [tf.reduce_max(psf)])
            # psf = tf.Print(psf, [tf.reduce_min(psf)])
            # normalize psf for convolution
            psf = tf.divide(psf, tf.reduce_sum(psf), name='norm_psf_c%s' % c)
            psfs.append(psf)

            # convolve
            # using custom fft implementation
            # crop central part of psf for convolution to be bounded by max 4K)
            crop = SHAPE // 2 - 450
            psf = psf[crop:-crop, crop:-crop]

            # cout = conv_fft2(incoming[:, :, :, c], psf, SHAPE, SHAPE - 2 * crop)
            # cout = tf.expand_dims(cout, -1, name='expand_fftOut_c%s' % c)
            cout = incoming[:, :, :, c]
            cout = tf.expand_dims(cout, -1, name='expand_fftOut_c%s' % c)
            coded_image.append(cout)

    # SKIP CONNECTIONS I
    # concat all tensors for output
    psf_out = tf.stack(psfs, 2, name='stack_psfs')
    psf_out = tf.expand_dims(psf_out, 0, name='expand_outPsf')

    print psf_out.shape
    psf_out = tf.tile(psf_out, [tf.shape(incoming)[0], 1, 1, 1], name='tile_psf')
    activations = tf.concat(coded_image, 3, name='concat_codedImages')

    # add gaussian noise to output (to simulate sensor noise), according to paper sec. 3.1.3, eq. 6
    gnoise = tf.random_normal(shape=tf.shape(activations), mean=0.0, stddev=tf.random_uniform([1], 0.0, 0.015,
                                                                                              dtype=tf.float32),
                              dtype=tf.float32, name='gaussian_noise')
    activations = activations  # + gnoise
    # downres test
    activations = ConcatWith('concat_decOutput', activations, [psf_out], 3)
    if SHAPE > 1024:
        # scale activations down for training - for high resolutions
        activations = area_downsample(activations, 5, 'downsampling_activations')
        # renormalize
        # activations = tf.concat([activations[:, :, :, 0:CHANNELS],
        #                         activations[:, :, :, CHANNELS:] / tf.reduce_sum(
        #                         activations[:, :, :, CHANNELS])], axis=3)

    return activations, slm_viz


def _decoder(enc_activations):
    """Decode into final image
        Decode with wide convolutional network, architecture based on u-net
    """
    # with TowerContext('tower0', is_training=True):
    # deactivated
    # adding one iteration of wiener deconvolution to the input od neural network, learning noise parameter
    # gamma = tf.get_variable('gamma_decoder', [1, 1], initializer=tf.zeros_initializer)
    # dir_deconv = wiener_deconvolution_tf(enc_activations[:, :, :, 0:CHANNELS],
    #                                      enc_activations[0, :, :, CHANNELS], gamma)
    # enc_activations = ConcatWith('concat_dirDeconv', enc_activations, [dir_deconv], 3)

    # shift range
    enc_activations = tf.identity(enc_activations, name='dec_input')
    # enc_activations = enc_activations * 2.0 - 1.0
    with tf.name_scope('decoder'):
        dec_out = reconstruction.unet4(enc_activations[:, :, :, 0:CHANNELS], CHANNELS, FEATURE_FACTOR)
        return dec_out


def inference(input, model=LOGDIR, d=100.0):
    # reset Graph, just in case
    tf.reset_default_graph()

    with tf.Graph().as_default():
        with TowerContext('tower0', is_training=False):
            # input placeholder with arbitrary batch size, will be one in single image mode
            x = tf.placeholder('float32', [None, INSHAPE, INSHAPE, CHANNELS])
            # x = tf.tile(x, [4, 1, 1, 1], name='tile_input')

            # build graph
            activations, slm = _encoder(x, d)
            prediction = _decoder(activations)
            # prediction = tf.Print(prediction, [prediction])

            prediction_scaled = tf_linear2srgb(prediction)
            x_scaled = tf_linear2srgb(x)
            psnr_pred = psnr(prediction_scaled, x_scaled, maxp=255)
            ssim_pred = tf.reduce_mean(tf.image.ssim(prediction_scaled, x_scaled, max_val=255))
            # enc_sharpness = tf_fm(activations)

            # set up model saver/loader
            saver = tf.train.Saver()

            # create session and launch model
            with tf.Session() as sess:
                # Restore variables
                sess.run(tf.global_variables_initializer())
                chkpt = tf.train.latest_checkpoint(model)
                saver.restore(sess, chkpt)
                print 'Model restored.'

                slm_p = slm.eval()
                # var = [v for v in tf.global_variables() if v.name == "conv_output_conv2D/b:0"][0]
                # print var.eval()

                # run session
                output, enc_out, psnrm, ssim = sess.run([prediction, activations, psnr_pred, ssim_pred],
                                                                   feed_dict={x: input})
                # lucy richardson
                # output = lucyrichardson_deconvolution.lr_deconv(enc_out[0, :, :, 0], enc_out[0, :, :, 1])

                return output[0], enc_out[0, :, :, CHANNELS:], slm_p, psnrm, ssim


# main
args = get_args()
# open image
try:
    f = open(args.input)
except IOError as e:
    print e.errno
    print args.input
else:
    im_data = np.asarray(bytearray(f.read()), dtype=np.uint8)
    # reading image as greyscale
    image = cv2.imdecode(np.asarray(bytearray(im_data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # image = center_crop_image(image, SLM_RES)
    print 'press key to begin processing'
    # cv2.imshow('imageInput', image)
    # cv2.waitKey(0)
    image = srgb2linear(image)
    # image = image.astype(np.float32)
    # image /= 255.0
    # image = image ** 0.5
    print image.shape
    y_pad = np.abs(image.shape[0] - image.shape[1]) // 2
    # image = center_crop_image(image, np.min(image.shape))
    image = np.pad(image, ((y_pad, y_pad), (0, 0)), 'constant', constant_values=(0, 0))
    image = cv2.resize(image, (INSHAPE, INSHAPE), interpolation=cv2.INTER_AREA)
    # reshape to 4D array
    image = np.reshape(image, [1, image.shape[0], image.shape[1], CHANNELS])
    print 'input_shape:'
    print image.shape
    # inference
    decoded, psf, slm_pattern, psnrm, ssim = inference(image, args.model, float(args.d))
    # print decoded.shape
    fmm = fm(decoded)
    decoded = linear2srgb(decoded)
    decoded = np.clip(decoded.astype(np.uint8), 0, 255)
    psf = psf / np.max(psf)
    print 'psnr: '
    print psnrm
    print 'ssim: '
    print ssim
    print 'fm (sharpness): '
    print fmm
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    cv2.imwrite(args.output, decoded)
    psf_output = os.path.splitext(args.output)[0]
    cv2.imwrite(psf_output + '_psf.png', psf * 255)
    # cv2.imshow('Decoded_Image', decoded)
   #  cv2.imshow('SLM_Image', slm_pattern)
    # cv2.imshow('PSF (normalized)', psf)
    # cv2.waitKey(0)






