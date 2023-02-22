#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Andreas Engelhardt
# EDOF example application of framework for Simulation of SLM driven computational camera applications
# and reconstruction.

"""
To start training fill out the global settings below, create a dataset lmdb using create_data_highres.py if
not already available. Then execute this script with the following commands:

train from scratch:
    python psfDesign_train.py --gpu 0 (,1,2...)
re-start previous checkpoint:
    python psfDesign_train.py --gpu 0 --load ../train_logs/psfDesign.../checkpoint

Requires: tensorflow 1.8 (gpu), tensorpack 0.86, numpy 1.1, lmdb 0.9, scipy 1.1
"""

import matplotlib

matplotlib.use('Agg')

import argparse
import cv2
from fft_tf import tf_fftshift2d, tf_ifftshift2d, tf_conv2d
import load_data
import numpy as np
import os
import tensorflow as tf
import reconstruction
from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.globvars import GlobalNS as use_global_argument
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_nr_gpu
import matplotlib.pyplot as plt

# ------------------------------------
# Global Settings
# ------------------------------------

# Number of sequences per update.
BATCH = 1
# Resolution of simulation grid.
SHAPE = 4600
# Dimension of input shape. This is the shape that the ground truth data is in.
INSHAPE = SHAPE // 5
# Scaling factor for feature number in reconstruction CNN (a factor of 2 cuts filters in half)
FEATURE_FACTOR = 1

# Number of band when using Zernike polynomials.
# ZERNIKE_MODES = 2

# Resolution of sensor (one dimension) assuming a square sensor.
SENSOR_SHAPE = SHAPE
# Resolution of phase variable, gets padded to a square of the first dimension-
SLM_SHAPE = [2300, 1294]
# Öength of longer size of SLM in [m], assuming 16:9 aspect for SLM here.
L_SLM = 15.36e-3
# Factor to represent aspect ratio of the phase mask.
ASPECT_FACTOR = 0.5625
# Padding for propagations, experimentally evaluated for depth range and grid spacings.
PAD = 200
# Aperture opening in [1/stops].
F_NUMBER = 5.6
# Length of the longer size of the camera sensor in [m].
L_SENSOR = 3.45e-6 * 4096  # set to the sensor size of a FLIR Oryx 4K camera.
# Focal length in [m] of the fixed lens element.
FOCAL_LENGTH = .1
# Distance of focus plane from pupil of optical element in m, -1 for infinity.
FOCUS_DISTANCE = 2.0

# Determine distance between optical elements and image plane (z)
if FOCUS_DISTANCE == -1:
    Z2_DISTANCE = FOCAL_LENGTH
else:
    Z2_DISTANCE = np.abs((FOCUS_DISTANCE * FOCAL_LENGTH) / (FOCUS_DISTANCE - FOCAL_LENGTH))

Z1_DISTANCE = 0.0835  # distance between SLM and lens in m - this is an approximated, measured value

# effective aperture of the lens (diameter of entrance pupil)
APERTURE = FOCAL_LENGTH / F_NUMBER
# Number of convolution layers in decoder network, must be multple of 2, one
# additional layer for output so it's N +1 conv layers in total when using the plainCNN. When using a different
# reconstruction method this parameter has no effect
N_LAYERS = 8
# Image channels used, e.g. 1 for greyscale, 3 for rgb
CHANNELS = 1
# Wavelengths corresponding to channels in [m], RGB data is read in the correct order r-g-b.
WAVELENGTHS = [5.2e-7]
# Whether to use fresnel approximation of incident spherical wavefront, default is off
CURVATURE_APPROX = False
# Determines the shape of the aperture in the fixed lens
RECT_APERTURE = False
# Whether we want to crop the incident field to the SLM aspect ratio. This should be True.
SLM_CROP = True
# A wavelength is chosen from the list randomly to enable multi spectral training at no extra memory cost.
RND_WVL = False
# Strength of random perspective transform applied to PSF, off if 0, adequate values are between 0.005 and 0.05
RND_TRANSFORM = 0
# Factor by which an unmodulated version of the psf is blended into the final psf to simulate zeroth order diffraction
ZEROTH_ORDER = 0.00
# Add blur to the generated PSF to model aberrations and limited resolution of camera system (SLM).
BLUR = True
# Factor to enforce smoothness of the phase pattern. Values between 0.01 and 0.5 seem appropriate.
REGULARIZATION_FACTOR = 0.00
# List of distances to be simulated during training.
# DISTANCES = [50., 6.0, 3.0, 2.0, 1.5, 1.2, 1., 0.85, 0.75, 0.66, 0.6, 0.55, 0.5]
# DISTANCES = [50, 5, 2, 1, 0.67, 0.5]
DISTANCES = [50., 4.21, 2.0, 1.36, 0.88, 0.6, 0.5]

# Set a fixed log dir outside of repository.
LOGDIR = '/graphics/scratch/engelhar/train_logs/model_slm45k_2prop_unet4_oryx_d2_rot'

# Path to training data lmdb.
TRAIN_PATH = '/graphics/scratch/engelhar/datasets/google-highres_train_80k.lmdb'

# Path to testing data lmdb.
VAL_PATH = '/graphics/scratch/engelhar/datasets/google-highres_val_800.lmdb'

# Path to data for export as lmdb.
TEST_PATH = '/graphics/scratch/engelhar/datasets/google-highres_val_800.lmdb'

# Path to write test data during or after training to.
EXPORT_PATH = '/graphics/scratch/engelhar/test_data/unet4_slm45k_2prop_oryx_d2_rot'
# maximum number of outputted during test callback, set to -1 if all in dataflow are to be outputted
MAX_TEST = 16

# Security measures.
if len(WAVELENGTHS) != CHANNELS:
    if RND_WVL:
        print 'Elements in list of wavelengths does not match the number of channels! On Purpose?...'
    else:
        print 'Elements in list of wavelengths has to match the number of channels! Exiting...'
        raise SystemExit(0)


# ------------------------------------
# Utilities
# ------------------------------------

# PSNR, taken from https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/tfutils/symbolic_functions.py
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

        # Set up first point set - mutated four corners.
        src_pts = np.zeros((4, 2), dtype=np.float32)

        src_pts[0] = [0 + rnd_pert[0], 0 + rnd_pert[1]]
        src_pts[1] = [shape[0] + rnd_pert[2], 0 + rnd_pert[3]]
        src_pts[2] = [shape[0] + rnd_pert[4], shape[1] + rnd_pert[5]]
        src_pts[3] = [0 + rnd_pert[6], shape[1] + rnd_pert[7]]

        # Determine dimensions of transformed image.
        nwidth = np.maximum(np.abs(src_pts[0][0] - src_pts[1][0]), np.abs(src_pts[2][0] - src_pts[3][0]))
        nheight = np.maximum(np.abs(src_pts[0][1] - src_pts[3][1]), np.abs(src_pts[1][1] - src_pts[2][1]))

        # Second point set, transformed image corners.
        dst = np.array([[0, 0],
                        [nwidth - 1, 0],
                        [nwidth - 1, nheight - 1],
                        [0, nheight - 1]], dtype=np.float32)

        # Use opencv to calculate perspective transform (closed form solution).
        # We map from destination to source points.
        tm = cv2.getPerspectiveTransform(dst, src_pts)

        # Compensate for shift of center point.
        # Center point is in destination coordinates.
        center_point = np.array([[[shape[0] / 2, shape[1] / 2]]], dtype=np.float32)

        # Get ransformed center point.
        ncenter = cv2.perspectiveTransform(center_point, tm)

        # Create translation matrix.
        translation = np.float32([(ncenter[0, 0, 0] - center_point[0, 0, 0]), (ncenter[0, 0, 1] - center_point[0, 0, 1])])

        translations.append(translation)

        tm = tm.flatten()

        # Store parameters in format suitable for tensorflow's image transform (8 parameters as flat vector).
        transforms.append(tm[0:8])

    return transforms, translations


# Based on a set of precomputed perspective transforms, perturbate an image.
def random_perturbation(image_tensor, strength, name, number=32):
    image_shape = image_tensor.shape.as_list()
    # Make sure tensor has 4 dims.
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
    # Create transforms and save in variable.
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
    # Only perturbate half of the time.
    output = tf.where(t_idx[1] > 0, image_transformed, image_tensor)

    # Convert back to input format.
    if len(image_shape) < 4:
        if len(image_shape) > 2:
            output = output[0, :, :, :]
        else:
            output = output[0, :, :, 0]
    return output


# Encode image data assumed to be linear with SRGB 'gamma' transfer function, color primaries remain untouched.
# Output is in range 0 to 255 float. Based on
# https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_(%22gamma%22)
def linear2srgb(im):
    im_srgb = im
    alpha = 0.055
    im_srgb[im <= 0.0031308] *= 12.92
    bool_mask = im > 0.0031308
    im_srgb[bool_mask] = im_srgb[bool_mask] ** (1.0 / 2.4) * (1.0 + alpha) - alpha

    return im_srgb * 255.0


# Linear 2 SRGB conversion using tensorflow ops.
def tf_linear2srgb(im):
    # im_srgb = im
    alpha = 0.055
    mask = im <= 0.0031308
    im_tmp = tf.where(mask, im * 12.92, im, name='cond1_linear2srgb')
    im_tmp2 = tf.pow(im, 1.0 / 2.4) * (1.0 + alpha) - alpha
    im_srgb = tf.where(tf.logical_not(mask), im_tmp2, im_tmp, name='cond2_linear2srgb')

    return im_srgb * 255.0


# Creates a circle like circ() above but with tensorflow ops, no anti-aliasing.
def tf_circ(x, y, d):
    r = tf.sqrt(x * x + y * y)
    circle = tf.cast(r <= d / 2.0, dtype=tf.float32)
    # Disabled for speed up.
    # mask = r == d/2
    # circle = tf.where(mask, tf.constant(0.5, shape=[SLM_SHAPE, SLM_SHAPE], dtype=tf.float32), circle, name='cond2_circle')
    return circle


# Creates a rectangular coordinate grid from vector x and side length d.
def tf_rect(x, y, d1, d2):
    x = tf.abs(x)
    y = tf.abs(y)
    y1 = tf.cast(x <= d1 / 2.0, dtype=tf.float32)
    y2 = tf.cast(y <= d2 / 2.0, dtype=tf.float32)
    return tf.multiply(y1, y2)


# Convert cartesian to polar coordinates (numpy arrays only).
def cart2pol(x, y):
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return theta, rho


# Creates Gaussian blur kernel, parameters: length (size) and sigma
def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel1 = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel1 / np.sum(kernel1)


# Computes the complex exponential with the help of Euler's formula since there is no Cuda kernel for it in (older) tf.
def tf_exp_complex(phase, name):
    return tf.complex(tf.cos(phase), tf.sin(phase), name=name)


# Method to downsample a 2D tensor by a factor using area interpolation (averaging).
def area_downsample(image, factor, name):
    factor = int(factor)
    output = tf.nn.avg_pool(image, [1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)
    return output


# Upsample a tensor by a factor of two - the reverse of area downsample.
def upsample_neareast(image, factor, name):
    # Extend to four dimensional tensor.
    img_shape = image.shape.as_list()
    if len(img_shape) < 4:
        image = tf.reshape(image, [1, img_shape[0], img_shape[1], 1])
        target_res_x = factor * img_shape[0]
        target_res_y = factor * img_shape[1]
    else:
        target_res_x = factor * img_shape[1]
        target_res_y = factor * img_shape[2]
    # Here we use the tensorflow high level operation. It is equal to a transposed Conv2d operation with stride 1/2.
    image_big = tf.image.resize_nearest_neighbor(image, [target_res_x, target_res_y], align_corners=True, name=name)
    if len(img_shape) < 4:
        image_big = image_big[0, :, :, 0]
    return image_big


# Compute modulo on GPU.
def modulo_tf(a, b):
    return a - tf.floor_div(a, b) * b


# Padded version of ang_spec_prop assuming u_in has a padding of "padding".
def ang_fresnel_prop_variable_padded(u_in, z, ds, do, r1sq, wavelength, res, padding):
    c = WAVELENGTHS.index(wavelength)
    wvl = tf.convert_to_tensor(wavelength, dtype=tf.float32)
    # Padding to reduce artifacts, we match the padding of the input.
    delta1 = ds
    delta2 = do
    n = res
    n_pad = padding
    n_padded = n + 2 * n_pad

    # wavenumber k
    k = 2 * np.pi / wvl

    # Source coordinates are represented by r1sq.

    # Spatial frequencies of source plane.
    delta_freq = 1 / (delta1 * n_padded)
    coordsf = tf.linspace(-n_padded // 2 * delta_freq, (n_padded // 2 - 1) * delta_freq, n_padded,
                          name='linspace_freqCoords_c%s' % c)
    fx1, fy1 = tf.meshgrid(coordsf, coordsf, name='meshgrid_freqCoordinates_c%s' % c)
    fsq = fx1 * fx1 + fy1 * fy1

    # Scaling parameter.
    m = delta2 / delta1

    # Observation plane coordinates.
    coords2 = tf.linspace(-n_padded // 2 * delta2, (n_padded // 2 - 1) * delta2, n_padded,
                          name='linspace_r2Coords_c%s' % c)
    x2, y2 = tf.meshgrid(coords2, coords2, name='meshgrid_r2Coordinates_c%s' % c)
    r2sq = x2 * x2 + y2 * y2

    # Quadratic phase factors:
    q1 = tf_exp_complex(k / 2. * (1. - m) / z * r1sq, name='phase_q1_c%s' % c)
    q2 = tf_exp_complex(np.pi * wavelength * -1.0 * z / m * fsq, name='phase_q2_c%s' % c)
    q3 = tf_exp_complex(k / 2. * (m - 1.) / (m * z) * r2sq, name='phase_q3_c%s' % c)

    # Compute output field at observation plane: Scaling factor and first phase factor.
    u = tf.multiply(u_in / m, q1, name='mult_q1_c%s' % c)
    # Perform fft.
    u_fft = tf_fftshift2d(tf.fft2d(tf_fftshift2d(u), name='fft_as_c%s' % c)) * (delta1 * delta1)
    # Apply second phase factor.
    u = u_fft * q2
    # Transform back
    factor = n_padded * delta_freq
    u = tf_ifftshift2d(tf.ifft2d(tf_ifftshift2d(u), name='ifft_as_c%s' % c)) * (factor * factor)

    # Remove padding and multiply third phase factor
    u_out = u * q3

    return u_out


# ------------------------------------
# Model
# ------------------------------------

class Model(ModelDesc):

    # Define inputs, incoming and expected need to be the same shape.
    # Additional a scalar to specify a distance is given.
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, INSHAPE, INSHAPE, CHANNELS), 'incoming'),
                InputDesc(tf.float32, (None, INSHAPE, INSHAPE, CHANNELS), 'expected'),
                InputDesc(tf.float32, (None,), 'distance')]

    def _encoder(self, incoming, depth):
        """Each incoming image is encoded into an image on the sensor simulating the image formation process in the
            camera with phase modulation applied on the way.
        """

        with tf.name_scope('optics_sim'):
            # Aperture coordinates, center point is slightly shifted since dimensions are multiple of 2
            # Adding padding around the real size of SLM
            padding = PAD
            # Sample rate.
            delta = L_SLM / float(SHAPE)
            # Extending the dimensions as padding.
            n = SHAPE + 2 * padding
            ext_length = L_SLM + delta * (padding * 2)
            delta = ext_length / float(n)
            coords = tf.linspace(float(- n / 2), float(n / 2 - 1), n)
            coords = tf.multiply(coords, delta, name='mult_coords1_delta')
            x, y = tf.meshgrid(coords, coords, name='meshgrid_apertureCoordinates')

            # Squared sum of coordinates SLM plane.
            squared_sum = x * x + y * y

            # Set up aperture coordinates.
            delta1 = APERTURE / float(SHAPE)
            ext_length2 = APERTURE + delta1 * (padding * 2)
            delta1 = ext_length2 / float(n)
            delta_inv = 1 / delta
            x_prime = x * delta_inv * delta1
            y_prime = y * delta_inv * delta1

            # Squared sum on aperture plane.
            squared_sum1 = x_prime * x_prime + y_prime * y_prime

            # Grid spacing on sensor plane.
            delta2 = L_SENSOR / SHAPE
            ext_length3 = L_SENSOR + delta2 * (2 * padding)
            delta2 = ext_length3 / float(n)

            # Create aperture mask.
            if RECT_APERTURE:
                aperture = tf_rect(x_prime, y_prime, APERTURE, APERTURE)
            else:
                aperture = tf_circ(x_prime, y_prime, APERTURE)

            # Optical Element::SLM
            with tf.name_scope('OpticalElement_SLM'):
                # Phase modulation in range [0, 2pi] mapped to a 0 to 1 range of the variable.
                quadrant1 = tf.get_variable('a_slm', [SLM_SHAPE[1] // 2, SLM_SHAPE[0] // 2],
                                            initializer=tf.zeros_initializer)
                tf.add_to_collection("SLM_coefficients", quadrant1)
                phi_slm = tf.concat([quadrant1, tf.reverse(quadrant1, [1])], axis=1)
                phi_slm = tf.concat([phi_slm, tf.reverse(phi_slm, [0])], axis=0)

                # Output slm pattern to tensorboard.
                viz_phi = tf.expand_dims(phi_slm, -1)
                viz_phi = tf.expand_dims(viz_phi, 0)
                viz_phi *= 255.0
                viz_slm = tf.cast(tf.clip_by_value(viz_phi, 0, 255), tf.uint8, name='slmviz')
                tf.summary.image('SLM_pattern', viz_slm, max_outputs=1)

                # Add uniform noise to SLM mask (to simulate phase flickering). Uniform noise is used in this case.
                unoise = tf.random_uniform(shape=tf.shape(phi_slm), minval=-.018, maxval=0.018, dtype=tf.float32,
                                           name='uniformnoise_SLM')
                phi_slm = phi_slm + unoise

                # Define available range as 0 to 1 which corresponds to a range of 2pi, centered aroung 0.
                phi_slm = tf.multiply(phi_slm, 2 * np.pi, name='shift_range') - np.pi

                # Pad and scale field.
                pad = int((SLM_SHAPE[0] - SLM_SHAPE[1]) * 0.5) + padding // 2
                phi_slm = tf.pad(phi_slm, tf.constant([[pad, pad], [padding // 2, padding // 2]]), "CONSTANT")
                phi_slm = upsample_neareast(phi_slm, 2, 'upsample_slm')

            # Crop the variable grid to the aspect ratio of the SLM and apply aperture mask.
            if SLM_CROP:
                x_length = APERTURE
                y_length = APERTURE * ASPECT_FACTOR
                crop_aspect = tf_rect(x_prime, y_prime, x_length, y_length)
                crop = crop_aspect

            # Distance d is randomly drawn from the predefined list of distances, one for each batch.
            # At inference time we will check if there is a distance given.
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

            # Tell tf the shape of the distance in case it got lost during the conditional.
            distance.set_shape(1)
            # Tensorboard output.
            add_moving_summary(distance[0])

            # Upsample incoming image to SHAPE x SHAPE.
            if INSHAPE < SHAPE:
                incoming = upsample_neareast(incoming, 5, 'upsample_incoming')

            # ::Wavelength dependent operations are set up in a loop over the channels::
            # Set up data structures to hold the simulation results.
            coded_image = []
            psfs = []
            # Loop over all wavelengths - if more than three do a stochastic version instead!
            for c in range(0, CHANNELS):

                # Choose a wavelength randomly.
                if RND_WVL:
                    wvl_idx = tf.random_uniform([1], 0.0, 1.0, dtype=tf.float32)
                    wvl_idx = tf.cast(wvl_idx * (len(WAVELENGTHS) - 1), tf.int32, name='cast_wvlIdx')
                    wavelengths_tensor = tf.convert_to_tensor(WAVELENGTHS)
                    wvl = tf.gather(wavelengths_tensor, wvl_idx, name='gather_wvl')
                else:
                    wvl = tf.convert_to_tensor(WAVELENGTHS[c], dtype=tf.float32, name='wavelength_c%s' % c)

                # Phase term for propagation from (virtual) object point source(s) to optical system of distance d.
                # Paraxial approximation of spherical wave emitted by a point source at distance d from optical element.
                if CURVATURE_APPROX:
                    phi_d = np.pi / (wvl * distance) * squared_sum
                    field_o = tf_exp_complex(phi_d, 'exp_phase_o_c%s' % c)
                else:
                    # To prevent overflow / loss of precision for long distances and short wavelength
                    # process in float64 and work with modulo 2pi with lower precision afterwards.
                    squared_sum2 = tf.cast(squared_sum, dtype=tf.float64)
                    distance = tf.cast(distance, dtype=tf.float64)
                    r = tf.sqrt(squared_sum2 + distance * distance, name='sqrt_r_c%s' % c)
                    k = tf.cast(2 * np.pi / wvl, dtype=tf.float64)
                    phi_d = modulo_tf(k * r, 2. * np.pi)
                    phi_d = tf.cast(phi_d, dtype=tf.float32)
                    field_o = tf_exp_complex(phi_d, 'exp_phase_o_c%s' % c)

                # Crop the variable grid to the aspect ratio of the SLM and apply aperture mask.
                if SLM_CROP:
                    crop = tf.complex(crop, crop)
                    field_o = tf.multiply(field_o, crop, name='mult_field_0_crop_c%s' % c)

                # If zeroth order diffraction peak is to be simulated, create an unmodulated field as well.
                if ZEROTH_ORDER > 0:
                    field_unmodulated = field_o
                    field_o = tf_exp_complex(phi_slm, 'exp_phase_o_c%s' % c) * field_o
                else:
                    field_o = tf_exp_complex(phi_slm, 'exp_phase_o_c%s' % c) * field_o

                # ::Propagation between SLM and lens::
                with tf.name_scope('Fresnel_Prop1'):
                    field_o = ang_fresnel_prop_variable_padded(field_o, Z1_DISTANCE, delta, delta1, squared_sum,
                                                               WAVELENGTHS[c], SHAPE, padding)

                # OpticalElement::lens
                with tf.name_scope('OpticalElement_Lens'):
                    # Phase term for fixed thin lens right behind ŜLM.
                    phi_lens = -np.pi / (wvl * tf.convert_to_tensor(FOCAL_LENGTH, tf.float32)) * squared_sum1
                    field_o = tf_exp_complex(phi_lens, 'exp_phase_lens_c%s' % c) * field_o

                    # Effect of aperture.
                    pupil_function = tf.multiply(tf.complex(aperture, 0.0, name='complex_aperture_c%s' % c), field_o,
                                                 name='mult_pupilFunction_c%s' % c)

                # ::Propagation between optics and sensor::
                with tf.name_scope('Fresnel_Prop2'):
                    uout = ang_fresnel_prop_variable_padded(pupil_function, Z2_DISTANCE, delta1, delta2,
                                                            squared_sum1,
                                                            WAVELENGTHS[c], SHAPE, padding)
                # Remove padding.
                uout = uout[padding:-padding, padding:-padding]

                # Psf is squared magnitude of amplitude spread function, which is the field
                # of the point source on sensor.
                uout = tf.abs(uout, name='abs_hMagnitude_c%s' % c)
                psf = tf.square(uout, name='square_hSquared_c%s' % c)

                # If we are simulating the zeroth order diffraction effect, perform two sets of propagations and compute
                # the weighted average of the two psfs.
                if ZEROTH_ORDER > 0:
                    field_unmodulated = ang_fresnel_prop_variable_padded(field_unmodulated, Z1_DISTANCE, delta,
                                                                         delta1, squared_sum, WAVELENGTHS[c],
                                                                         SHAPE, padding)
                    field_unmodulated = tf_exp_complex(phi_lens, 'exp_phase_lens_c%s' % c) * field_unmodulated
                    field_unmodulated = tf.multiply(tf.complex(aperture, 0.0, name='complex_aperture_c%s' % c),
                                                    field_unmodulated, name='mult_pupilFunction_c%s' % c)
                    uout_unmodulated = ang_fresnel_prop_variable_padded(field_unmodulated, Z2_DISTANCE, delta1,
                                                                        delta2, squared_sum1, WAVELENGTHS[c],
                                                                        SHAPE, padding)
                    uout_unmodulated = uout_unmodulated[padding:-padding, padding:-padding]
                    uout_unmodulated = tf.abs(uout_unmodulated, name='magnitude_h_c%s' % c)
                    psf_unmodulated = tf.square(uout_unmodulated, name='hSquared_c%s' % c)
                    psf = ZEROTH_ORDER * psf_unmodulated + (1. - ZEROTH_ORDER) * psf

                # ::Convolution::
                # Using image formation model as convolution with shift invariant psf: I_enc = I_gt x psf.
                if BLUR:
                    # Blur psf slightly to model aberrations.
                    kernel = gkern(5, 1.5)
                    # print kernel
                    psf_d = tf.expand_dims(psf, 0)
                    psf_d = tf.expand_dims(psf_d, -1)
                    kernel = np.reshape(kernel, (kernel.shape[1], kernel.shape[0], 1, 1))
                    psf_d = tf.nn.conv2d(psf_d, kernel, strides=[1, 1, 1, 1], padding='SAME')
                    psf = psf_d[0, :, :, 0]

                # Normalize psf for convolution.
                psf = tf.divide(psf, tf.reduce_sum(psf), name='norm_psf_c%s' % c)
                psfs.append(psf)

                # Crop central part of psf for convolution for EDOF. The shape of the cropped patch has been found
                # experimentally.
                crop = SHAPE // 2 - 450
                psf = psf[crop:-crop, crop:-crop]

                # ::Random perspective perturbation, if wanted::
                if RND_TRANSFORM > 0:
                    psf = random_perturbation(psf, RND_TRANSFORM, name='random_transform_c%s' % c)

                # Convolve using custom fft implementation
                cout = tf_conv2d(incoming[:, :, :, c], psf, SHAPE, SHAPE - 2 * crop)
                cout = tf.expand_dims(cout, -1, name='expand_fftOut_c%s' % c)
                coded_image.append(cout)

        # ::Prepare encoder output::
        # Concat all tensors for output. This basically simulates the sensor.
        with tf.name_scope('Sensor'):
            psf_out = tf.stack(psfs, 2, name='stack_psfs')
            psf_out = tf.expand_dims(psf_out, 0, name='expand_outPsf')

            print psf_out.shape
            psf_out = tf.tile(psf_out, [tf.shape(incoming)[0], 1, 1, 1], name='tile_psf')
            activations = tf.concat(coded_image, 3, name='concat_codedImages')

            # Scale activations down for reconstruction - for high resolutions.
            if SHAPE > INSHAPE:
                activations = area_downsample(activations, SHAPE//INSHAPE, 'downsampling_activations')

            # The psf is renormalized.
            if SHAPE > INSHAPE:
                psf_out = area_downsample(psf_out, SHAPE//INSHAPE, 'downsampling_psfs')
                psf_out = tf.divide(psf_out, tf.reduce_sum(psf_out), name='renorm_psf')

            # Add gaussian noise to output (to simulate sensor noise).
            sigma = tf.random_uniform([1], 0.005, 0.016, dtype=tf.float32)
            # sigma = tf.Print(sigma, [sigma], message='Stdev_Gaussian', first_n=50, name='print_stdev')
            gnoise = tf.random_normal(shape=tf.shape(activations), mean=0.0, stddev=sigma,
                                      dtype=tf.float32, name='gaussian_noise')
            activations = tf.clip_by_value(activations + gnoise, 0.0, 1.0, name='clip_sensor_output')
            # Concat output
            activations = ConcatWith('concat_decOutput', activations, [psf_out], 3)

        return activations

    def _decoder(self, enc_activations):
        """Decode into final image
            The decoder is a convolutional neural network or any other method chosen from reconstruction.py
        """
        with tf.name_scope('decoder'):
            dec_out = reconstruction.unet4(enc_activations, CHANNELS, FEATURE_FACTOR)
            return dec_out

    # Set up the tensorflow graph.
    def _build_graph(self, inputs):
        incoming, expected, depth = inputs
        # Connect encoder and decoder
        activations = self._encoder(incoming, depth)
        prediction = self._decoder(activations)

        # Total variation regularization.
        slm_var = tf.get_collection('SLM_coefficients')[0]
        slm_var = tf.expand_dims(slm_var, -1)
        variation = tf.image.total_variation(slm_var, name='total_variation_regularizer')
        reg = tf.get_variable('regularization_factor', initializer=float(REGULARIZATION_FACTOR), trainable=False)
        variation = tf.reduce_sum(variation) * reg
        # Compute loss: mean-squared-error as reconstruction error.
        self.cost = tf.add(tf.reduce_mean(tf.squared_difference(prediction, expected), name="mse"), 1e-8 * variation,
                           name='reg_mse')
        # Add to tensorboard output.
        add_moving_summary(self.cost)

        # Just for TensorBoard track peak-signal-to-noise ratio and structural similarity index metric.
        with tf.name_scope('psnr'):
            prediction_scaled = tf_linear2srgb(prediction)
            expected_scaled = tf_linear2srgb(expected)

            prediction_psnr = psnr(prediction_scaled, expected_scaled, maxp=255., name="prediction")
            prediction_ssim = tf.reduce_mean(tf.image.ssim(prediction_scaled, expected_scaled, max_val=255),
                                             name='ssim')

        add_moving_summary(prediction_psnr, prediction_ssim)

        # Use tensorboard for visualization of image tensors.
        with tf.name_scope("visualization"):
            psf_viz = activations[:, :, :, CHANNELS:]
            psf_viz = psf_viz / tf.reduce_max(psf_viz)
            difference = tf.abs(prediction - expected)
            difference = tf.divide(difference, tf.reduce_max(difference), name='normalize_difference')
            viz = (tf.concat([expected, prediction, difference, activations[:, :, :, 0:CHANNELS], psf_viz], 2))
            viz = tf_linear2srgb(viz)
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            pred_export = tf.clip_by_value(prediction * 255.0, 0, 255)
            pred_export = tf.cast(pred_export, tf.uint8, name='export_viz')
        tf.summary.image('expected, estimate, absDifference, encoderOut, psf', viz,
                         max_outputs=max(5, BATCH))

    # ADAM optimizer with settings according to original paper vy Kingma and Ba.
    def _get_optimizer(self):
        # Starting with a low learning rate.
        lr = tf.get_variable('learning_rate', initializer=float(5e-5), trainable=False)
        tf.summary.scalar('learning_rate' + '-summary', lr)
        return tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)


# ----------------------------------------
# DATA
# ----------------------------------------

def get_data(subset='train'):
    assert subset in ['train', 'test', 'export']
    isTrain = (subset == 'train')

    # Load dataset from lmdb file using load_data.py.
    if isTrain:
        data_path = TRAIN_PATH
    elif subset == 'export':
        data_path = TEST_PATH
    else:
        data_path = VAL_PATH

    ds = load_data.loaddata(data_path)
    if isTrain:
        # For training, we augment the data: Do random crop, rotation and flips.
        augmentors = [
            # imgaug.Brightness(30, clip=False),
            # imgaug.Contrast((0.4, 1.5)),
            imgaug.ResizeShortestEdge(2048, cv2.INTER_CUBIC),
            imgaug.RandomCrop([int(INSHAPE), int(INSHAPE)]),
            imgaug.Flip(horiz=True),
            imgaug.Flip(vert=True),
            imgaug.Rotation(max_deg=180.0, center_range=(0.5, 0.5), step_deg=90.0)
        ]
    else:
        # For testing, we just use the same data again and again.
        augmentors = [
            imgaug.ResizeShortestEdge(2048, cv2.INTER_CUBIC),
            imgaug.CenterCrop([int(INSHAPE), int(INSHAPE)]),
        ]
    # Augment all data.
    ds = AugmentImageComponents(ds, augmentors, index=[0])

    # For EDOF, just copy the data point and add the distance parameter.
    if subset == 'export':
        ds = MapData(ds, lambda dp: [dp[0], dp[0]])
    else:
        ds = MapData(ds, lambda dp: [dp[0], dp[0], 0.])
    # Create batches.
    ds = PrintData(ds)
    ds = BatchData(ds, BATCH)
    # Start multiple processes for data pre-processing parallel to training.
    if subset == 'export':
        return ds
    else:
        ds = PrefetchDataZMQ(ds, 6 if isTrain else 8, hwm=30)

        return ds


# ----------------------------------------
# CONFIG
# ----------------------------------------


# Callback to clip the coefficients to the positive domain.
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


# Callback to shuffle variables around.
class ShuffleVariable(Callback):
    def _setup_graph(self):
        varlist = tf.get_collection("SLM_coefficients")
        ops = []
        for v in varlist:
            n = v.op.name
            logger.info("Shuffle {}".format(n))
            ops.append(tf.assign(v, fftshift(v, SLM_SHAPE)))
        self._op = tf.group(*ops, name='clip')

    def _trigger(self):
        print 'shuffling tensor'
        self._op.run()


# Callback to save final SLM pattern to disk.
class OutputSLM(Callback):
    def _setup_graph(self):
        self.varlist = tf.get_collection("SLM_coefficients")

    def _after_train(self):
        for var in self.varlist:
            pattern = self.trainer.sess.run(var)
            output = "./image_SLM45k_2prop_zo_unet4_oryx_d2_rot.png"
            pattern = np.uint8(np.clip(pattern * 255, 0, 255))
            cv2.imwrite(output, pattern)


# Callback to write out test data, variable trigger.
class TestDataExport(Callback):
    def __init__(self):
        # setup data flow for test
        self.test_ds = get_data('export')
        self.test_ds.reset_state()
        self.distances = [50, 5, 3.8, 2, 1, 0.8, 0.75, 0.5, 0.45]

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(['incoming', 'expected', 'distance'],
                                               ['visualization/viz', 'psnr/prediction'])

    def _trigger(self):
        print 'Generating test data...'
        idx = 0
        psnr_accum = np.zeros(len(self.distances))
        for input, expected in self.test_ds.get_data():
            idx_d = 0
            for d in self.distances:
                outputs = self.pred(input, expected, (float(d),))
                im = outputs[0]
                # Write image to disk.
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
        psnr_accum /= float(MAX_TEST)
        # Create and save psnr plot.
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.xticks(list(range(0, len(self.distances))), self.distances)
        ax.bar(list(range(0, len(self.distances))), psnr_accum, color='c')
        ax.set_xlabel('Distances (depth)')
        ax.set_ylabel('average psnr')
        fig.savefig(EXPORT_PATH + '/psnr_distance_plot.pdf', bbox_inches='tight')
        plt.close(fig)
        # Write out psnr values.
        with open(EXPORT_PATH + '/psnr.txt', 'w') as f:
            f.write(str(psnr_accum)[1:-1])


# Finally, set up the configuration for the trainer.
def get_config():
    # Using a fixed logging dir?
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
            # Perform validation run and display metrics.
            InferenceRunner(dataset_val, [ScalarStats(['mse:0', 'psnr/prediction:0', 'psnr/ssim:0'])]),
            # Using a cyclic learning rate here which is helpful since we don't use batchnorm.
            ScheduledHyperParamSetter('learning_rate', [(8, 2e-4), (40, 1e-4), (95, 1e-4)]),
            # Export test data every 50 epochs.
            PeriodicTrigger(TestDataExport(), every_k_epochs=50)
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(['mse:0', 'psnr/prediction:0']),
            MergeAllSummaries(),
            OutputSLM()
        ],
        model=Model(),
        steps_per_epoch=3200,
        max_epoch=100,
        # If resuming training this needs to be set accordingly.
        starting_epoch=100
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()
    # Use_global_argument(args).
    return args


if __name__ == '__main__':
    args = get_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Get config and set up multi GPU trainer. Using the standard sync'd version here which corresponds to data
    # parallel training
    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    launch_train_with_config(
        config,
        SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
