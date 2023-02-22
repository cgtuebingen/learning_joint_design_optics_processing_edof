import numpy as np
from tensorpack import *
import tensorflow as tf
from fft_tf import conv_fft2, fftshift2d_tf, ifftshift2d_tf, fftshift, ifftshift


# Perform one iteration of deconvolution via Wiener filtering in frequency space, y is a regularizer.
def wiener_deconvolution_tf(img, kern, y, clip_values=True):
    print(img.shape)
    print(kern.shape)
    # Renormalize psf, just in case.
    kern = tf.divide(kern, tf.reduce_sum(kern), name='norm_psf')
    # k = tf.expand_dims(kern, 0)
    k = fftshift2d_tf(kern)
    k_f = tf.fft2d(tf.complex(k, 0.0), name='fft_k_wiener')
    x_f = tf.fft2d(tf.complex(img, 0.0), name='fft_x_wiener')

    y = tf.expand_dims(y, 0, name='expandDims_wiener')
    # operation - Wiener deconvolution
    inverse_k = tf.divide(tf.conj(k_f), tf.conj(k_f) * k_f + tf.complex(y, 0.0), name='inverseKernel_wiener')
    dconv = tf.multiply(x_f, inverse_k, name='conv_fft_wiener')
    dconv_r = tf.real(tf.ifft2d(dconv, name='ifft_wiener'))
    if clip_values:
        dconv_r = tf.clip_by_value(dconv_r, 0., 1.0, name='clip_wiener')
    return dconv_r


# Shorthand for a conv2D layer with ReLU on input --> the other way around compared to standard layer.
# Dilation is supported.
def reluconv2d(name, x, channels, kernel_size=3, stride=1, dilation=1, use_bn=False, feature_factor=1.0):
    x = tf.nn.relu(x, name='%s_relu' % name)
    x = Conv2D('%s_conv2D' % name, x, int(channels / feature_factor), kernel_size=kernel_size, stride=stride,
               dilation_rate=dilation)
    # x = Conv2D('%s_conv2D' % name, x, int(channels / feature_factor), kernel_size=kernel_size, stride=stride,
    #            dilation_rate=1)
    if use_bn:
        x = BatchRenorm('%s_bn' % name, x, rmax=3, dmax=5)
    return x


# Shorthand for a conv2DTranspose (Deconv) layer with ReLU on input --> the other way around compared to standard layer.
def reluconv2dTranspose(name, x, channels, kernel_size=3, stride=1, use_bn=False, feature_factor=1.0):
    x = tf.nn.relu(x, name='%s_relu' % name)
    x = Deconv2D('%s_conv2D' % name, x, int(channels / feature_factor), kernel_shape=kernel_size, stride=stride)
    # x = Conv2DTranspose('%s_conv2D' % name, x, int(channels / feature_factor), kernel_size=kernel_size, stride=stride)
    if use_bn:
        x = BatchRenorm('%s_bn' % name, x,  rmax=3, dmax=5)
    return x


# sub pixel shuffle operation as described at
# https://github.com/tetrachrome/subpixel
def _phase_shift(i, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = i.get_shape().as_list()
    bsize = tf.shape(i)[0]
    x = tf.reshape(i, [bsize, a, b, r, r])
    x = tf.transpose(x, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    x = tf.split(x, a, 1)  # a, [bsize, b, r, r]
    x = tf.concat([tf.squeeze(xi) for xi in x], 2)  # bsize, b, a*r, r
    x = tf.split(x, b, 1)  # b, [bsize, a*r, r]
    x = tf.concat([tf.squeeze(xi) for xi in x], 2)  # bsize, a*r, b*r
    return tf.reshape(x, [bsize, a*r, b*r, 1])


# Upsampling using pixel shift as proposed in https://arxiv.org/abs/1609.05158.
def relu_psupsanple(name, x, channels, kernel_size=3, stride=1, scaling_factor=2, use_bn=False, feature_factor=1.0):
    x = tf.nn.relu(x, name='%s_relu' % name)
    channel_no = int(channels / feature_factor)
    channel_no = channel_no * (scaling_factor * scaling_factor)
    x = Conv2D('%s_conv2D' % name, x, channel_no, kernel_shape=kernel_size, stride=stride)
    x = tf.depth_to_space(x, scaling_factor, name='%s_shuffle' % name)
    # x = Conv2DTranspose('%s_conv2D' % name, x, int(channels / feature_factor), kernel_size=kernel_size, stride=stride)
    if use_bn:
        x = BatchRenorm('%s_bn' % name, x,  rmax=3, dmax=5)
    return x


def unet3(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d2_c3', c, 256, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d2_c4', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')

    # block 3 h/4 x w/4 --> h/2 x w/2
    with tf.name_scope('upblock1'):
        c = reluconv2dTranspose('u1_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
        c = tf.add(c, skip_connections[1])
        block_in = c
        # optional: add dilation
        c = reluconv2d('u1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('u1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='upblock1_residual')

    # block 4 h/2 x w/2 --> h x w
    with tf.name_scope('upblock2'):
        # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
        c = reluconv2dTranspose('u2_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
        # c = tf.add(c, skip_connections[0])
        block_in = c
        c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
        c = reluconv2d('u2_c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('u2_c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='upblock2_residual')

    # output layers
    # c = ConcatWith('skip_connection2', c, [enc_activations], 3)
    c = reluconv2d('c3', c, 32, feature_factor=feature_factor)
    dec_out = reluconv2d('conv_output', c, out_channels)
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet4(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock3_residual')

    # block 3 h/8 x w/8 --> h/4 x w/4
    with tf.name_scope('upblock1'):
        c = reluconv2dTranspose('u1_c0', c, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
        c = tf.add(c, skip_connections[2])
        block_in = c
        # optional: add dilation
        c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='upblock1_residual')

    # block 4 h/4 x w/4 --> h/2 x w/2
    with tf.name_scope('upblock2'):
        c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
        c = tf.add(c, skip_connections[1])
        block_in = c
        # optional: add dilation
        c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='upblock2_residual')

    # block 5 h/2 x w/2 --> h x w
    with tf.name_scope('upblock3'):
        # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
        c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
        # c = tf.add(c, skip_connections[0])
        block_in = c
        c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
        c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='upblock3_residual')

    # output layers
    # c = ConcatWith('skip_connection2', c, [enc_activations], 3)
    c = reluconv2d('c3', c, 32, feature_factor=feature_factor)
    dec_out = reluconv2d('conv_output', c, out_channels)
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet5(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        # c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        # c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        # c = reluconv2d('d3_c4', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d3_c5', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock3_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c_dec1 = tf.add(c, block_in, name='upblock3_residual')

    # upsampling network 2
    with tf.name_scope('upnet2'):
        with tf.variable_scope("upnet2"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c_dec2 = tf.add(c, block_in, name='upblock3_residual')

    # output layers
    c = ConcatWith('concat_outblock', c_dec1, [c_dec2], 3)
    c = reluconv2d('c3', c, 32, feature_factor=feature_factor)
    dec_out = reluconv2d('conv_output', c, out_channels)
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet5_reduced(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d3_c5', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c6', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock3_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up1_c3', c, 32, feature_factor=feature_factor)
            c_dec1 = reluconv2d('conv_output', c, out_channels)

    # # upsampling network 2
    # with tf.name_scope('upnet2'):
    #     with tf.variable_scope("upnet2"):
    #         # block 3 h/8 x w/8 --> h/4 x w/4
    #         with tf.name_scope('upblock1'):
    #             c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             c = tf.add(c, skip_connections[2])
    #             block_in = c
    #             # optional: add dilation
    #             c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
    #             c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock1_residual')
    #
    #         # block 4 h/4 x w/4 --> h/2 x w/2
    #         with tf.name_scope('upblock2'):
    #             c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             c = tf.add(c, skip_connections[1])
    #             block_in = c
    #             # optional: add dilation
    #             c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
    #             c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock2_residual')
    #
    #         # block 5 h/2 x w/2 --> h x w
    #         with tf.name_scope('upblock3'):
    #             # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
    #             c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             # c = tf.add(c, skip_connections[0])
    #             block_in = c
    #             c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
    #             c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
    #             c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock3_residual')
    #
    #         c = reluconv2d('up2_c3', c, 32, feature_factor=feature_factor)
    #
    #         c_dec2 = reluconv2d('conv_output', c, out_channels)

    # output layers
    # dec_out = ConcatWith('concat_outblocks', c_dec1, [c_dec2], 3)
    dec_out = c_dec1
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet5_ps(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c5', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d3_c6', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock3_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = relu_psupsanple('u1_c0', c_enc, 256, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = relu_psupsanple('u2_c0', c, 128, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = relu_psupsanple('u3_c0', c, 64, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up1_c3', c, 32, feature_factor=feature_factor)
            c_dec1 = reluconv2d('conv_output', c, out_channels)

    # # upsampling network 2
    # with tf.name_scope('upnet2'):
    #     with tf.variable_scope("upnet2"):
    #         # block 3 h/8 x w/8 --> h/4 x w/4
    #         with tf.name_scope('upblock1'):
    #             c = relu_psupsanple('u1_c0', c_enc, 256, feature_factor=feature_factor)
    #             c = tf.add(c, skip_connections[2])
    #             block_in = c
    #             # optional: add dilation
    #             c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
    #             c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock1_residual')
    #
    #         # block 4 h/4 x w/4 --> h/2 x w/2
    #         with tf.name_scope('upblock2'):
    #             c = relu_psupsanple('u2_c0', c, 128, feature_factor=feature_factor)
    #             c = tf.add(c, skip_connections[1])
    #             block_in = c
    #             # optional: add dilation
    #             c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
    #             c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock2_residual')
    #
    #         # block 5 h/2 x w/2 --> h x w
    #         with tf.name_scope('upblock3'):
    #             # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
    #             c = relu_psupsanple('u3_c0', c, 64, feature_factor=feature_factor)
    #             # c = tf.add(c, skip_connections[0])
    #             block_in = c
    #             c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
    #             c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
    #             c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock3_residual')
    #
    #         c = reluconv2d('up2_c3', c, 32, feature_factor=feature_factor)
    #
    #         c_dec2 = reluconv2d('conv_output', c, out_channels)

    # output layers
    # dec_out = ConcatWith('concat_outblocks', c_dec1, [c_dec2], 3)
    dec_out = c_dec1
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Standard CNN architecture with skip connection inspired by Henz et al.
#       "Deep Joint Design of Color Filter Arrays and Demosaicing"
#       http://inf.ufrgs.br/~bhenz/projects/joint_cfa_demosaicing/
def plain_cnn(activations, out_channels, feature_factor=1.0, n_layers=8):
    skip_connections = []
    with argscope([Conv2D], activation=tf.nn.relu, kernel_size=3, strides=1):
        c = activations
        for n in range(n_layers):

            if n < int(n_layers / 2):
                c = Conv2D('conv_%i' % n, c, int(64 / feature_factor))
            else:
                if n == int(n_layers / 2):
                    c = ConcatWith('concat_mid', c, [c, activations], 3)
                    skip_connections.append(c)
                c = Conv2D('conv_%i' % n, c, int(128 / feature_factor))

        # skip connection: concat with input
        cc = ConcatWith('concat_output', c, [activations, skip_connections[0]], 3)
        dec_out = Conv2D('conv_output', cc, out_channels)
        # shift range back - not necessary
    return dec_out


# standard CNN architecture with skip connection inspired by Henz et al.
#       "Deep Joint Design of Color Filter Arrays and Demosaicing"
#       http://inf.ufrgs.br/~bhenz/projects/joint_cfa_demosaicing/
# only one skip connection from start to end
def plain_cnn_reduced(activations, out_channels, feature_factor=1.0, n_layers=8):
    skip_connections = []
    with argscope([Conv2D], activation=tf.nn.relu, kernel_size=3, strides=1):
        c = activations
        for n in range(n_layers):

            if n < int(n_layers / 2):
                c = Conv2D('conv_%i' % n, c, int(64 / feature_factor))
            else:
                if n == int(n_layers / 2):
                    c = ConcatWith('concat_mid', c, [c, activations], 3)
                    skip_connections.append(c)
                c = Conv2D('conv_%i' % n, c, int(128 / feature_factor))

        # skip connection: concat with input
        cc = ConcatWith('concat_output', c, [activations], 3)
        dec_out = Conv2D('conv_output', cc, out_channels)
        # shift range back - not necessary
    return dec_out


# performs n iterations of deconvolution via Wiener filtering in frequency space, y is a regularizer
def wiener_deconvolution_iterative(img, kern, iter, y, clip_values=True):
    print(img.shape)
    print(kern.shape)
    # renormalize psf, just in case
    kern = tf.divide(kern, tf.reduce_sum(kern), name='norm_psf')
    img_shape = img.shape.as_list()
    pad = img_shape[1]//2
    img = tf.pad(img, [[0, 0], [pad, pad], [pad, pad]], 'SYMMETRIC')
    kern = tf.pad(kern, [[0, 0], [pad, pad], [pad, pad]], 'CONSTANT')

    deblurred = wiener_iteration(img, img, kern, y)
    if clip_values:
        deblurred = tf.clip_by_value(deblurred, 0., 1.0, name='clip_wiener')
    for i in range(0, iter):
        deblurred = wiener_iteration(img, deblurred, kern, y)
        if clip_values:
            deblurred = tf.clip_by_value(deblurred, 0., 1.0)

    return deblurred[:, pad:-pad, pad:-pad]


def wiener_iteration(img, estimate, kern, y):
    # k = tf.expand_dims(kern, 0)
    k = fftshift2d_tf(kern)
    k_f = tf.fft2d(tf.complex(k, 0.0))
    x_f = tf.fft2d(tf.complex(img, 0.0))

    y = tf.expand_dims(y, 0)

    adj_conv = x_f * tf.conj(k_f)
    numerator = adj_conv + tf.fft2d(tf.complex(y * estimate, 0.))

    kernel_mag = tf.square(tf.abs(k_f))

    denominator = tf.complex(kernel_mag + y, 0.)
    filtered = tf.divide(numerator, denominator)

    deconv_r = tf.real(tf.ifft2d(filtered))
    return deconv_r


# Unet like architecture with one more resolution level compared to unet3.
def unet6_reduced(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c3', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c5', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c6', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock3_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up1_c3', c, 64, feature_factor=feature_factor)
            c_dec1 = reluconv2d('conv_output', c, out_channels)

    # upsampling network 2
    with tf.name_scope('upnet2'):
        with tf.variable_scope("upnet2"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up2_c3', c, 32, feature_factor=feature_factor)

            c_dec2 = reluconv2d('conv_output', c, out_channels)

    # output layers
    dec_out = ConcatWith('concat_outblocks', c_dec1, [c_dec2], 3)
    # dec_out = c_dec1
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet7(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c5', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d3_c6', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock3_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            # block 6 h x w --> h*2 x w*2
            with tf.name_scope('upblock4'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                tensor_shapes = c.get_shape().as_list()
                c = relu_psupsanple('u4_c0', c, 64, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                crop = tensor_shapes[1]//2
                c = c[:, crop:-crop, crop:-crop, :]
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u4_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u4_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up1_c4', c, 64, feature_factor=feature_factor)
            c_dec1 = reluconv2d('conv_output', c, out_channels)

    # # upsampling network 2
    # with tf.name_scope('upnet2'):
    #     with tf.variable_scope("upnet2"):
    #         # block 3 h/8 x w/8 --> h/4 x w/4
    #         with tf.name_scope('upblock1'):
    #             c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             c = tf.add(c, skip_connections[2])
    #             block_in = c
    #             # optional: add dilation
    #             c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
    #             c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock1_residual')
    #
    #         # block 4 h/4 x w/4 --> h/2 x w/2
    #         with tf.name_scope('upblock2'):
    #             c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             c = tf.add(c, skip_connections[1])
    #             block_in = c
    #             # optional: add dilation
    #             c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
    #             c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock2_residual')
    #
    #         # block 5 h/2 x w/2 --> h x w
    #         with tf.name_scope('upblock3'):
    #             # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
    #             c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             # c = tf.add(c, skip_connections[0])
    #             block_in = c
    #             c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
    #             c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
    #             c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock3_residual')
    #
    #         c = reluconv2d('up2_c3', c, 32, feature_factor=feature_factor)
    #
    #         c_dec2 = reluconv2d('conv_output', c, out_channels)

    # output layers
    # dec_out = ConcatWith('concat_outblocks', c_dec1, [c_dec2], 3)
    dec_out = c_dec1
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet7_deconv(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c5', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d3_c6', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock3_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            # block 6 h x w --> h*2 x w*2
            with tf.name_scope('upblock4'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                tensor_shapes = c.get_shape().as_list()
                c = reluconv2dTranspose('u4_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                crop = tensor_shapes[1]//2
                c = c[:, crop:-crop, crop:-crop, :]
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u4_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u4_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up1_c4', c, 64, feature_factor=feature_factor)
            c_dec1 = reluconv2d('conv_output', c, out_channels)

    # # upsampling network 2
    # with tf.name_scope('upnet2'):
    #     with tf.variable_scope("upnet2"):
    #         # block 3 h/8 x w/8 --> h/4 x w/4
    #         with tf.name_scope('upblock1'):
    #             c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             c = tf.add(c, skip_connections[2])
    #             block_in = c
    #             # optional: add dilation
    #             c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
    #             c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock1_residual')
    #
    #         # block 4 h/4 x w/4 --> h/2 x w/2
    #         with tf.name_scope('upblock2'):
    #             c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             c = tf.add(c, skip_connections[1])
    #             block_in = c
    #             # optional: add dilation
    #             c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
    #             c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock2_residual')
    #
    #         # block 5 h/2 x w/2 --> h x w
    #         with tf.name_scope('upblock3'):
    #             # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
    #             c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
    #             # c = tf.add(c, skip_connections[0])
    #             block_in = c
    #             c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
    #             c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
    #             c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
    #             c = tf.add(c, block_in, name='upblock3_residual')
    #
    #         c = reluconv2d('up2_c3', c, 32, feature_factor=feature_factor)
    #
    #         c_dec2 = reluconv2d('conv_output', c, out_channels)

    # output layers
    # dec_out = ConcatWith('concat_outblocks', c_dec1, [c_dec2], 3)
    dec_out = c_dec1
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet7_full(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c5', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d3_c6', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock3_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            # block 6 h x w --> h*2 x w*2
            with tf.name_scope('upblock4'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                tensor_shapes = c.get_shape().as_list()
                c = reluconv2dTranspose('u4_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                crop = tensor_shapes[1]//2
                c = c[:, crop:-crop, crop:-crop, :]
                block_in = c
                c = ConcatWith('skip_connection_concat2', c, [skip_connections[0]], 3)
                c = reluconv2d('u4_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u4_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock4_residual')

            c = reluconv2d('up1_c4', c, 64, feature_factor=feature_factor)
            c_dec1 = reluconv2d('conv_output', c, out_channels)

    # upsampling network 2
    with tf.name_scope('upnet2'):
        with tf.variable_scope("upnet2"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up2_c3', c, 32, feature_factor=feature_factor)

            c_dec2 = reluconv2d('conv_output', c, out_channels)

    # output layers
    dec_out = ConcatWith('concat_outblocks', c_dec1, [c_dec2], 3)
    # dec_out = c_dec1
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet8(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # activations = activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        c = reluconv2d('c3', c, 1, feature_factor=1)
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        c = reluconv2d('d1_c3', c, 1, feature_factor=1)
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c3', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        c = reluconv2d('d2_c4', c, 1, feature_factor=1)
        skip_connections.append(c)

    # block 3 h/4 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d3_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c4', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d3_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d3_c5', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d3_c6', c, 512, dilation=16, feature_factor=feature_factor)
        c = reluconv2d('d3_c7', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock3_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = ConcatWith('skip_connection_concat', c, [activations], 3)
            c = reluconv2d('up1_c4', c, 32, feature_factor=feature_factor)
            c_dec1 = reluconv2d('conv_output', c, out_channels)

    # upsampling network 2
    with tf.name_scope('upnet2'):
        with tf.variable_scope("upnet2"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[1])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 64, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[0]], 3)
                c = reluconv2d('u3_c1', c, 64, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 64, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = ConcatWith('skip_connection_concat', c, [activations], 3)
            c = reluconv2d('up1_c4', c, 32, feature_factor=feature_factor)

            c_dec2 = reluconv2d('conv_output', c, out_channels)

    # output layers
    dec_out = ConcatWith('concat_outblocks', c_dec1, [c_dec2], 3)
    # dec_out = c_dec1
    dec_out = tf.nn.relu(dec_out)

    return dec_out


# Unet like architecture with one more resolution level compared to unet3.
def unet8_small(activations, out_channels, feature_factor=1.0):
    skip_connections = []
    # enc_activations = enc_activations * 2.0 - 1.0

    with tf.name_scope('preblock'):
        c = activations
        # preprocessing layer
        c = reluconv2d('c0', c, 64, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('c1', c, 64, feature_factor=feature_factor)
        c = reluconv2d('c2', c, 64, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='preblock_residual')
        skip_connections.append(c)

    # block 1 hxw --> h/2 x w/2
    with tf.name_scope('downblock1'):
        c = reluconv2d('d1_c0', c, 128, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d1_c1', c, 128, feature_factor=feature_factor)
        c = reluconv2d('d1_c2', c, 128, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock1_residual')
        skip_connections.append(c)

    # block 2 h/2 x w/2 --> h/4 x w/4
    with tf.name_scope('downblock2'):
        c = reluconv2d('d2_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d2_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d2_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock2_residual')
        skip_connections.append(c)

    # block 4 h/2 x w/4 --> h/8 x w/8
    with tf.name_scope('downblock3'):
        c = reluconv2d('d3_c0', c, 256, stride=2, feature_factor=feature_factor)
        block_in = c
        c = reluconv2d('d3_c1', c, 256, feature_factor=feature_factor)
        c = reluconv2d('d3_c2', c, 256, feature_factor=feature_factor)
        c = tf.add(c, block_in, name='downblock3_residual')
        skip_connections.append(c)

    # block 3 h/8 x w/8 --> h/16 x w/16
    with tf.name_scope('downblock4'):
        c = reluconv2d('d4_c0', c, 512, stride=2, feature_factor=feature_factor)
        block_in = c
        # optional: add dilation
        c = reluconv2d('d4_c1', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d4_c4', c, 512, feature_factor=feature_factor)
        c = reluconv2d('d4_c2', c, 512, dilation=2, feature_factor=feature_factor)
        c = reluconv2d('d4_c3', c, 512, dilation=4, feature_factor=feature_factor)
        c = reluconv2d('d4_c5', c, 512, dilation=8, feature_factor=feature_factor)
        c = reluconv2d('d4_c6', c, 512, dilation=16, feature_factor=feature_factor)
        # c = reluconv2d('d4_c2', c, 512, feature_factor=feature_factor)
        # c = reluconv2d('d4_c3', c, 512, feature_factor=feature_factor)
        # c = reluconv2d('d4_c5', c, 512, feature_factor=feature_factor)
        # c = reluconv2d('d4_c6', c, 512, feature_factor=feature_factor)
        c_enc = tf.add(c, block_in, name='downblock4_residual')

    # upsampling network 1
    with tf.name_scope('upnet1'):
        with tf.variable_scope("upnet1"):
            # block 3 h/16 x w/16 --> h/8 x w/8
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[3])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 128, stride=2, kernel_size=4, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[1]], 3)
                c = reluconv2d('u3_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up1_c4', c, 64, feature_factor=feature_factor)
            c_dec1 = reluconv2d('conv_output', c, out_channels)

    # upsampling network 2
    with tf.name_scope('upnet2'):
        with tf.variable_scope("upnet2"):
            # block 3 h/8 x w/8 --> h/4 x w/4
            with tf.name_scope('upblock1'):
                c = reluconv2dTranspose('u1_c0', c_enc, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[3])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u1_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u1_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock1_residual')

            # block 4 h/4 x w/4 --> h/2 x w/2
            with tf.name_scope('upblock2'):
                c = reluconv2dTranspose('u2_c0', c, 256, stride=2, kernel_size=4, feature_factor=feature_factor)
                c = tf.add(c, skip_connections[2])
                block_in = c
                # optional: add dilation
                c = reluconv2d('u2_c1', c, 256, feature_factor=feature_factor)
                c = reluconv2d('u2_c2', c, 256, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock2_residual')

            # block 5 h/2 x w/2 --> h x w
            with tf.name_scope('upblock3'):
                # c = ConcatWith('skip_connection1', c, [skip_connections[1]], 3)
                c = reluconv2dTranspose('u3_c0', c, 128, stride=2, kernel_size=5, feature_factor=feature_factor)
                # c = tf.add(c, skip_connections[0])
                block_in = c
                c = ConcatWith('skip_connection_concat', c, [skip_connections[1]], 3)
                c = reluconv2d('u3_c1', c, 128, feature_factor=feature_factor)
                c = reluconv2d('u3_c2', c, 128, feature_factor=feature_factor)
                c = tf.add(c, block_in, name='upblock3_residual')

            c = reluconv2d('up2_c3', c, 64, feature_factor=feature_factor)

            c_dec2 = reluconv2d('conv_output', c, out_channels)

    # output layers
    dec_out = ConcatWith('concat_outblocks', c_dec1, [c_dec2], 3)
    # dec_out = c_dec1
    dec_out = tf.nn.relu(dec_out)

    return dec_out
