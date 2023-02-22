import numpy as np
import cv2

# ------------------------------------
# Global Settings
# ------------------------------------

BATCH = 1                # number of sequences per update
SHAPE = 3200             # dimension of image patches
INSHAPE = SHAPE            # dimension of input shape if different from shape
FEATURE_FACTOR = 1       # scaling factor for feature number

# ZERNIKE_MODES = 2        # used bands for of Zernike polynomials
SENSOR_RES = SHAPE         # Resolution of sensor (one dimension) assuming a square sensor, so actual resolution is square
SENSOR_PITCH = 8.0e-6    # Pixel distance in m - using the pitch of the SLM pixels here for now
SLM_RES = SHAPE            # Resolution of SLM (one dimension), assuming a square SLM or a cropped/scaled display,
L_SLM = 15.36e-3 + 0.96e-3  # 8.64e-3          # length of one smaller size of SLM in m
                         # prototype is Full HD
F_NUMBER = 5.6            # aperture opening 1/stops
L_SENSOR = 15.15e-3  # 0.75     # length of one size of camera sensor in m
FOCAL_LENGTH = .1      # focal length in m of the lens right behind the SLM
FOCUS_DISTANCE = -1    # distance of focus plane from pupil of optical element in m

# distance between optical elements and image plane (z)
if FOCUS_DISTANCE == -1:
    Z_DISTANCE = FOCAL_LENGTH
else:
    Z_DISTANCE = np.abs((FOCUS_DISTANCE * FOCAL_LENGTH) / (FOCUS_DISTANCE - FOCAL_LENGTH))

# diameter of aperture (region of SLM)
APERTURE = FOCAL_LENGTH / F_NUMBER  # effective aperture of the lens (diameter of entrance pupil)
                         # circular aperture is 8.64e-3 (L_SLM), rectangulat 12.219e-3

N_LAYERS = 8             # number of convolution layers in decoder network, must be multple of 2, added one
                         # additional layer for output so it's currently N +1 conv layers
CHANNELS = 1             # image channels used, e.g. 1 for greyscale, 3 for rgb
WAVELENGTH = 5.5e-7    # wavelengths corresponding to channels in m

SLM_CROP = True


def linear2srgb(im):
    im_srgb = im
    alpha = 0.055
    im_srgb[im <= 0.0031308] *= 12.92
    bool_mask = im > 0.0031308
    im_srgb[bool_mask] = im_srgb[bool_mask] ** (1.0 / 2.4) * (1.0 + alpha) - alpha

    return im_srgb * 256.0


def srgb2linear(im):
    im_linear = im = im / 256.0
    alpha = 0.055
    im_linear[im <= 0.04045] *= 0.077399381
    bool_mask = im > 0.04045
    im_linear[bool_mask] = ((im_linear[bool_mask] + alpha) / (1 + alpha)) ** 2.4

    return im_linear


# convolution using fft numpy implementation - reference
def convolve_npfft(img, kernel):
    c_size = img.shape[0] + kernel.shape[0]
    im_pad = (c_size - img.shape[0])/2
    k_pad = (c_size - kernel.shape[0])/2
    # pad to avoid edge artifacts
    k = np.pad(kernel, ((k_pad, k_pad), (k_pad, k_pad)), 'constant', constant_values=(0, 0))
    # shift kernel to have position 0,0 of kernel at array position x, y
    k = np.fft.fftshift(k)
    # transform kernel
    k_f = np.fft.fft2(k)
    # pad to avoid edge artifacts
    im = np.pad(img, ((im_pad, im_pad), (im_pad, im_pad)), 'constant', constant_values=(0, 0))
    # transform image
    x_f = np.fft.fft2(im)
    print 'shape of fft:'
    print x_f.shape
    # do the convolution which corresponds to component wise multiplication in frequency domain
    conv = np.multiply(x_f, k_f)
    # transform back, only take real part, imag should be almost zero
    conv_op = np.real(np.fft.ifft2(conv))
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.imshow('convolved image', conv_op/np.max(conv_op))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # remove padding
    conv_op = conv_op[im_pad:-im_pad, im_pad:-im_pad]

    # output
    return conv_op


def rect(x, d):
    x = np.abs(x)
    y = np.zeros(x.shape)
    y[x < d/2.] = 1.0
    y[x == d/2] = 0.5
    return y


# fft wrappers
def ft2(g, delta):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * (delta*delta)


def ift2(g, delta):
    size = g.shape[0]
    factor = (size * delta) * (size * delta)
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(g))) * factor


# gaussian kernel for testing, parameters: length (size) and sigma
def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel1 = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel1 / np.sum(kernel1)


# variable grid angular spectrum propagation
def ang_spec_prop(u_in, wvl, d1, d2, z):
    n = u_in.shape[0]
    k = 2*np.pi / wvl
    coords1 = np.linspace(-n // 2 * d1, (n // 2 - 1) * d1, n)
    x1, y1 = np.meshgrid(coords1, coords1)
    r1sq = x1 * x1 + y1 * y1

    # spatial frequencies of source plane
    df = 1 / (n*d1)
    coordsf = np.linspace(-n // 2 * df, (n // 2 - 1) * df, n)
    fx1, fy1 = np.meshgrid(coordsf, coordsf)
    fsq = fx1 * fx1 + fy1 * fy1
    # scaling parameter
    m = d2 / d1

    # observation plane coordinates
    coords2 = np.linspace(-n // 2 * d2, (n // 2 - 1) * d2, n)
    x2, y2 = np.meshgrid(coords2, coords2)
    r2sq = x2 * x2 + y2 * y2

    # quadratic phase factors
    q1 = np.exp(1j*k/2*(1-m)/z*r1sq)
    q2 = np.exp(-1j*np.pi*np.pi*2*z/m/k*fsq)
    q3 = np.exp(1j*k/2*(m-1)/(m*z)*r2sq)

    # compute out field
    u_out = q3 * ift2(q2 * ft2(u_in/m * q1, d1), df)

    return u_out


# variable grid angular spectrum propagation
def ang_spec_prop_padded(u_in, wvl, d1, d2, z, window):
    n = u_in.shape[0]
    n_pad = n // 4
    k = 2*np.pi / wvl
    # source coordinates
    coords1 = np.linspace(-n // 2 * d1, (n // 2 - 1) * d1, n)
    x1, y1 = np.meshgrid(coords1, coords1)
    r1sq = x1 * x1 + y1 * y1

    # spatial frequencies of source plane
    n_padded = n + 2 * n_pad
    df = 1 / (n_padded*d1)
    coordsf = np.linspace(-n_padded // 2 * df, (n_padded // 2 - 1) * df, n_padded)
    fx1, fy1 = np.meshgrid(coordsf, coordsf)
    fsq = fx1 * fx1 + fy1 * fy1
    # scaling parameter
    m = d2 / d1

    # observation plane coordinates
    coords2 = np.linspace(-n // 2 * d2, (n // 2 - 1) * d2, n)
    x2, y2 = np.meshgrid(coords2, coords2)
    r2sq = x2 * x2 + y2 * y2

    # quadratic phase factors
    q1 = np.exp(1j*k/2*(1-m)/z*r1sq)
    q2 = np.exp(-1j*np.pi*np.pi*2*z/m/k*fsq)
    q3 = np.exp(1j*k/2*(m-1)/(m*z)*r2sq)

    # compute out field
    u = u_in/m * q1
    u_padded = np.pad(u, ((n_pad, n_pad), (n_pad, n_pad)), 'constant', constant_values=(0, 0))
    u = ft2(u_padded, d1)
    # cv2.namedWindow('uin', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('uin', 1024, 1024)
    # cv2.imshow('uin', np.real(u) / np.max(np.real(u)))
    # cv2.waitKey(0)
    u = ift2(u * q2, df)
    u_out = u[n_pad:-n_pad, n_pad:-n_pad] * q3
    # in one line - difficult to debug
    # u_out = q3 * ift2(q2 * ft2(u_in/m * q1, d1), df)

    return u_out


def get_psf(distance):
    k = 2*np.pi / WAVELENGTH
    d = APERTURE
    print 'Aperture: ' + str(d)

    delta = L_SLM/SHAPE
    coords = np.linspace(-SHAPE//2 * delta, (SHAPE//2-1) * delta, SHAPE)
    x, y = np.meshgrid(coords, coords)
    squared_sum1 = x * x + y * y

    delta1 = d / SHAPE
    coords1 = np.linspace(-SHAPE // 2 * delta1, (SHAPE // 2 - 1) * delta1, SHAPE)
    x1, y1 = np.meshgrid(coords1, coords1)
    squared_sum2 = x1 * x1 + y1 * y1
    # window to avoid anti-aliasing
    win = np.exp(-np.power(coords1/(0.1*d), 8))
    window = np.outer(win, np.transpose(win))

    x_length = d/2048 * 1920
    y_length = d/2048 * 1080
    crop = rect(x1, x_length) * rect(y1, y_length)


    # source spherical wave
    r = np.sqrt(squared_sum1 + distance * distance)
    phi_d = k * r
    u1 = 1. * np.exp(1j * phi_d) / r
    u1 = u1 * crop
    # propagation to lens
    u1 = ang_spec_prop(u1, WAVELENGTH, delta, delta1, 0.112)
    # u1 = ang_spec_prop(u1, WAVELENGTH, delta, delta1, 0.112)

    # debug out
    # print k
    # lens element
    arg2 = k / (2*FOCAL_LENGTH)
    phi = np.exp(-1j * arg2 * squared_sum2)
    u2 = phi * u1

    # Sensor plane
    delta2 = L_SENSOR / SHAPE

    # cv2.imshow('window_Image', window)
    # cv2.waitKey(0)
    # single propagation
    # u3 = ang_spec_prop(u2, WAVELENGTH, delta1, delta2, Z_DISTANCE)
    # propagation to sensor
    u3 = ang_spec_prop(u2, WAVELENGTH, delta1, delta2, Z_DISTANCE)

    psf = np.abs(u3) * np.abs(u3)
    psf = psf / np.sum(psf)
    # cv2.namedWindow('crop', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('crop', 1024, 1024)
    cv2.imshow('crop', psf)
    cv2.waitKey(0)
    crop = SLM_RES // 2 - 348
    return psf[crop:-crop, crop:-crop]

# paths to in- and output image
# open image
finput = '/Volumes/Samsung_T5/psfDesign_repo/camera_test1/tifs/non_modulated_debayer_fullscale_cropped1k.tif'
output = '../focus_test/np_asTest_2prop_focus-2_2_4k_f56_100mm.jpg'
try:
    f = open(finput)
except IOError as e:
    print e.errno
    print finput
else:
    im_data = np.asarray(bytearray(f.read()), dtype=np.uint8)
    # reading image as greyscale
    image = cv2.imdecode(np.asarray(bytearray(im_data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # image = center_crop_image(image, SLM_RES)
    image = cv2.resize(image, (SENSOR_RES, SENSOR_RES), interpolation=cv2.INTER_CUBIC)
    image = srgb2linear(image)
    print 'in image stats:'
    print np.min(image)
    print np.max(image)

    # get psf
    # psf = get_psf(FOCUS_DISTANCE)
    # arbitrary distance
    psf = get_psf(2.)
    # convolve image
    out_img = convolve_npfft(image, psf)
    print 'out image stats:'
    print np.min(image)
    print np.max(out_img)
    out_img = linear2srgb(out_img)
    out_img = out_img.astype(np.uint8)
    cv2.imwrite(output, out_img)
    cv2.imwrite(output + '_psf.jpg', psf * 256)
    cv2.namedWindow('Decoded_Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Decoded_Image', 1024, 1024)
    cv2.imshow('Decoded_Image', out_img)
    cv2.imshow('PSF (normalized)', (psf/np.max(psf)))
    cv2.waitKey(0)
