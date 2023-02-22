import numpy as np
import cv2

# ------------------------------------
# Global Settings
# ------------------------------------

BATCH = 1                # number of sequences per update
SHAPE = 2048             # dimension of image patches
INSHAPE = SHAPE            # dimension of input shape if different from shape
FEATURE_FACTOR = 1       # scaling factor for feature number

# ZERNIKE_MODES = 2        # used bands for of Zernike polynomials
SENSOR_RES = SHAPE         # Resolution of sensor (one dimension) assuming a square sensor, so actual resolution is square
SENSOR_PITCH = 8.0e-6    # Pixel distance in m - using the pitch of the SLM pixels here for now
SLM_RES = SHAPE            # Resolution of SLM (one dimension), assuming a square SLM or a cropped/scaled display,
L_SLM = 8.64e-3          # length of one smaller size of SLM in m
# L_SLM = 15.38e-3
                         # prototype is Full HD
F_NUMBER = 8.           # aperture opening 1/stops
L_SENSOR = 15.15e-3      # length of one size of camera sensor in m
FOCUS_DISTANCE = 2.0    # distance of focus plane from pupil of optical element in m
# L_SENSOR = 24.0e-3        # shorter side of full format sensor
FOCAL_LENGTH = .035       # focal length in m of the lens right behind the SLM

# distance between optical elements and image plane (z)
Z_DISTANCE = np.abs((FOCUS_DISTANCE * FOCAL_LENGTH) / (FOCUS_DISTANCE - FOCAL_LENGTH))

# diameter of aperture (region of SLM)
APERTURE = FOCAL_LENGTH / F_NUMBER  # effective aperture of the lens (diameter of entrance pupil)
                         # circular aperture is 8.64e-3 (L_SLM), rectangulat 12.219e-3

N_LAYERS = 8             # number of convolution layers in decoder network, must be multple of 2, added one
                         # additional layer for output so it's currently N +1 conv layers
CHANNELS = 1             # image channels used, e.g. 1 for greyscale, 3 for rgb
WAVELENGTH = 5.5e-7    # wavelengths corresponding to channels in m


def linear2srgb(im):
    im_srgb = im
    alpha = 0.055
    im_srgb[im <= 0.0031308] *= 12.92
    bool_mask = im > 0.0031308
    im_srgb[bool_mask] = im_srgb[bool_mask] ** (1.0 /-
           2.4) * (1.0 + alpha) - alpha

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
    im_pad = int((c_size - img.shape[0])/2)
    k_pad = int((c_size - kernel.shape[0])/2)
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
    y = float(x < d/2.)
    y[x == d/2] = 0.5
    return y


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
    # u_in = u_in * (window + 1j*window)
    kernel2 = gkern(20, 10.0)
    u = u_in/m * q1
    u_padded = np.pad(u, ((n_pad, n_pad), (n_pad, n_pad)), 'constant', constant_values=(0, 0))
    window = np.pad(window, ((n_pad, n_pad), (n_pad, n_pad)), 'constant', constant_values=(0, 0))
    u = ft2(u_padded, d1) # * (window + 1j*window)
    # u = convolve_npfft(np.real(u), kernel2) + 1j * convolve_npfft(np.imag(u), kernel2)
    # u = np.fft.fft2(u_padded) # * (d1 * d1)
    cv2.namedWindow('uin', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('uin', 1024, 1024)
    cv2.imshow('uin', np.real(u) / np.max(np.real(u)))
    cv2.waitKey(0)
    u = ift2(u * q2, df)  # * (window + 1j*window)
    u_out = u[n_pad:-n_pad, n_pad:-n_pad] * q3
    # u_out = q3 * ift2(q2 * ft2(u_in/m * q1, d1), df)

    return u_out


def get_psf(distance):
    k = 2*np.pi / WAVELENGTH
    d = APERTURE
    print('Aperture: ' + str(d))

    delta1 = d/SHAPE
    coords1 = np.linspace(-SHAPE//2 * delta1, (SHAPE//2-1) * delta1, SHAPE)
    x1, y1 = np.meshgrid(coords1, coords1)
    sqaured_sum1 = x1 * x1 + y1 * y1

    # source spherical wave
    r = np.sqrt(sqaured_sum1 + distance * distance)
    phi_d = k * r
    u1 = 1. * np.exp(1j * phi_d) / r
    # debug out
    # print k
    # lens element
    arg2 = k / (2*FOCAL_LENGTH)
    phi = np.exp(-1j * arg2 * sqaured_sum1)
    u2 = phi * u1

    # Sensor plane
    delta2 = L_SENSOR / SHAPE

    # window to avoid anti-aliasing
    win = np.exp(-np.power(coords1/(0.45*d), 8))
    window = np.outer(win, np.transpose(win))
    # cv2.imshow('window_Image', window)
    # cv2.waitKey(0)
    # u3 = ang_spec_prop(u2, WAVELENGTH, delta1, delta2, Z_DISTANCE)
    # u3 = ang_spec_prop_padded(u2, WAVELENGTH, delta1, delta2*0.5, Z_DISTANCE*0.5, window) # * (window + 1j * window)
    # u3 = ang_spec_prop_padded(u3, WAVELENGTH, delta2*0.5, delta2, Z_DISTANCE*(1-0.5), window)
    # single propagation
    u3 = ang_spec_prop(u2, WAVELENGTH, delta1, delta2, Z_DISTANCE)

    psf = np.abs(u3) * np.abs(u3)
    psf = psf / np.sum(psf)
    return psf


# open image
finput = '/graphics/projects/scratch/chen/joint_design/camera_unit_test/gratings/patterns/slant_155.5088.png'
output = 'np_sim_results/simulated_psf.png'
try:
    f = open(finput)
except IOError as e:
    print(e.errno)
    print(finput)
else:
#    im_data = np.asarray(bytearray(f.read()), dtype=np.uint8)
#    # reading image as greyscale
#    image = cv2.imdecode(np.asarray(bytearray(im_data), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(finput)
    # image = center_crop_image(image, SLM_RES)
    image = srgb2linear(image)
    print(np.min(image))
    print(np.max(image))
    # cv2.imshow('Input_Image', image)
    # cv2.waitKey(0)
    # psf = get_psf(FOCUS_DISTANCE)
    psf = get_psf(.8)

    out_img = convolve_npfft(image, psf)
    print('out image stats:')
    print(np.min(image))
    print(np.max(out_img))
    out_img = linear2srgb(out_img)
    out_img = out_img.astype(np.uint8)
    cv2.imwrite(output, out_img)
    cv2.imwrite(output + '_psf.png', psf * 256)
    cv2.imshow('Decoded_Image', out_img)
    cv2.imshow('PSF (normalized)', (psf/np.max(psf))[500:550, 500:550])
    cv2.waitKey(0)
