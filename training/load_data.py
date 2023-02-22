from tensorpack import *
import cv2
import numpy as np


# decoding png in dataflow
class ImageDecode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            return cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
        super(ImageDecode, self).__init__(ds, func, index=index)


# linearize image data assumed to be in SRGB gamma, color primaries remain untouched
# output is in range 0 to 1 float
# https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_(%22gamma%22)
# def srgb2linear(im):
#     im = im.astype(np.float32)
#     im *= 0.00390625
#     im_linear = im
#     alpha = 0.055
#     im_linear[im <= 0.04045] *= 0.077399381
#     bool_mask = im > 0.04045
#     im_linear[bool_mask] += alpha
#     im_linear[bool_mask] *= 1/(1 + alpha)
#     im_linear[bool_mask] = np.power(im_linear[bool_mask], 2.4, out=im_linear[bool_mask])
#     # im_linear[bool_mask] **= 2.4
#
#     return im_linear


def srgb2linear(im):
    im = im.astype(np.float32)
    im /= 255.0
    im_linear = im
    alpha = 0.055
    im_linear[im <= 0.04045] *= 0.077399381
    bool_mask = im > 0.04045
    im_linear[bool_mask] = ((im_linear[bool_mask] + alpha) / (1 + alpha)) ** 2.4

    return np.clip(im_linear, 0, 1)


def srgb2float(im):
    im = im.astype(np.float32)
    im /= 255.0
    im_linear = im
    # alpha = 0.055
    # im_linear[im <= 0.04045] *= 0.077399381
    # bool_mask = im > 0.04045
    # im_linear[bool_mask] = ((im_linear[bool_mask] + alpha) / (1 + alpha)) ** 2.4

    return np.clip(im_linear, 0, 1)


def reject_small_data(im, minres):
    if (im.shape[0] >= minres) and (im.shape[1] >= minres):
        return im
    else:
        return None


def loaddata(lmdb_path):
    ds = LMDBData(lmdb_path, shuffle=False)
    # ds = PrefetchData(ds, 500, 1)
    ds = LMDBDataPoint(ds)
    ds = ImageDecode(ds, 0)
    ds = MapDataComponent(ds, lambda x: reject_small_data(x, 2048), 0)
    ds = AugmentImageComponent(ds, [imgaug.Grayscale(keepdims=True)], index=0)
    ds = MapDataComponent(ds, lambda x: srgb2linear(x), 0)
    # there is only one data component, return it
    return ds


def loaddata_rgb(lmdb_path):
    ds = LMDBData(lmdb_path, shuffle=False)
    # ds = PrefetchData(ds, 500, 1)
    ds = LMDBDataPoint(ds)
    ds = ImageDecode(ds, 0)
    ds = MapDataComponent(ds, lambda x: reject_small_data(x, 2048), 0)
    ds = AugmentImageComponent(ds, [imgaug.ColorSpace(cv2.COLOR_BGR2RGB)], index=0)
    ds = MapDataComponent(ds, lambda x: srgb2linear(x), 0)
    # there is only one data component, return it
    return ds


def loaddata_srgb(lmdb_path):
    ds = LMDBData(lmdb_path, shuffle=False)
    # ds = PrefetchData(ds, 500, 1)
    ds = LMDBDataPoint(ds)
    ds = ImageDecode(ds, 0)
    ds = MapDataComponent(ds, lambda x: reject_small_data(x, 2048), 0)
    ds = AugmentImageComponent(ds, [imgaug.Grayscale(keepdims=True)], index=0)
    ds = MapDataComponent(ds, lambda x: srgb2float(x), 0)
    # there is only one data component, return it
    return ds
