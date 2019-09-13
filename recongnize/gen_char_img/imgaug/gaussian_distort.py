#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: distort.py
# $Date: Wed Oct 28 13:02:37 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import cv2
import numpy as np
# from pykdtree.kdtree import KDTree
from scipy.ndimage import filters, interpolation


def _do_gaussian_distortion_single_channel(gray_img, nxys, mode, cval):
    output_shape = (gray_img.shape[0], gray_img.shape[1])
    return interpolation.map_coordinates(gray_img, nxys, order=1, mode=mode, cval=cval).reshape(output_shape)


def gaussian_distortion(img, xys, mode='constant', cval=0):
    if img.ndim == 3:
        return cv2.merge(
            [_do_gaussian_distortion_single_channel(gray_img, xys, mode, cval) for gray_img in cv2.split(img)])
    else:
        ret = _do_gaussian_distortion_single_channel(img, xys, mode, cval)
    return ret


def warp_perspective_point(point, mat):
    assert len(point) == 2, point
    mat = np.array(mat)
    assert mat.ndim == 2 and mat.shape == (3, 3), mat
    r = np.dot(mat, [point[0], point[1], 1])
    return r[0] / r[2], r[1] / r[2]


class GaussianDistortion:
    def __init__(self,
                 rng,
                 distort_range,
                 dsigma_range,
                 cval=0,
                 mode='constant',
                 change_every_iter=50,
                 fprop_distorted_coords=False):
        self.distort_range = distort_range
        self.dsigma_range = dsigma_range
        self.cval = cval
        self.mode = mode
        self.change_every_iter = change_every_iter
        self.fprop_distorted_coords = fprop_distorted_coords
        self._rng = rng
        self.count = 0

    def distort_img(self, img):
        return self._augment_given_image(img, self._get_augment_param_by_image(img))

    def _get_augment_param_by_image(self, img):
        param = dict()
        dsigma = self._rng.rand() * (self.dsigma_range[1] - self.dsigma_range[0])
        distort = self._rng.rand() * (self.distort_range[1] - self.distort_range[0])
        if self.count == 0 or self.last_shape != img.shape[:2]:
            h, w = img.shape[:2]
            xys = [filters.gaussian_filter(xs, dsigma) for xs in self._rng.randn(2, h, w)]
            self.xys = xys
            xys = [xs * distort / np.amax(xs) for xs in xys]
            dx = xys[0]
            dy = xys[1]
            self.nxys = np.asarray([(x + dx[x, y], y + dy[x, y]) for x in range(h) for y in range(w)]).T
            self.last_shape = img.shape[:2]
            if self.fprop_distorted_coords:
                coords = np.array((self.nxys[1], self.nxys[0])).swapaxes(1, 0).reshape(h, w, 2).astype('float32')

                src = np.array([coords[0, 0], coords[0, -1], coords[-1, -1], coords[-1, 0]])
                dst = np.array([(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0])],
                               dtype='float32')

                self.global_perspective_mat = cv2.getPerspectiveTransform(src, dst)
                self. ree = self._build_kdtree(coords)
            self.count = 0
        self.count = (self.count + 1) % self.change_every_iter

        if self.fprop_distorted_coords:
            param['kdtree'] = self.kdtree
            param['shape'] = self.last_shape
            param['global_perspective_mat'] = self.global_perspective_mat

        param.update(dsigma=dsigma, distort=distort, nxys=self.nxys, mode=self.mode, cval=self.cval)
        return param

    def _augment_given_image(self, img, param):
        nxys, mode, cval = [param[name] for name in ['nxys', 'mode', 'cval']]
        ret = gaussian_distortion(img, nxys, mode, cval)
        return ret

    def _build_kdtree(self, coords):
        rect_center = coords.copy()
        rect_center[:, :-1] += coords[:, 1:]
        rect_center[:-1, :] += coords[1:, :]
        rect_center[:-1, :-1] += coords[1:, 1:]
        rect_center *= 0.25
        return KDTree(rect_center[:-1, :-1].reshape(-1, 2))

    def _fprop_coords(self, coords, param):
        if 'kdtree' not in param:
            return coords
        kdtree = param['kdtree']
        shape = param['shape']
        nxys = param['nxys']
        mat = param['global_perspective_mat']

        nr_prefetch = 10
        _, indexes = kdtree.query(np.asarray(coords, dtype='float32'), k=nr_prefetch)
        return [self._find_coord(coords[i], indexes[i], nxys, shape, kdtree, mat) for i in range(len(indexes))]

    def _coord_in_mask(self, coord, mask):
        x, y = list(map(int, coord))
        h, w = mask.shape
        if not (x >= 0 and x < w and y >= 0 and y < h):
            return False
        return mask[y, x] > 0

    def _find_coord(self, coord, indexes, nxys, shape, kdtree, mat):
        start = 0
        while start < len(nxys[0]):
            for idx in indexes[start:]:
                i = idx / (shape[1] - 1)
                j = idx % (shape[1] - 1)
                poly = self._get_poly_by_top_left_index(i, j, nxys, shape[1])
                if self._poly_contains(poly, coord):
                    return self._get_perspective_coord(coord, poly, [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)])
            # XXX: Can not find a quadrangle containing this point in the
            # nearest neighbors, and we do not inspect further on nearest
            # neighbors for efficiency concenrs.
            # Fallback to apply global perspective transform.
            return warp_perspective_point(coord, mat)

    def _poly_contains(self, poly, coord):
        vec = np.array(poly) - coord
        cp = np.cross(vec, np.roll(vec, 1, axis=0))
        if (cp >= 0).all() or (cp <= 0).all():
            return True
        return False

    def _get_perspective_coord(self, coord, quad0, quad1):
        mat = cv2.getPerspectiveTransform(np.array(quad0, dtype='float32'), np.array(quad1, dtype='float32'))
        return warp_perspective_point(coord, mat)

    def _get_poly_by_top_left_index(self, i, j, nxys, w):
        p = i * w + j
        return [(nxys[1][idx], nxys[0][idx]) for idx in [p, p + 1, p + w + 1, p + w]]


# vim: foldmethod=marker
