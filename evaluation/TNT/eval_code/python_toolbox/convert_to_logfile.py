#!/usr/bin/env python
#
# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# Python script to convert SfM data into the tanksandtempls log file format
# Example:
# python convert_to_logfile.py [COLMAP SfM file] \
#   						   [output log-filename] \
#				               [folder of the input images] \
#							   [COLMAP/MVE/VSFM] \
#							   [image format] \
# check https://www.tanksandtemples.org/tutorial/ for further examples

import sys
import os
import glob
import numpy as np
from numpy import matrix
import read_write_model as read_model


def write_SfM_log(T, i_map, filename):
    with open(filename, 'w') as f:
        ii = 0
        for t in T:
            p = t.tolist()
            f.write(' '.join(map(str, i_map[ii])) + '\n')
            f.write('\n'.join(' '.join(map('{0:.12f}'.format, p[i])) \
             for i in range(4)))
            f.write('\n')
            ii = ii + 1


def quat2rotmat(qvec):
    rotmat = np.array([1- 2 * qvec[2]**2 - 2 * qvec[3]**2, \
    2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], \
    2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2], \
	\
    2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], \
    1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, \
    2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1], \
	\
    2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], \
    2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], \
    1 - 2 * qvec[1]**2 - 2 * qvec[2]**2])
    rotmat = rotmat.reshape(3, 3)
    return rotmat


# Usage: convert_COLMAP_to_log([COLMAP SfM file], [output log-filename], \
#    						   [folder of the input images] [image format])
# Example: convert_COLMAP_to_log('sparse/0/', 'colmap.log', 'images/','jpg')
def convert_COLMAP_to_log(filename, logfile_out, input_images, formatp):
    dirname = os.path.dirname(filename)
    cameras, images, points3D = read_model.read_model(dirname, '.bin')
    jpg_list = glob.glob(input_images + '/*.' + formatp)
    jpg_list.sort()
    nr_of_images = len(jpg_list)

    T = []
    i_map = []
    TF = []
    i_mapF = []

    ii = 0
    for key, im in images.items():
        qvec = im[1]
        r = quat2rotmat(qvec)
        translation = im[2]
        w = np.zeros((4, 4))
        w[3, 3] = 1
        w[0:3, 0:3] = r
        w[0:3, 3] = translation
        A = matrix(w)
        T.append(A.I)
        image_name = im[4]
        matching = [i for i, s in enumerate(jpg_list) if image_name in s]
        ii = im[0]
        i_map.append([ii, matching[0], 0])
    idm = np.identity(4)
    # log file needs an entry for every input image, if image is not part of
    # the SfM bundle it will be assigned to the identity matrix
    for k in range(0, nr_of_images):
        try:
            # find the id of view nr. k
            view_id = [i for i, item in enumerate(i_map) if k == item[1]][0]
            i_mapF.append(np.array([k, k, 0.0], dtype='int'))
            TF.append(T[view_id])
        except:
            i_mapF.append(np.array([k, -1, 0.0], dtype='int'))
            TF.append(idm)
    write_SfM_log(TF, i_mapF, logfile_out)


# USAGE: convert_MVE_to_log([MVE SfM file] [output log-filename] \
#                               [folder containing the mve views])
# EXAMPLE: convert_MVE_to_log('synth_0.out','mve.log','views/')
def convert_MVE_to_log(filename, logfile_out, views_folder):
    bundlefile = filename
    lines = open(bundlefile).read().split('\n')
    views_list = glob.glob(views_folder + '/*/')
    views_list.sort()
    nr_of_views = len(views_list)
    nr_of_images = int(lines[1].split(' ')[0])

    T = []
    i_map = []
    TF = []
    i_mapF = []

    image_list = []
    ii = 0
    for x in range(0, nr_of_views):
        meta_filename = views_list[x] + 'meta.ini'
        lines_m = open(meta_filename).read().split('\n')
        matching_i = [i for i, s in enumerate(lines_m) if 'name = ' in s]
        orig_img = lines_m[matching_i[0]].split('= ')[1]
        image_list.append(orig_img)
    image_list.sort()
    for x in range(0, nr_of_views):
        meta_filename = views_list[x] + 'meta.ini'
        lines_m = open(meta_filename).read().split('\n')
        matching_r = [i for i, s in enumerate(lines_m) if 'rotation' in s]
        if (len(matching_r)):
            r_line = lines_m[matching_r[0]].split('= ')[1]
            r = np.array(r_line.split(' '), dtype='double').reshape(3, 3)
            matching_t = [
                i for i, s in enumerate(lines_m) if 'translation' in s
            ]
            t_line = lines_m[matching_t[0]].split('= ')[1]
            translation = np.array(t_line.split(' ')[0:3], dtype='double')
            w = np.zeros((4, 4))
            w[3, 3] = 1
            w[0:3, 0:3] = r
            w[0:3, 3] = translation
            A = matrix(w)
            T.append(A.I)
            matching_i = [i for i, s in enumerate(lines_m) if 'name = ' in s]
            orig_img = lines_m[matching_i[0]].split('= ')[1]
            matching_io = [i for i, s in enumerate(image_list) if orig_img in s]
            i_map.append([x, matching_io[0], 0])
    idm = np.identity(4)
    # log file needs an entry for every input image, if image is not part of
    # the SfM bundle it will be assigned to the identity matrix
    for k in range(0, nr_of_images):
        try:
            # find the bundler id of view nr. k
            view_id = [i for i, item in enumerate(i_map) if k == item[1]][0]
            i_mapF.append(np.array([k, k, 0.0], dtype='int'))
            TF.append(T[view_id])
        except:
            i_mapF.append(np.array([k, -1, 0.0], dtype='int'))
            TF.append(idm)
    write_SfM_log(TF, i_mapF, logfile_out)


# Usage: convert_VSFM_to_log([VSfM *.nvm SfM file] [output log-filename] \
#							 [folder of the input images] [image format])
# Example: convert_VSFM_to_log('result.nvm', 'vsfm.log', 'images/', 'jpg')
def convert_VSFM_to_log(filename, logfile_out, input_images, formatp):
    bundlefile = filename
    lines = open(bundlefile).read().split('\n')
    nr_of_views = int(lines[2])
    jpg_list = glob.glob(input_images + '/*.' + formatp)
    jpg_list.sort()
    nr_of_images = len(jpg_list)

    T = []
    i_map = []
    TF = []
    i_mapF = []
    ii = 0

    for x in range(3, nr_of_views + 3):
        line_split = lines[x].split(' ')
        qvec = np.array([float(line_split[1]), float(line_split[2]), \
             float(line_split[3]), float(line_split[4])])
        r = quat2rotmat(qvec)
        t1x = np.array([float(line_split[5]), float(line_split[6]), \
            float(line_split[7])])
        r = r * np.array([[1], [-1], [-1]])  # no idea why this is necessary
        translation = np.dot(-r, t1x)
        w = np.zeros((4, 4))
        w[3, 3] = 1
        w[0:3, 0:3] = r
        w[0:3, 3] = translation
        A = matrix(w)
        T.append(A.I)
        image_name = line_split[0].split('\t')[0]
        matching = [i for i, s in enumerate(jpg_list) if image_name in s]
        ii = x - 3
        i_map.append([ii, matching[0], 0])
    idm = np.identity(4)

    # log file needs an entry for every input image, if image is not part of
    # the SfM bundle it will be assigned to the identity matrix
    for k in range(0, nr_of_images):
        try:
            # find the bundler id of view nr. k
            view_id = [i for i, item in enumerate(i_map) if k == item[1]][0]
            i_mapF.append(np.array([k, view_id, 0.0], dtype='int'))
            TF.append(T[view_id])
        except:
            i_mapF.append(np.array([k, -1, 0.0], dtype='int'))
            TF.append(idm)
    write_SfM_log(TF, i_mapF, logfile_out)


if __name__ == '__main__':

    filename = sys.argv[1]
    logfile_out = sys.argv[2]
    input_images = sys.argv[3]
    method = sys.argv[4]
    formatp = sys.argv[5]

    if method == 'COLMAP':
        convert_COLMAP_to_log(filename, logfile_out, input_images, formatp)
    if method == 'MVE':
        convert_MVE_to_log(filename, logfile_out, input_images)
    if method == 'VSFM':
        convert_VSFM_to_log(filename, logfile_out, input_images, formatp)
