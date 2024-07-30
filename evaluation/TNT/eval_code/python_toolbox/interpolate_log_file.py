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
# This python script is for interpolating camera poses

import sys
import math
import numpy as np


class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
         "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def write_trajectory(traj, filename):
    with open(filename, 'w') as f:
        for x in traj:
            p = x.pose.tolist()
            f.write(' '.join(map(str, x.metadata)) + '\n')
            f.write('\n'.join(
                ' '.join(map('{0:.12f}'.format, p[i])) for i in range(4)))
            f.write('\n')


def read_mapping(filename):
    mapping = []
    with open(filename, 'r') as f:
        n_sampled_frames = int(f.readline())
        n_total_frames = int(f.readline())
        mapping = np.zeros(shape=(n_sampled_frames, 2))
        metastr = f.readline()
        for iter in range(n_sampled_frames):
            metadata = list(map(int, metastr.split()))
            mapping[iter, :] = metadata
            metastr = f.readline()
    return [n_sampled_frames, n_total_frames, mapping]


def transform_matrix_4d_to_vector_6d(pose):
    pose_vec = list(range(6))
    R = pose[0:3, 0:3]
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = R.flat
    sy = math.sqrt(r00 * r00 + r10 * r10)
    if ~(sy < 1e-6):
        pose_vec[0] = math.atan2(r21, r22)
        pose_vec[1] = math.atan2(-r20, sy)
        pose_vec[2] = math.atan2(r10, r00)
    else:
        pose_vec[0] = math.atan2(-r12, r11)
        pose_vec[1] = math.atan2(-r20, sy)
        pose_vec[2] = 0
    pose_vec[3:] = pose[0:3, 3]
    return pose_vec


def euler_2_rotation_matrix(x, y, z):
    cosx = math.cos(x)
    sinx = math.sin(x)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    cosy = math.cos(y)
    siny = math.sin(y)
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    cosz = math.cos(z)
    sinz = math.sin(z)
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    return Rz.dot(Ry.dot(Rx))


def transform_vector_6d_to_matrix_4d(pose_vec):
    pose = np.identity(4)
    pose[0:3, 3] = pose_vec[3:]
    pose[0:3, 0:3] = euler_2_rotation_matrix(pose_vec[0], pose_vec[1],
                                             pose_vec[2])
    return pose


if __name__ == "__main__":

    print('')
    print('==================================================================')
    print('Python script for interpolating camera poses')
    print('==================================================================')
    print('Algorithm : ')
    print('   1) Transform n-SE(3) camera pose matrices to nx6 matrix,')
    print('      where each row contains euler angles and translation')
    print('   2) Independently interpolate each column of nx6 matrix')
    print('      using 1D cubic interpolation')
    print('==================================================================')

    if len(sys.argv) != 4:
        print('Usage : python %s [input_log] [mapping_txt_file] [output_log]' %
              sys.argv[0])
        print(
            'Example : python %s ../test_data/test.log ../test_data/mapping.txt ../test_data/test_interpolated.log'
            % sys.argv[0])
        print('')
        print('Convention of [input_log]')
        print('[frame ID] [frame ID] 0')
        print('[R t]')
        print('[0 1]')
        print(': (repeats)')
        print('')
        print('Convention of [mapping_txt_file]')
        print('[number of input camera poses]')
        print('[number of desired number of interpolated poses]')
        print('[Image ID] [video frame ID]')
        print(': (repeats)')
        sys.exit()

    # read files
    trajectory = read_trajectory(sys.argv[1])
    n_sampled_frames, n_total_frames, mapping = read_mapping(sys.argv[2])
    print('%d camera poses are loaded' % n_sampled_frames)
    print('Input poses are interpolated to %d poses' % n_total_frames)

    # make nx6 matrix
    n_trajectory = len(trajectory)
    pose_matrix = np.zeros(shape=(n_trajectory, 6))
    for iter in range(n_trajectory):
        pose_vector = transform_matrix_4d_to_vector_6d(trajectory[iter].pose)
        pose_matrix[iter, :] = pose_vector

    # interpolation
    pose_frame_desired = np.linspace(1, n_total_frames, n_total_frames)
    pose_matrix_interpolation = np.zeros(shape=(n_total_frames, 6))
    for iter in range(6):
        pose_element_slice = pose_matrix[:, iter]
        pose_frame_id = mapping[:, 1]
        pose_matrix_interpolation[:, iter] = np.interp(pose_frame_desired,
                                                       pose_frame_id,
                                                       pose_element_slice)

    # transform interpolated vector to SE(3) and output result
    traj_interpolated = []
    for iter in range(n_total_frames):
        pose_vector = pose_matrix_interpolation[iter, :]
        pose = transform_vector_6d_to_matrix_4d(pose_vector)
        metadata = [iter, iter, n_total_frames]
        traj_interpolated.append(CameraPose(metadata, pose))
    write_trajectory(traj_interpolated, sys.argv[3])
