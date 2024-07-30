#!/bin/bash

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
# This script generates a COLMAP reconstruction from a numbe rof input imagess
# Usage: sh get_colmap_reconstruction.sh <COLMAP-exe-directory> <image-set-directory> <project-directory>

colmap_folder=$1/
iname=$2/
outf=$3/

DATABASE=${outf}sample_reconstruction.db

PROJECT_PATH=${outf}
mkdir -p ${PROJECT_PATH}
mkdir -p ${PROJECT_PATH}/images

cp -n ${iname}*.jpg ${PROJECT_PATH}/images

${colmap_folder}/colmap feature_extractor \
    --database_path ${DATABASE} \
    --image_path ${PROJECT_PATH}/images \
	--ImageReader.camera_model RADIAL \
	--ImageReader.single_camera 1 \
	--SiftExtraction.use_gpu 1
	
${colmap_folder}/colmap exhaustive_matcher \
    --database_path ${DATABASE} \
    --SiftMatching.use_gpu 1 
    
mkdir ${PROJECT_PATH}/sparse
${colmap_folder}/colmap mapper \
    --database_path ${DATABASE} \
    --image_path ${PROJECT_PATH}/images \
    --output_path ${PROJECT_PATH}/sparse

mkdir ${PROJECT_PATH}/dense

${colmap_folder}/colmap image_undistorter \
    --image_path ${PROJECT_PATH}/images \
    --input_path ${PROJECT_PATH}/sparse/0/ \
    --output_path ${PROJECT_PATH}/dense \
    --output_type COLMAP --max_image_size 1500

${colmap_folder}/colmap patch_match_stereo \
    --workspace_path $PROJECT_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

${colmap_folder}/colmap stereo_fusion \
    --workspace_path $PROJECT_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $PROJECT_PATH/dense/fused.ply
