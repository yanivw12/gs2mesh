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
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

import sys
import os
import argparse
import zipfile
import hashlib
import requests
if (sys.version_info > (3, 0)):
    pversion = 3
    from urllib.request import Request, urlopen
else:
    pversion = 2
    from urllib2 import Request, urlopen

id_download_dict = {
    'Auditorium.mp4': '0B-ePgl6HF260SmdGUzJSX0ZfZXc',
    'Auditorium.zip': '0B-ePgl6HF260N1VHWFBTSWQ2MDg',
    'Ballroom.mp4': '0B-ePgl6HF260MzlmY2Jwa0dqME0',
    'Ballroom.zip': '0B-ePgl6HF260aS1hQXJHeHFxNVE',
    'Barn.mp4': '0B-ePgl6HF260ZlBZcHFrTHFLdGM',
    'Barn.zip': '0B-ePgl6HF260NzQySklGdXZyQzA',
    'Church.mp4': '0B-ePgl6HF260dnlGMkFkNlpibG8',
    'Church.zip': '0B-ePgl6HF260SmhXM0czaHJ3SU0',
    'Caterpillar.mp4': '0B-ePgl6HF260Z00xVWgyN2c3WEU',
    'Caterpillar.zip': '0B-ePgl6HF260b2JNbnZYYjczU2s',
    'Courthouse.mp4': '0B-ePgl6HF260TEpnajBqRFJ1enM',
    'Courthouse.zip': '0B-ePgl6HF260bHRNZTJnU1pWMVE',
    'Courtroom.mp4': '0B-ePgl6HF260b0JZeUJlUThSWjQ',
    'Courtroom.zip': '0B-ePgl6HF260UmZIQVgtLXhtZUE',
    'Family.mp4': '0B-ePgl6HF260UmNxYmlQeDhmeFE',
    'Family.tar.gz': '0B-ePgl6HF260SWRlRDZCRXZRZlk',
    'Family.zip': '0B-ePgl6HF260NVRhRmxnTW4tQTQ',
    'Francis.mp4': '0B-ePgl6HF260emtkUElRT0lXQ3M',
    'Francis.tar.gz': '0B-ePgl6HF260MnVqcW1EWDVMcFE',
    'Francis.zip': '0B-ePgl6HF260SHk4ejdaSEhqd28',
    'Horse.mp4': '0B-ePgl6HF260RGFBcF9iTk5XQTA',
    'Horse.tar.gz': '0B-ePgl6HF260eE9EVTdpS3hYamc',
    'Horse.zip': '0B-ePgl6HF260VFdBc0RvQjJuQXc',
    'Ignatius.mp4': '0B-ePgl6HF260T19oUTIyUTRwTE0',
    'Ignatius.zip': '0B-ePgl6HF260d0l0ZDNSZ3ZxREk',
    'Lighthouse.mp4': '0B-ePgl6HF260T184cUdCbFFBVEE',
    'Lighthouse.zip': '0B-ePgl6HF260dHpldktMNV9NRTA',
    'M60.mp4': '0B-ePgl6HF260dG9nTzZHdkRJblE',
    'M60.zip': '0B-ePgl6HF260b2lSTWxwLU1CQ2s',
    'Meetingroom.mp4': '0B-ePgl6HF260V3BFSFFTZFJwSWc',
    'Meetingroom.zip': '0B-ePgl6HF260cV9lNmlZZGp6aUU',
    'Museum.mp4': '0B-ePgl6HF260ZXRwck5rWk4tc2c',
    'Museum.zip': '0B-ePgl6HF260RTY4Ml9Ubm9fUkk',
    'Palace.mp4': '0B-ePgl6HF260X21ac1ZXNmx3VTA',
    'Palace.zip': '0B-ePgl6HF260ZHlJejlXbmFKS3M',
    'Panther.mp4': '0B-ePgl6HF260bVRndWVYRGM4c0U',
    'Panther.zip': '0B-ePgl6HF260SUNBeVhMc1hpb28',
    'Playground.mp4': '0B-ePgl6HF260d0JoR2pWak9RbnM',
    'Playground.zip': '0B-ePgl6HF260TVktaTFyclFhaDg',
    'Temple.mp4': '0B-ePgl6HF260N1VTMGNES0FsaDA',
    'Temple.zip': '0B-ePgl6HF260V2VaSG5GTkl5dmc',
    'Train.mp4': '0B-ePgl6HF260YUttRUI4U0xtS1E',
    'Train.zip': '0B-ePgl6HF260UFNWeXk3MHhCT00',
    'Truck.mp4': '0B-ePgl6HF260aVVZMzhSdVc5Njg',
    'Truck.zip': '0B-ePgl6HF260NEw3OGN4ckF0dnM',
    'advanced_video.chk': '0B-ePgl6HF260RWJIcjRPRnlUS28',
    'advanced_video.zip': '0B-ePgl6HF260OXgzbEJleDVSZ0k',
    'image_sets_md5.chk': '0B-ePgl6HF260dE5zR3FhQmxVbHc',
    'intermediate_video.chk': '0B-ePgl6HF260SVdpbG1peXBOYnM',
    'intermediate_video.zip': '0B-ePgl6HF260UU1zUTd6SzlmczA',
    'advanced_image.zip': '0B-ePgl6HF260UXlhWDBiNVZvdk0',
    'intermediate_image.zip': '0B-ePgl6HF260UU1zUTd6SzlmczA',
    'advanced_image.chk': '0B-ePgl6HF260RWJIcjRPRnlUS28',
    'intermediate_image.chk': '0B-ePgl6HF260SVdpbG1peXBOYnM',
    'md5.txt': '0B-ePgl6HF260QTlJUXpqc3RQOGM',
    'training.zip': '0B-ePgl6HF260dU1pejdkeXdMb00',
    'video_set_md5.chk': '0B-ePgl6HF260M2h5Q3o1bGdpc1U'
}

sep = os.sep
parser = argparse.ArgumentParser(description='Tanks and Temples file' +
                                 'downloader')
parser.add_argument(
    '--modality',
    type=str,
    help='(image|video|both) ' +
    'choose if you want to download video sequences (very big) or pre sampled' +
    ' image sets',
    default='image')
parser.add_argument(
    '--group',
    type=str,
    help='(intermediate|advanced|both|training|all)' +
    ' choose if you want to download intermediate, advanced or training dataset',
    default='both')
parser.add_argument('--pathname',
                    type=str,
                    help='chose destination path name, default = local path',
                    default='')
parser.add_argument('-s',
                    action='store_true',
                    default=False,
                    dest='status',
                    help='show data status')
parser.add_argument('--unpack_off',
                    action='store_false',
                    default=True,
                    dest='unpack',
                    help='do not un-zip the folders after download')
parser.add_argument('--calc_md5_off',
                    action='store_false',
                    default=True,
                    dest='calc_md5',
                    help='do not calculate md5sum after download')


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token2(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def get_confirm_token2(response):
    for key, value in response.headers.items():
        if key.startswith('Set-Cookie'):
            return value.split('=')[1].split(';')[0]
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    total_filesize = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                total_filesize += CHUNK_SIZE
                sys.stdout.write("\r%5.0f MB downloaded" %
                                 (float(total_filesize) / 1000000))
                sys.stdout.flush()
    sys.stdout.write("\rDownload Complete              \n")
    sys.stdout.flush()
    return chunk


def generate_file_md5(filename, blocksize=2**20):
    m = hashlib.md5()
    with open(filename, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()


def download_video(pathname, scene, image_md5_dict, calc_md5):
    scene_out_dir = pathname + 'videos'
    download_file = scene + '.mp4'
    print('\ndownloading video ' + download_file.split('/')[-1])
    idd = id_download_dict[download_file]
    download_file_local = scene_out_dir + sep + scene + '.mp4'
    download_file_from_google_drive(idd, download_file_local)

    if (calc_md5):
        h_md5 = generate_file_md5(download_file_local)
        print('\nmd5 downloaded: ' + h_md5)
        print('md5 original:   ' + video_md5_dict[scene])
        md5_check = h_md5 == video_md5_dict[scene]

        if (not md5_check):
            print('\nWarning: MD5 does not match, delete file and restart' +
                  ' download\n')
    else:
        if (unpack):
            extr_dir = scene_out_dir
            zip_file = scene_out_dir + sep + scene + '.zip'
            if (zipfile.is_zipfile(zip_file)):
                if not os.path.exists(extr_dir):
                    os.makedirs(extr_dir)
                zip = zipfile.ZipFile(zip_file, 'r')
                zip.extractall(extr_dir)


def check_video(pathname, scene, image_md5_dict):
    scene_out_dir = pathname + 'videos'
    ret_str = ' '
    download_file_local = scene_out_dir + sep + scene + '.mp4'
    if os.path.exists(download_file_local):
        h_md5 = generate_file_md5(download_file_local)
        md5_check = h_md5 == video_md5_dict[scene]
        if (md5_check):
            ret_str = 'X'
        else:
            ret_str = '?'
    else:
        ret_str = ' '
    return ret_str


def download_image_sets(pathname, scene, image_md5_dict, calc_md5):
    scene_out_dir = pathname + 'image_sets'
    download_file = scene + '.zip'
    download_file_local = scene_out_dir + sep + scene + '.zip'
    print('\ndownloading image set ' + download_file.split('/')[-1])
    idd = id_download_dict[download_file]
    download_file_from_google_drive(idd, download_file_local)

    if (calc_md5):
        h_md5 = generate_file_md5(download_file_local)
        print('\nmd5 downloaded: ' + h_md5)
        print('md5 original:   ' + image_md5_dict[scene])
        md5_check = h_md5 == image_md5_dict[scene]

        if (md5_check):
            if (unpack):
                extr_dir = scene_out_dir

                zip_file = scene_out_dir + sep + scene + '.zip'
                if (zipfile.is_zipfile(zip_file)):
                    if not os.path.exists(extr_dir):
                        os.makedirs(extr_dir)
                    zip = zipfile.ZipFile(zip_file, 'r')
                    zip.extractall(extr_dir)
        else:
            print('\nWarning: MD5 does not match, delete file and restart' +
                  ' download\n')


def check_image_sets(pathname, scene, image_md5_dict):
    scene_out_dir = pathname + 'image_sets'
    ret_str = ''
    download_file_local = scene_out_dir + sep + scene + '.zip'
    if os.path.exists(download_file_local):
        h_md5 = generate_file_md5(download_file_local)
        md5_check = h_md5 == image_md5_dict[scene]
        if (md5_check):
            ret_str = 'X'
        else:
            ret_str = '?'
    else:
        ret_str = ' '
    return ret_str


def print_status(sequences, modality, pathname, intermediate_list,
                 advanced_list, training_list, image_md5_dict, video_md5_dict):
    #print('intermediate Dataset \t\t\t Video \t\t\t image set')
    print('\n\n data status: \n\n')
    print('[X] - downloaded    [ ] - missing    [?] - being downloaded or ' +
          'corrupted    [n] - not checked')

    if (sequences == 'intermediate' or sequences == 'both' or
            sequences == 'all' or sequences == ''):
        print('\n\n---------------------------------------------------------' +
              '--------')
        line_new = '%12s  %12s  %12s' % (' intermediate Dataset', 'Video',
                                         'image set')
        print(line_new)
        print('-----------------------------------------------------------' +
              '------')
        for scene in intermediate_list:
            line_new = '%12s  %19s  %10s' % (
                scene, check_video(pathname, scene, video_md5_dict) if
                (modality == 'video' or modality == 'both' or modality == '')
                else 'n', check_image_sets(pathname, scene, image_md5_dict) if
                (modality == 'image' or modality == 'both' or
                 modality == '') else 'n')
            print(line_new)

    if (sequences == 'advanced' or sequences == 'both' or sequences == 'all' or
            sequences == ''):
        print('\n\n------------------------------------------------------' +
              '---------')
        line_new = '%12s  %16s  %12s' % (' advanced Dataset', 'Video',
                                         'image set')
        print(line_new)
        print('---------------------------------------------------------------')
        for scene in advanced_list:
            #print(scene + '\t\t\t X \t\t\t X')
            line_new = '%12s  %19s  %10s' % (
                scene, check_video(pathname, scene, video_md5_dict) if
                (modality == 'video' or modality == 'both' or modality == '')
                else 'n', check_image_sets(pathname, scene, image_md5_dict) if
                (modality == 'image' or modality == 'both' or
                 modality == '') else 'n')
            print(line_new)

    if (sequences == 'training' or sequences == 'all' or sequences == ''):
        print('\n\n------------------------------------------------------' +
              '---------')
        line_new = '%12s  %16s  %12s' % (' training Dataset', 'Video',
                                         'image set')
        print(line_new)
        print('---------------------------------------------------------------')
        for scene in training_list:
            #print(scene + '\t\t\t X \t\t\t X')
            line_new = '%12s  %19s  %10s' % (
                scene, check_video(pathname, scene, video_md5_dict) if
                (modality == 'video' or modality == 'both' or modality == '')
                else 'n', check_image_sets(pathname, scene, image_md5_dict) if
                (modality == 'image' or modality == 'both' or
                 modality == '') else 'n')
            print(line_new)


if __name__ == "__main__":
    intermediate_list = [
        'Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther',
        'Playground', 'Train'
    ]
    advanced_list = [
        'Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple'
    ]
    training_list = [
        'Barn', 'Caterpillar', 'Church', 'Courthouse', 'Ignatius',
        'Meetingroom', 'Truck'
    ]

    args = parser.parse_args()
    sequences = args.group
    calc_md5 = args.calc_md5

    if sequences == 'intermediate':
        scene_list = intermediate_list
    elif sequences == 'advanced':
        scene_list = advanced_list
    elif sequences == 'training':
        scene_list = training_list
    elif sequences == 'both':
        scene_list = intermediate_list + advanced_list
    elif sequences == 'all':
        scene_list = intermediate_list + advanced_list + training_list
    elif sequences == '':
        scene_list = intermediate_list + advanced_list
    else:
        sys.exit('Error! Unknown group parameter, see help [-h]')
    scene_list.sort()

    modality = args.modality
    unpack = args.unpack
    status_print = args.status
    pathname = args.pathname
    if pathname:
        pathname = pathname + sep
    # download md5 checksum file and create md5 dict for image sets zip files:
    image_md5_dict = {}
    scene_out_dir = pathname + 'image_sets'
    fname = scene_out_dir + sep + 'image_sets_md5.chk'
    idd = id_download_dict['image_sets_md5.chk']

    print('\ndownloading md5 sum file for image sets')
    download_file_from_google_drive(idd, fname)

    with open(fname) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    for line in content:
        md5 = line.split(' ')[0]
        scene_name = line.split(' ')[-1][0:-4]
        image_md5_dict.update({scene_name: md5})
    # download md5 checksum file and create md5 dict for videos:
    video_md5_dict = {}
    scene_out_dir = pathname + 'videos'
    fname = scene_out_dir + sep + 'video_set_md5.chk'
    idd = id_download_dict['video_set_md5.chk']

    print('\ndownloading md5 sum file for videos')
    download_file_from_google_drive(idd, fname)

    with open(fname) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    for line in content:
        md5 = line.split(' ')[0]
        scene_name = line.split(' ')[-1][0:-4]
        video_md5_dict.update({scene_name: md5})
    if (len(sys.argv) == 1):
        print_status('both', 'both', pathname, intermediate_list, advanced_list,
                     training_list, image_md5_dict, video_md5_dict)
    elif status_print and (len(sys.argv) == 2):
        print_status('both', 'both', pathname, intermediate_list, advanced_list,
                     training_list, image_md5_dict, video_md5_dict)
    elif status_print:
        print_status(sequences, modality, pathname, intermediate_list,
                     advanced_list, training_list, image_md5_dict,
                     video_md5_dict)
    elif sequences or modality:
        for scene in scene_list:
            if modality == 'video':
                download_video(pathname, scene, video_md5_dict, calc_md5)
            elif modality == 'image':
                download_image_sets(pathname, scene, image_md5_dict, calc_md5)
            elif modality == 'both':
                download_image_sets(pathname, scene, image_md5_dict, calc_md5)
                download_video(pathname, scene, video_md5_dict, calc_md5)
            else:
                sys.exit('Error! Unknown modality parameter, see help [-h]')
