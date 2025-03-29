# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # Load test identity image (Assumes single identity image)
    testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)

    # Load multiple expression images from directory
    exp_images_list = sorted([os.path.join(args.exp_path, f) for f in os.listdir(args.exp_path) if f.endswith(('.jpg', '.png'))])

    # Run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)

    # Identity reference
    i = 0
    name = testdata[i]['imagename']
    images = testdata[i]['image'].to(device)[None, ...]
    with torch.no_grad():
        id_codedict = deca.encode(images)

    # Set identity's pose and expression to neutral
    id_codedict['pose'][:, 3:] = 0  # Reset expression-related pose
    id_codedict['pose'][:, :3] = 0  # Zero rotation (neck)
    id_codedict['exp'] = torch.zeros_like(id_codedict['exp'])

    # Loop over each expression image
    for exp_image_path in exp_images_list:
        exp_name = os.path.basename(exp_image_path).split('.')[0]  # Extract filename without extension
        expdata = datasets.TestData(exp_image_path, iscrop=args.iscrop, face_detector=args.detector)

        # Process expression image
        exp_images = expdata[i]['image'].to(device)[None, ...]
        with torch.no_grad():
            exp_codedict = deca.encode(exp_images)

        # Transfer expression
        id_codedict['pose'][:, 3:] = exp_codedict['pose'][:, 3:]
        id_codedict['exp'] = exp_codedict['exp']

        transfer_opdict, _ = deca.decode(id_codedict)

        # Save only the .obj file into the output folder
        obj_path = os.path.join(savefolder, f'{exp_name}.obj')
        deca.save_obj_no_tex(obj_path, transfer_opdict)

    print(f'-- All .obj files saved in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--image_path', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp/7.jpg', type=str, 
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details' )
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())