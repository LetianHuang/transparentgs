# -*- coding: utf-8 -*-
""" 
@Time    : 2022/5/1 17:04
@Author  : HCF
@FileName: main.py
@SoftWare: PyCharm
"""

import os.path as osp
import sys
from EM.EMComAlphaMatte import EMCompAlphaMatte
from EM.EMPixelCorres import EmPixelTroch, glob_imgs
import cv2
from scipy.io import loadmat
import numpy as np
from glob import glob
import scipy.io as scio
import math
import torch
import os
import matplotlib
import re
import torchvision

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def compute_mask(args, recompute=False, fscale=0.5):
    model_dir = osp.join(args.path, args.mode)
    ref_a_path = osp.join(model_dir, 'ref')
    save_alpha_path = osp.join(model_dir, 'mask')
    if (not osp.exists(ref_a_path)) or (not osp.exists(ref_a_path)):
        assert 'there does not exist ref and obj to calculate mask fold'
    ref_img = sorted(glob_imgs(ref_a_path))
    bkcols = []
    for i in range(len(ref_img)):
        bkcol = torch.from_numpy(
            cv2.cvtColor(cv2.GaussianBlur(
                cv2.resize(cv2.imread(ref_img[i]), None, fx=fscale, fy=fscale, interpolation=cv2.INTER_CUBIC),
                (1, 1), 0, 0), cv2.COLOR_BGR2GRAY)).float()
        bkcols.append(bkcol)

    if not osp.exists(save_alpha_path):
        os.makedirs(save_alpha_path)
    epsilon = 0.33* math.sqrt(3) * 255#0.25
    for i in range(args.views):
        obj_a_path = osp.join(model_dir, str(i + 1))
        alphamattePath = save_alpha_path + '/closing' + str(i + 1) + '.jpg'

        if not osp.exists(alphamattePath) or recompute:
            EMCompAlphaMatte(bkcols, obj_a_path, epsilon, i, alphamattePath, model_dir, fscale)
        print('compute over' + str(i))
    print('====================compute mask over!!!!!===========================')
    print('====================compute mask over!!!!!===========================')
    print('====================compute mask over!!!!!===========================')
    return save_alpha_path
