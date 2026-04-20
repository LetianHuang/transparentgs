
import torch
import os.path as osp
import math
from glob import glob
import os
import cv2
import numpy as np
from .EMPixelCorres import glob_imgs
def EMCompAlphaMatte(bkcols, obj_folder, epsilon, index, save_alpha_path, model_dir, fscale=0.25):
    save_path = osp.join(model_dir, 'mask/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # obj_img =  sorted(glob(osp.join(obj_folder, "*.jpg")))
    obj_img = sorted(glob_imgs(obj_folder))
    img_cnt = len(bkcols)
    for i in range(img_cnt):
        Color = torch.from_numpy(
            cv2.cvtColor(
                cv2.GaussianBlur(
                    cv2.resize(cv2.imread(obj_img[i]), None, fx=fscale, fy=fscale, interpolation=cv2.INTER_CUBIC)
                    , (1, 1), 0, 0)
                , cv2.COLOR_BGR2GRAY)
        ).float()
        bkcol = bkcols[i]
        [m, n] = bkcol.shape
        alpha = torch.zeros((m, n))

        delta = (Color - bkcol)[:, :, None]
        mask = torch.sqrt(torch.sum(delta * delta, dim=2).float()) > epsilon
        mask = mask[:, :, None].repeat((1, 1, 3))
        alpha[mask[..., 0]] = 1
        if i==0:

            A = alpha
        else:
            A = A+alpha

    A_ = np.sign(A)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(A_.numpy(), cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(save_alpha_path, closing * 255)




