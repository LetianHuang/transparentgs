
import numpy as np
from glob import glob
import os.path as osp
import cv2
import torch
import os

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def EmPixelTroch(img_path, alpha_path, index, model_dir, fscale=0.25):
    thres = 0.50
    # load row image
    if not osp.exists(img_path):
        assert 'The path of image does not exist!!'
    else:
        all_img_list = sorted(glob_imgs(img_path))

    len_image = len(all_img_list)
    half = int(len_image/2)
    if len_image <=0:
        assert 'there is no image !'
    if len_image % 2!=0:
        assert 'there is a fault in image number!'

    power = torch.flipud(2 ** torch.from_numpy(np.arange(half)))
    all_img_np = []

    for i in range(len(all_img_list)):
        all_img_np.append(cv2.cvtColor(
            cv2.resize(cv2.imread(all_img_list[i]), None, fx=fscale, fy=fscale, interpolation=cv2.INTER_CUBIC),

                                       cv2.COLOR_BGR2GRAY))
    all_imgs = torch.from_numpy(np.stack(np.array(all_img_np) / 255.0, axis=0))
    H, W = all_imgs.shape[1], all_imgs.shape[2]
    if not osp.exists(alpha_path):
        assert 'there is np alpha matt image!'
    else:
        alpha_matte = cv2.cvtColor(cv2.imread(alpha_path), cv2.COLOR_BGR2GRAY) > 127
        alpha_matte = torch.from_numpy(alpha_matte)
    if alpha_matte.shape[0] != H or alpha_matte.shape[1] != W:
        assert 'Alpha matte is no consistent size with image'

    corresp = torch.zeros((H, W, 2)).int() # row and col index
    m = pow(2, half)
    z = torch.zeros((m, 1)).int()
    for i in range(0, 1 << half):
        z[i] = i ^ (i >> 1) #2048 1

    alpha_matte = alpha_matte.bool()
    corresp[~alpha_matte.bool()] = -1
    temp = corresp[alpha_matte.bool()]
    v = all_imgs.permute(1,2,0)[alpha_matte.bool()]
    v[v > thres] = 1
    v[v < thres] = 0
    v = v * (power[None, :].repeat(v.shape[0], 2))
    v1 = v[:, :half].sum(dim=1).int()
    v2 = v[:, half:].sum(dim=1).int()
    z = z.permute(1,0).repeat(v.shape[0],1)
    #
    index = torch.where(z == v1.unsqueeze(1).repeat(1, m))
    temp[:, :1] = (index[1]+1).unsqueeze(1)
    #
    index = torch.where(z == v2.unsqueeze(1).repeat(1, m))
    temp[:, 1:] = (index[1] + 1).unsqueeze(1)

    temp[torch.where((torch.max(v,dim=1).values-torch.min(v,dim=1).values) == 0)] = -2
    corresp[alpha_matte.bool()] = temp
    #compute light map
    # no_light_map = torch.zeros_like(alpha_matte)
    light_temp = alpha_matte[alpha_matte.bool()]
    light_temp[torch.where((torch.max(v,dim=1).values-torch.min(v,dim=1).values) == 0)]=0
    #
    light_map = torch.zeros((H, W)).bool()
    light_map[alpha_matte.bool()] = light_temp
    light_map = light_map[:,:,None].repeat(1,1,3)

    dir = img_path.split('/')[-1]
    save_nolight_path = img_path.split('/' + dir)[0]
    cv2.imwrite(osp.join(save_nolight_path, 'corr') + '/no_light' + dir + '.png',
                light_map.detach().cpu().numpy() * 255)

    return corresp.numpy()
