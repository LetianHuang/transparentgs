# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image
import cv2

from . import util
from . import renderutils as ru
import pyexr

################
#######################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    )
                                    # indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.mtx = None      
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=False) # True if required to be optimized
        self.register_parameter('env_base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx
    
    def getmax(self):
        print(self.base.max())
        print(self.base.mean())
        print(self.base.min())

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def regularizer_dif(self):
        white = (self.diffuse[..., 0:1] + self.diffuse[..., 1:2] + self.diffuse[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.diffuse - white))

    def sample(self, dir): #dir:[h, w, 3]
        h, w, _ = dir.shape

        roughness = torch.zeros(h, w, 1).unsqueeze(0).contiguous()
        roughness = roughness.cuda()  
        miplevel = self.get_mip(roughness)
        spec = dr.texture(self.specular[0][None, ...], 
                          dir[None, ...].contiguous(),
                          mip=list(m[None, ...] for m in self.specular[1:]), 
                          mip_level_bias=miplevel[..., 0],
                          filter_mode='linear-mipmap-linear', boundary_mode='cube')

        return spec

    def shade(self, pos, normal, view_pos):
        wo = util.safe_normalize(view_pos - pos)
        reflvec = util.safe_normalize(util.reflect(wo, normal))
        # print(f"---------------------- reflvec.shape: {reflvec.shape} --------------------")

        roughness = torch.zeros(*reflvec.shape[:-1], 1).contiguous().cuda()

        miplevel = self.get_mip(roughness)
        spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(),
                          mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0],
                          filter_mode='linear-mipmap-linear', boundary_mode='cube')

        return spec

def srgb_to_linear(tensor):
    mask = tensor <= 0.04045
    tensor_rgb = torch.where(
        mask,
        tensor / 12.92,
        torch.pow((tensor + 0.055) / 1.055, 2.4)
    )
    return tensor_rgb


######################################################################################
# Load and store
######################################################################################
ENVMAP_SIZE=512 # 512
# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() != ".exr":

        latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    else:
        img = pyexr.read(fn)
        H, W = img.shape[:2]
        # img = np.concatenate((img[..., 3*W//4:, :], img[..., :3*W//4, :]), -2)
        latlong_img = torch.tensor(img, dtype=torch.float32, device='cuda')*scale
        #print("light intensity max: ", img.max())
    cubemap = util.latlong_to_cubemap(latlong_img[...,:3].contiguous(), [ENVMAP_SIZE, ENVMAP_SIZE]) 
    #print("after transform",cubemap.shape)
    l = EnvironmentLight(cubemap)
    l.build_mips()
    return l

def load_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr" or os.path.splitext(fn)[1].lower() == ".exr" or os.path.splitext(fn)[1].lower() == ".png":
        return _load_env_hdr(fn, scale)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def _load_env_hdr_win2(fn, scale=1.0, depth=False):
    #print("util.load_hdr_win")
    latlong_img = torch.tensor(util.load_hdr_win2(fn), dtype=torch.float32, device='cuda')*scale
    #print("util.load_hdr_win done")

    if depth:
        cubemap = util.latlong_to_cubemap(latlong_img[...,3:4].expand(-1, -1, 3).contiguous(), [512, 512])
    else:
        cubemap = util.latlong_to_cubemap(latlong_img[...,0:3].contiguous(), [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

def load_env_win2(fn, scale=1.0, depth=False):
    if os.path.splitext(fn)[1].lower() == ".hdr" or os.path.splitext(fn)[1].lower() == ".exr" or os.path.splitext(fn)[1].lower() == ".png" or os.path.splitext(fn)[1].lower() == ".jpg":
        return _load_env_hdr_win2(fn, scale, depth)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]


def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [ENVMAP_SIZE, ENVMAP_SIZE*2])
    # print("envlight max: ",color.max())

    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    ENVMAP_SIZE=base_res
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base)

def extract_env_map(light, resolution=[ENVMAP_SIZE, ENVMAP_SIZE*2]):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    color = util.cubemap_to_latlong(light.base, resolution)
    return color**(1/2.2)

def extract_env_map_c2w(light, c2w, resolution=[ENVMAP_SIZE, ENVMAP_SIZE*2]):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    color = util.cubemap_to_latlong_c2w(light.base, resolution,c2w)
    return color
