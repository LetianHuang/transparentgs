import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import math
import sys

from shader.NVDIFFREC import util
from shader.NVDIFFREC import renderutils as ru
from shader.NVDIFFREC.light import load_env, load_env_win2, EnvironmentLight

from PIL import Image

import Imath
import OpenEXR
import json

def Luminance(color):
    #color : [S, 3]
    #print("color.shape:", color.shape)

    # return : [S, 1]
    return torch.sum(torch.tensor([0.299, 0.587, 0.114]).cuda() * color, dim=1, keepdim=True)

def make_light(rgb):
    rgb_luminance = Luminance(rgb)
    luminance_mask = rgb_luminance > 0.999
    mul = torch.ones_like(rgb_luminance) + luminance_mask.float() * 1
    return mul * rgb, rgb_luminance

def refract2(l, normal, eta1, eta2):
    # l ... x 3 
    # normal ... x 3
    # eta1 float
    # eta2 float
    l = -l
    cos_theta = torch.sum(l * (-normal), dim=-1).unsqueeze(-1)  # [10, 1, 192, 256]
    i_p = l + normal * cos_theta
    t_p = eta1 / eta2 * i_p

    t_p_norm = torch.sum(t_p * t_p, dim=-1)
    totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(-1)

    t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(-1).expand_as(normal) * (-normal)
    t = t_i + t_p
    t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=-1), min=1e-10)).unsqueeze(-1)

    cos_theta_t = torch.sum(t * (-normal), dim=-1).unsqueeze(-1)

    e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
            torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
    e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
            torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

    attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach()

    return t, attenuate, totalReflectMask

def save_exr(file_path, tensor_image):
    if tensor_image.shape[-1] != 3:
        raise ValueError("Expected tensor shape [H, W, 3] for an RGB image.")
    
    image_np = tensor_image.detach().cpu().numpy()
    
    height, width, _ = image_np.shape

    header = OpenEXR.Header(width, height)
    header['channels'] = dict(R=Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                              G=Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                              B=Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)))
    
    R = image_np[:, :, 0].astype(np.float32).tobytes()
    G = image_np[:, :, 1].astype(np.float32).tobytes()
    B = image_np[:, :, 2].astype(np.float32).tobytes()
    
    exr_file = OpenEXR.OutputFile(file_path, header)
    exr_file.writePixels({'R': R, 'G': G, 'B': B})
    exr_file.close()

def read_exr_3channel(file_path, dtype=np.float16):
    exr_file = OpenEXR.InputFile(file_path)

    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    channels = ['R', 'G', 'B']

    img = np.zeros((height, width, len(channels)), dtype=dtype)
    for i, channel in enumerate(channels):
        img[:, :, i] = np.frombuffer(exr_file.channel(channel), dtype=dtype).reshape(height, width)

    img_tensor = torch.from_numpy(img)
    return img_tensor

import cv2
from skimage.transform import resize
def read_exr_3channel_upsample(file_path, dtype=np.float16):
    exr_file = OpenEXR.InputFile(file_path)

    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    channels = ['R', 'G', 'B'] 

    img = np.zeros((height, width, len(channels)), dtype=dtype)
    for i, channel in enumerate(channels):
        img[:, :, i] = np.frombuffer(exr_file.channel(channel), dtype=dtype).reshape(height, width)

    final_resolution = np.array([height*2, width*2])
    img = resize(img, final_resolution, order=1)

    img_tensor = torch.from_numpy(img)
    return img_tensor


def read_exr_4channel(file_path, dtype=np.float16):
    exr_file = OpenEXR.InputFile(file_path)
    
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    channels = ['R', 'G', 'B', 'A']  

    img = np.zeros((height, width, len(channels)), dtype=dtype)
    for i, channel in enumerate(channels):
        img[:, :, i] = np.frombuffer(exr_file.channel(channel), dtype=dtype).reshape(height, width)

    img_tensor = torch.from_numpy(img)
    return img_tensor


class EnvironmentLightProbes(torch.nn.Module):
    (r"""---------------------------------------------------------------------------------------------------------------------------""")
    def __init__(self, rgbmap : EnvironmentLight, depthmap : EnvironmentLight):
        super(EnvironmentLightProbes, self).__init__()
        self.rgbmap = rgbmap
        self.depthmap = depthmap

    def xfm(self, mtx):
        self.rgbmap.xfm(mtx)
        self.depthmap.xfm(mtx)
    
    def getmax(self):
        self.rgbmap.getmax()
        self.depthmap.getmax()

    def clone(self):
        return EnvironmentLightProbes(self.rgbmap.clone(), self.depthmap.clone())

    def clamp_(self, min=None, max=None):
        self.rgbmap.clamp_(min, max)
        # self.depthmap.clamp_(min, max)

    def get_mip(self, roughness):
        return self.rgbmap.get_mip(roughness)
    
    def build_mips(self, cutoff=0.99):
        self.rgbmap.build_mips(cutoff)
        self.depthmap.build_mips(cutoff)

    def regularizer(self):
        return self.rgbmap.regularizer()

    def regularizer_dif(self):
        return self.rgbmap.regularizer_dif()
    
    def light_probes_sample(self, rays, miplevel, num_iters=2):
        import torchvision
        rays_o, rays_d = rays[..., :3], rays[..., 3:]
        for iter in range(num_iters):
            depth = dr.texture(self.depthmap.base[None, ...], rays_d.contiguous(), filter_mode='linear', boundary_mode='cube')
            # torchvision.utils.save_image(depth[0].permute(2, 0, 1) / 5, f"./{iter}_depth.png")
            rays_d = util.safe_normalize(rays_o + depth * rays_d)
            dis = torch.sqrt(torch.sum(torch.square(rays_o + depth * rays_d), -1, keepdim=True)).expand_as(depth)
            # torchvision.utils.save_image(dis[0].permute(2, 0, 1) / 5, f"./{iter}_dis.png")
        print(f"self.rgbmap.base.shape: {self.rgbmap.base.shape}")
        return dr.texture(self.rgbmap.base[None, ...], rays_d.contiguous(), filter_mode='linear', boundary_mode='cube')
    (r"""---------------------------------------------------------------------------------------------------------------------------""")

ENVMAP_SIZE=512

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-14) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)

def refract_vector(x, n, ior):
    dot_product = torch.einsum('ij,ij->i', x, n)
    indices_to_flip = dot_product >= 0
    n[indices_to_flip] = -n[indices_to_flip]

    cos_theta_i = -torch.sum(x * n, dim=1)
    sin_theta_i = torch.sqrt(1 - cos_theta_i ** 2)
    sin2_theta_i = sin_theta_i ** 2

    sin2_theta_t = sin2_theta_i * (ior ** 2)
    total_internal_reflection = sin2_theta_t > 1.0

    cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_theta_t, min=0.0))
    refracted = (ior * x + (ior * cos_theta_i - cos_theta_t)[:, None] * n)
    refracted = safe_normalize(refracted)

    return refracted, total_internal_reflection

def distort_direction(dir):
    angle_x = math.radians(0.0)  
    angle_y = math.radians(0.0)  
    angle_z = math.radians(180.0) 

    rotation_x = torch.tensor([
        [1., 0., 0.],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ])

    rotation_y = torch.tensor([
        [math.cos(angle_y), 0., math.sin(angle_x)],
        [0, 1., 0.],
        [-math.sin(angle_x), 0., math.cos(angle_y)]
    ])

    rotation_z = torch.tensor([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0., 0., 1.]
    ])

    rotation_matrix = rotation_z @ rotation_x
    rotation_matrix = rotation_matrix.cuda()

    result = torch.matmul(dir.view(8, -1, 3), rotation_matrix)
    result = result.unsqueeze(1)

    return result

def compute_relative_position(points, txx):
    grid_min = txx.min(dim=0).values  # [-1, -1, -1]
    grid_max = txx.max(dim=0).values  # [1, 1, 1]
    grid_size = grid_max - grid_min  # [2, 2, 2]

    normalized_points = (points - grid_min) / grid_size

    return normalized_points

def compute_weights(relative_positions):
    x, y, z = relative_positions[:, 0], relative_positions[:, 1], relative_positions[:, 2]

    weights = torch.stack([
        x * y * z,                    # [0, 0, 0]
        x * y * (1 - z),        # [0, 0, 1]
        x * (1 - y) * z,        # [0, 1, 0]
        x * (1 - y) * (1 - z),              # [0, 1, 1]
        (1 - x) * y * z,        # [1, 0, 0]
        (1 - x) * y * (1 - z),              # [1, 0, 1]
        (1 - x) * (1 - y) * z,              # [1, 1, 0]
        (1 - x) * (1 - y) * (1 - z)                     # [1, 1, 1]
    ], dim=1)  

    return weights

def trilinear_interpolation(rgb, weights):
    num_features = rgb.shape[2]
    rgb = rgb.permute(1, 0, 2)  # rgb: [S, 8, x]
    return torch.sum(rgb * weights.unsqueeze(-1).expand(-1, -1, num_features), dim = 1)

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

def save_img(path, img):
    array = (img * 255).astype(np.uint8)

    image = Image.fromarray(array)
    image.save(path)

def get_interpolatio_result(cos_depth, weights):
    #value: [8, S, 3]
    #weights: [S, 8]
    interpolated_depth = trilinear_interpolation(cos_depth, weights)

    return interpolated_depth

def trilinear_interpolation2(rgb, weights):
    n_rays, _, _, _ = weights.shape
    weights = weights.reshape(n_rays, 64)

    num_features = rgb.shape[2]
    # rgb: [64, S, x]
    # weights: [S, 64]
    rgb = rgb.permute(1, 0, 2)  # rgb: [S, 8, x]
    return torch.sum(rgb * weights.unsqueeze(-1).expand(-1, -1, num_features), dim = 1)

def get_interpolatio_result2(cos_depth, weights):
    #value: [64, S, 3]
    #weights: [S, 4, 4, 4]
    n_rays, _, _, _ = weights.shape
    interpolated_depth = trilinear_interpolation2(cos_depth, weights.reshape(n_rays, 64))
  
    return interpolated_depth

import time
def compute_trilinear_weights(positions, p0, p1, grid_size=4):
    S = positions.shape[0]
    #[S, grid_size, grid_size, grid_size]
    weights = torch.zeros((S, grid_size, grid_size, grid_size), dtype=torch.float32).cuda()

    positions_grid = (positions.cuda() - p0) / (p1 - p0) * (grid_size-1)

    base_idx = torch.floor(positions_grid).to(torch.int64).cuda()  # [S, 3]
    offset = positions_grid - base_idx  # [S, 3]

    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                neighbor_idx = base_idx + torch.tensor([dx, dy, dz], dtype=torch.int64).cuda()

                weight = (
                    (offset[:, 0] if dx else 1 - offset[:, 0]) *
                    (offset[:, 1] if dy else 1 - offset[:, 1]) *
                    (offset[:, 2] if dz else 1 - offset[:, 2])
                )

                valid_mask = (
                    (neighbor_idx[:, 0] >= 0) & (neighbor_idx[:, 0] < grid_size) &
                    (neighbor_idx[:, 1] >= 0) & (neighbor_idx[:, 1] < grid_size) &
                    (neighbor_idx[:, 2] >= 0) & (neighbor_idx[:, 2] < grid_size)
                )

                weights[valid_mask,
                        neighbor_idx[valid_mask, 0],
                        neighbor_idx[valid_mask, 1],
                        neighbor_idx[valid_mask, 2]] += weight[valid_mask]

    return weights


MODULE_LABEL = ""

def test_sample_64(prob_positions, rays, rgb_probes, depth_probes, num_iters=2):
    #envs: [8, EnvironmentLightProbes]
    #prob_positions: [4, 4, 4, 3]
    #offsets: [3]
    #rays: [S, 6]

    rays_o, rays_d = rays[..., :3], rays[..., 3:]
    n_rays = rays_o.shape[0]

    total_rgb_probes = rgb_probes
    total_depth_probes = depth_probes

    #weights = torch.zeros([4, 4, 4]).cuda()
    #[S, 4, 4, 4]
    global MODULE_LABEL
    if MODULE_LABEL == "":
        try:
            from compute_trilinear_weights import _C
            print(f"use CUDA to accelerate weights computation")
            MODULE_LABEL = "CUDA"
            weights = _C.compute_trilinear_weights(rays_o, prob_positions[0,0,0], prob_positions[3,3,3], 4) # or compute_trilinear_weights
        except:
            print(f"have not used CUDA to accelerate weights computation")
            MODULE_LABEL = "Python"
            weights = compute_trilinear_weights(rays_o, prob_positions[0,0,0], prob_positions[3,3,3], 4)
    elif MODULE_LABEL == "CUDA":
        from compute_trilinear_weights import _C
        weights = _C.compute_trilinear_weights(rays_o, prob_positions[0,0,0], prob_positions[3,3,3], 4) # or compute_trilinear_weights
    else:
        weights = compute_trilinear_weights(rays_o, prob_positions[0,0,0], prob_positions[3,3,3], 4)

    #[S, 4, 4, 4, 3]
    dealt_dir = rays_o.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 4, 4, 4, 1) - prob_positions.unsqueeze(0).repeat(n_rays, 1, 1, 1, 1)
    #[S, 4, 4, 4, 1]
    dealt_length = torch.sqrt(torch.sum(dealt_dir**2, dim=-1)).unsqueeze(-1)
    #[S, 4, 4, 4, 1]
    cos_dealt = torch.sum(safe_normalize(dealt_dir) * rays_d.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 4, 4, 4, 1), dim=-1).unsqueeze(-1)
    #[S, 4, 4, 4, 1]
    dealt_depth = dealt_length * cos_dealt

    #[S, 3] -> [64, 1, S, 3]
    rays_d = rays_d.unsqueeze(0).unsqueeze(0).repeat(64, 1, 1, 1)
    rays_d_origin = rays_d

    for iter in range(num_iters):
        #[64, 1, S, 3]
        rays_d_sample = rays_d
        depth = dr.texture(
            total_depth_probes, #[64, 6, res, res, 3]
            rays_d_sample.contiguous(), #[64, 1, S, 3]
            filter_mode='linear', 
            boundary_mode='cube')
        
        #mask = depth < (1e-4)
        # depth = torch.where(depth < 1e-8, torch.tensor(1000.0).cuda(), depth)
        # depth = torch.where(depth > 1e+3, torch.tensor(1000.0).cuda(), depth)
        # #depth = torch.where(depth < 1e-6, torch.tensor(100.0).cuda(), depth)
        # #depth = torch.where(depth is nan(), torch.tensor(5.0).cuda(), depth)
        # depth = torch.nan_to_num(depth, nan=1000)

        #[64, S, 3]
        depth = depth.squeeze(1)

        #[64, S, 3]
        cos = torch.sum(rays_d * rays_d_origin, dim=-1).squeeze(1).unsqueeze(-1).repeat(1, 1, 3)
        cos_depth = depth * cos
        cos_depth = cos_depth + dealt_depth.reshape(n_rays, 64, 1).permute(1, 0, 2).expand(-1, -1, 3)

        #[64, S, 1]
        interpolated_depth = trilinear_interpolation2(cos_depth, weights)
        #interpolated_depth = interpolated_depth * mask + 10000. * torch.ones_like(interpolated_depth) * ~mask

        #[64, S, 3]
        target_position = rays_o + rays_d_origin.squeeze(1) * interpolated_depth

        #[64, S, 3]
        #rays_d = target_position - prob_positions.unsqueeze(1)
        rays_d = target_position - prob_positions.reshape(64, 3).unsqueeze(1)

        #[64, 1, S, 3]
        rays_d = rays_d.unsqueeze(1)
        rays_d = safe_normalize(rays_d)

    #[64, 1, S, 3]
    rays_d_sample = rays_d
    final_rgb = dr.texture(
        total_rgb_probes, #[64, 6, res, res, 3]
        rays_d_sample.contiguous(), #[64, 1, S, 3]
        filter_mode='linear', 
        boundary_mode='cube')
    #[64, S, 3]    
    final_rgb = final_rgb[...,0:3].squeeze(1)

    final_depth = dr.texture(
        total_depth_probes, #[64, 6, res, res, 3]
        rays_d_sample.contiguous(), #[64, 1, S, 3]
        filter_mode='linear', 
        boundary_mode='cube')

    cos = torch.sum(rays_d * rays_d_origin, dim=-1).squeeze(1).unsqueeze(-1).repeat(1, 1, 3)
    #[64, S, 3]    
    final_depth = final_depth[...,0:3].squeeze(1)
    cos_final_depth = final_depth * cos
    cos_final_depth = cos_final_depth + dealt_depth.reshape(n_rays, 64, 1).permute(1, 0, 2).expand(-1, -1, 3)

    #[S, 3]
    interpolated_rgb = trilinear_interpolation2(final_rgb, weights) 
    interpolated_depth = trilinear_interpolation2(cos_final_depth, weights)  
    return interpolated_rgb, interpolated_depth


def test_sample_8(rgb_probes, depth_probes, prob_positions, rays, num_iters=2):
    #envs: [8, EnvironmentLightProbes]
    #prob_positions: [8, 3]
    #rays:      [S, 6]

    rays_o, rays_d = rays[..., :3], rays[..., 3:]
    n_rays = rays_o.shape[0]

    #[8, S, 3]
    dealt_dir = rays_o.unsqueeze(0).repeat(8, 1, 1) - prob_positions.unsqueeze(1).repeat(1, n_rays, 1)
    
    #[8, S, 1]
    dealt_length = torch.sqrt(torch.sum(dealt_dir**2, dim=-1)).unsqueeze(-1)

    #[8, S, 1]
    cos_dealt = torch.sum(safe_normalize(dealt_dir) * rays_d.unsqueeze(0).repeat(8, 1, 1), dim=-1).unsqueeze(-1)
    
    #[8, S, 1]
    dealt_depth = dealt_length * cos_dealt

    #[S, 3] -> [8, 1, S, 3]
    rays_d_gt = rays_d.unsqueeze(0).unsqueeze(0)
    rays_d = rays_d.unsqueeze(0).unsqueeze(0).repeat(8, 1, 1, 1)
    #rays_d_gt = rays_d.unsqueeze(0).unsqueeze(0)
    rays_d_origin = rays_d

    relative_positions = compute_relative_position(rays_o, prob_positions) 
    #print("relative_positions:", relative_positions)
    #[S, 8]
    weights = compute_weights(relative_positions)  # [S, 8]

    #[S, 3] -> [8, S, 3]
    rays_o = rays_o.unsqueeze(0).repeat(8, 1, 1)

    for iter in range(num_iters):
        #[8, 1, S, 3]
        rays_d_sample = rays_d
        #rays_d_sample = distort_direction(rays_d).type(torch.float32)
        depth = dr.texture(
            depth_probes, #[8, 6, res, res, 3]
            rays_d_sample.contiguous(), #[8, 1, S, 3]
            filter_mode='linear', 
            boundary_mode='cube')

        #[8, S, 3]
        depth = depth.squeeze(1)

        #[8, S, 3]
        cos = torch.sum(rays_d * rays_d_origin, dim=-1).squeeze(1).unsqueeze(-1).repeat(1, 1, 3)

        cos_depth = depth * cos

        cos_depth = cos_depth + dealt_depth.expand(-1, -1, 3)

        interpolated_depth = trilinear_interpolation(cos_depth, weights)

        #[8, S, 3]
        target_position = rays_o + rays_d_origin.squeeze(1) * interpolated_depth

        #[8, S, 3]
        rays_d = target_position - prob_positions.unsqueeze(1)
        #[8, 1, S, 3]
        rays_d = rays_d.unsqueeze(1)
        rays_d = safe_normalize(rays_d)

    #[8, 1, S, 3]
    rays_d_sample = rays_d
    final_rgb = dr.texture(
        rgb_probes, #[8, 6, res, res, 3]
        rays_d_sample.contiguous(), #[8, 1, S, 3]
        filter_mode='linear', 
        boundary_mode='cube')
    #[8, S, 3]    
    final_rgb = final_rgb[...,0:3].squeeze(1)

    final_depth = dr.texture(
        depth_probes, #[8, 6, res, res, 3]
        rays_d_sample.contiguous(), #[8, 1, S, 3]
        filter_mode='linear', 
        boundary_mode='cube')

    cos = torch.sum(rays_d * rays_d_origin, dim=-1).squeeze(1).unsqueeze(-1).repeat(1, 1, 3)
    #[8, S, 3]    
    final_depth = final_depth[...,0:3].squeeze(1)
    cos_final_depth = final_depth * cos
    cos_final_depth = cos_final_depth + dealt_depth.expand(-1, -1, 3)

    #[S, 3]
    interpolated_rgb = trilinear_interpolation(final_rgb, weights)  # [S, 3]
    interpolated_depth = trilinear_interpolation(cos_final_depth, weights)  # [S, 3]
    return interpolated_rgb, interpolated_depth

def load_probes8(path, center, scale):
    txx = torch.tensor([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]).cuda()

    probe_positions = scale.cuda() * txx + center.cuda()

    lis = [3, 0]

    env_list = []
    for x in lis:
        for y in lis:
            for z in lis:
                file_name = "{0}{1}{2}.exr".format(x, y, z)
                depth_name = "{0}{1}{2}_depth.exr".format(x, y, z)
                if sys.platform == "win32":
                    rgb_env = load_env_win2(os.path.join(path, file_name)).cuda()
                    d_env = load_env_win2(os.path.join(path, depth_name)).cuda()
                else:
                    rgb_env = load_env(os.path.join(path, file_name)).cuda()
                    d_env = load_env(os.path.join(path, depth_name)).cuda()
                #rgb_env = load_env(os.path.join(path, file_name)).cuda()
                #d_env = load_env(os.path.join(path, depth_name)).cuda()
                rgb_env.build_mips()
                d_env.build_mips()
                env_list.append(EnvironmentLightProbes(rgb_env, d_env))

    rgb_probes = [
        env_list[0].rgbmap.base,
        env_list[1].rgbmap.base,
        env_list[2].rgbmap.base,
        env_list[3].rgbmap.base,
        env_list[4].rgbmap.base,
        env_list[5].rgbmap.base,
        env_list[6].rgbmap.base,
        env_list[7].rgbmap.base
    ]

    depth_probes = [
        env_list[0].depthmap.base,
        env_list[1].depthmap.base,
        env_list[2].depthmap.base,
        env_list[3].depthmap.base,
        env_list[4].depthmap.base,
        env_list[5].depthmap.base,
        env_list[6].depthmap.base,
        env_list[7].depthmap.base
    ]

    rgb_probes = torch.stack(rgb_probes, dim=0).float().cuda()
    depth_probes = torch.stack(depth_probes, dim=0).float().cuda()

    return {
        "probe_positions": probe_positions,
        "env_list": env_list,
        "rgb_probes": rgb_probes,
        "depth_probes": depth_probes
    }

def probes_sample_8(rays, probes, total_iter=0):
    probe_positions = probes["probe_positions"]
    rgb_probes = probes["rgb_probes"]
    depth_probes = probes["depth_probes"]

    rgb, depth = test_sample_8(rgb_probes, depth_probes, probe_positions, rays, total_iter)
    return rgb, depth

def load_probes64(path, center, scale):
    pmin = center - scale
    pmax = center + scale
    pdelt = (pmax-pmin) / 3.
    new_probe_positions = torch.zeros([4, 4, 4, 3]).cuda()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                new_probe_positions[i, j, k] = pmin.cuda() + pdelt.cuda() * torch.tensor([i, j, k]).cuda()

    env_list = [
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []]]

    for i in range(4):
        for j in range(4):
            for k in range(4):
                file_name = "{0}{1}{2}.exr".format(i, j, k)
                depth_name = "{0}{1}{2}_depth.exr".format(i, j, k)
                if sys.platform == "win32":
                    rgb_env = load_env_win2(os.path.join(path, file_name)).cuda()
                    d_env = load_env_win2(os.path.join(path, depth_name)).cuda()
                else:
                    rgb_env = load_env(os.path.join(path, file_name)).cuda()
                    d_env = load_env(os.path.join(path, depth_name)).cuda()
                #rgb_env = load_env(os.path.join(path, file_name)).cuda()
                #d_env = load_env(os.path.join(path, depth_name)).cuda()
                rgb_env.build_mips()
                d_env.build_mips()
                env_list[i][j].append(EnvironmentLightProbes(rgb_env, d_env))

    #probes_num = torch.tensor([4,4,4]).cuda()

    _, h, w, _ = env_list[0][0][0].rgbmap.base.shape
    total_rgb_probes = torch.zeros([4, 4, 4, 6, h, w, 3])
    total_depth_probes = torch.zeros([4, 4, 4, 6, h, w, 3])
    for i in range(4):
        for j in range(4):
            for k in range(4):
                #print("env_list[i][j][k].rgbmap.base.dtype:", env_list[i][j][k].rgbmap.base.dtype)
                total_rgb_probes[i,j,k] = env_list[i][j][k].rgbmap.base.to(torch.float16)
                total_depth_probes[i,j,k] = env_list[i][j][k].depthmap.base.to(torch.float16)

    _, _, _, _, h, w, _ = total_rgb_probes.shape
    total_rgb_probes = total_rgb_probes.reshape(64, 6, h, w, 3).cuda()
    total_depth_probes = total_depth_probes.reshape(64, 6, h, w, 3).cuda()

    return {
        #"txx": txx,
        "probe_positions": new_probe_positions,
        "rgb_probes": total_rgb_probes,
        "depth_probes": total_depth_probes
    }


def probes_sample_64(rays, probes, total_iter=0):
    #txx = probes["txx"]
    probe_positions = probes["probe_positions"]
    rgb_probes = probes["rgb_probes"]
    depth_probes = probes["depth_probes"]

    rgb, depth = test_sample_64(probe_positions, rays, rgb_probes, depth_probes, total_iter)
    #rgb = rgb.reshape(1, H, W, 3)
    return rgb, depth

def load_probes1(path, center, scale):
    env_list = []

    # file_name = "000.exr"
    # depth_name = "000_depth.exr"
    file_name = "000.exr"
    depth_name = "000_depth.exr"
    
    #print("os.path.join(path, file_name):", os.path.join(path, file_name))
    print("sys.platform:", sys.platform)
    if sys.platform == "win32":
        rgb_env = load_env_win2(os.path.join(path, file_name)).cuda()
        d_env = load_env_win2(os.path.join(path, depth_name)).cuda()
    else:
        rgb_env = load_env(os.path.join(path, file_name)).cuda()
        d_env = load_env(os.path.join(path, depth_name)).cuda()
    rgb_env.build_mips()
    d_env.build_mips()
    env_list.append(EnvironmentLightProbes(rgb_env, d_env))

    rgb_probes = [
        env_list[0].rgbmap.base,
    ]

    depth_probes = [
        env_list[0].depthmap.base,
    ]

    rgb_probes = torch.stack(rgb_probes, dim=0).float().cuda()
    depth_probes = torch.stack(depth_probes, dim=0).float().cuda()

    probe_positions = center - scale
    probe_positions = probe_positions.unsqueeze(0).cuda()
    return {
        "probe_positions": probe_positions,
        "env_list": env_list,
        "rgb_probes": rgb_probes,
        "depth_probes": depth_probes
    }

def probes_sample_1(rays, probes, total_iter=0):
    #txx = probes["txx"]
    probe_positions = probes["probe_positions"]
    rgb_probes = probes["rgb_probes"]
    depth_probes = probes["depth_probes"]

    rgb, depth = test_sample_1(rgb_probes, depth_probes, probe_positions, rays, total_iter)
    #rgb = rgb.reshape(1, H, W, 3)
    return rgb, depth

def test_sample_1(rgb_probes, depth_probes, prob_positions, rays, num_iters=2):
    #envs: [1, EnvironmentLightProbes]
    #prob_positions: [1, 3]
    #rays:      [S, 6]

    rays_o, rays_d = rays[..., :3], rays[..., 3:]
    n_rays = rays_o.shape[0]

    #[1, S, 3]
    dealt_dir = rays_o.unsqueeze(0).repeat(1, 1, 1) - prob_positions.unsqueeze(1).repeat(1, n_rays, 1)
    
    #[1, S, 1]
    dealt_length = torch.sqrt(torch.sum(dealt_dir**2, dim=-1)).unsqueeze(-1)

    #[1, S, 1]
    cos_dealt = torch.sum(safe_normalize(dealt_dir) * rays_d.unsqueeze(0).repeat(1, 1, 1), dim=-1).unsqueeze(-1)
    
    #[1, S, 1]
    dealt_depth = dealt_length * cos_dealt

    #[S, 3] -> [1, 1, S, 3]
    rays_d_gt = rays_d.unsqueeze(0).unsqueeze(0)
    rays_d = rays_d.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
    #rays_d_gt = rays_d.unsqueeze(0).unsqueeze(0)
    rays_d_origin = rays_d

    #[S, 3] -> [1, S, 3]
    rays_o = rays_o.unsqueeze(0).repeat(1, 1, 1)

    for iter in range(num_iters):
        #[1, 1, S, 3]
        rays_d_sample = rays_d
        #rays_d_sample = distort_direction(rays_d).type(torch.float32)
        #print("depth_probes.shape:", depth_probes.shape)
        #print("depth_probes[0, :, 256, 256, :]:", depth_probes[0, :, 256, 256, :])
        depth = dr.texture(
            depth_probes, #[1, 6, res, res, 3]
            rays_d_sample.contiguous(), #[1, 1, S, 3]
            filter_mode='linear', 
            boundary_mode='cube')

        #print("depth:", depth)

        #[1, S, 3]
        depth = depth.squeeze(1)

        #[1, S, 3]
        cos = torch.sum(rays_d * rays_d_origin, dim=-1).squeeze(1).unsqueeze(-1).repeat(1, 1, 3)

        cos_depth = depth * cos

        cos_depth = cos_depth + dealt_depth.expand(-1, -1, 3)

        #interpolated_depth = trilinear_interpolation(cos_depth, weights)

        #[1, S, 3]
        target_position = rays_o + rays_d_origin.squeeze(1) * cos_depth

        #[1, S, 3]
        rays_d = target_position - prob_positions.unsqueeze(1)
        #[1, 1, S, 3]
        rays_d = rays_d.unsqueeze(1)
        rays_d = safe_normalize(rays_d)

    #[1, 1, S, 3]
    rays_d_sample = rays_d
    final_rgb = dr.texture(
        rgb_probes, #[1, 6, res, res, 3]
        rays_d_sample.contiguous(), #[1, 1, S, 3]
        filter_mode='linear', 
        boundary_mode='cube')
    #[1, S, 3]    
    final_rgb = final_rgb[...,0:3].squeeze(1)

    final_depth = dr.texture(
        depth_probes, #[1, 6, res, res, 3]
        rays_d_sample.contiguous(), #[1, 1, S, 3]
        filter_mode='linear', 
        boundary_mode='cube')

    cos = torch.sum(rays_d * rays_d_origin, dim=-1).squeeze(1).unsqueeze(-1).repeat(1, 1, 3)
    #[1, S, 3]    
    final_depth = final_depth[...,0:3].squeeze(1)
    
    return final_rgb, final_depth

def load_tensors_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    center_str = data["center"]
    center_values = [float(x.strip()) for x in center_str.split(",")]
    center_tensor = torch.tensor(center_values, dtype=torch.float32)

    scale_str = data["scale"]
    scale_values = [float(x.strip()) for x in scale_str.split(",")]
    scale_tensor = torch.tensor(scale_values, dtype=torch.float32)

    return center_tensor, scale_tensor