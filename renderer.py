# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  lthuang@smail.nju.edu.cn or 1193897855@qq.com

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import trimesh
import torch.nn.functional as F

import torch
import raytracing
from shader.NVDIFFREC.util import reflect, refract, rgb_to_srgb, srgb_to_rgb

from gaussian_renderer import GaussianModel, simple_render, simple_render_panorama, simple_render_fisheye
from utils.graphics_utils import fov2focal, focal2fov

import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from util import load_probes1, load_probes8, load_probes64, probes_sample_1, probes_sample_8, probes_sample_64,  load_tensors_from_json
from mesh_utils import mesh2gs, create_dodecahedron

import math
import cv2

def generate_envmap_directions(H, W, device='cpu'):
    v, u = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, device=device),  # theta
        torch.linspace(0.5, W - 0.5, W, device=device),  # phi
        indexing='ij'
    )

    theta = v / H * math.pi
    phi = u / W * 2 * math.pi

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.cos(theta)
    z = torch.sin(theta) * torch.sin(phi)

    dirs = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
    return dirs

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.fovx = focal2fov(fov2focal(fovy * math.pi / 180, H), W) * 180 / math.pi
        self.center = np.array([0, 0, 0], dtype=np.float32) # np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!
        self.forward = np.array([0, 0, 1], dtype=np.float32)

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focaly = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        focalx = self.W / (2 * np.tan(np.radians(self.fovx) / 2))
        return np.array([focalx, focaly, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def orbit2(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.forward * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz]) # 0.0005


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['rays_m'] = torch.ones_like(rays_o[..., 0])

    return results

@torch.cuda.amp.autocast(enabled=False)
def get_rays_panorama(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])
    
    theta = (H * 0.5 - j) * math.pi / H
    phi = (i - W * 0.5) * 2 * math.pi / W

    xs = torch.cos(theta) * torch.sin(phi)
    ys = -torch.sin(theta)
    zs = torch.cos(theta) * torch.cos(phi)
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['rays_m'] = torch.ones_like(rays_o[..., 0])

    return results

@torch.cuda.amp.autocast(enabled=False)
def get_rays_fisheye(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    tan_fovx = math.tan(focal2fov(fx, W) * 0.5)
    tan_fovy = math.tan(focal2fov(fy, H) * 0.5)
    tan_fovxy = math.sqrt(tan_fovx * tan_fovx + tan_fovy * tan_fovy)
    tmp = 2.0 * math.atan(tan_fovxy)
    fy = H / tmp
    fx = W / tmp

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    xs = (i - cx) / fx
    ys = (j - cy) / fy

    phi = torch.atan2(ys, xs)
    theta = torch.sqrt(xs * xs + ys * ys)

    Th = torch.ones_like(theta) * H / (2 * fy)

    mask = ((theta < Th) & (theta > -Th)) * 1.0

    xs = torch.sin(theta) * torch.cos(phi)
    ys = torch.sin(theta) * torch.sin(phi)
    zs = torch.cos(theta)
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['rays_m'] = mask

    return results

class GUI:
    def __init__(self, opt, debug=True):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.debug = debug
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        # self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.do_upsample = False

        self.spp = 1.0
        self.gs_scale = 1.0

        self.iters = opt.iters
        self.num_probes = opt.numProbes

        # self.model_path = opt.model_path

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation

        self.mode = 'normal'
        self.cam_mode = 'pinhole' # choose from ['pinhole', 'panorama', 'fisheye']
        self.bkg = 'black' # choose from ['black', 'white', '3DGS']
        self.norm_mode = 'smooth'

        # load mesh
        if opt.mesh == '':
            self.mesh = create_dodecahedron()
        else:
            self.mesh = trimesh.load(opt.mesh, force='mesh', skip_material=False) # True
        self.IOR = 1.45

        self.mesh_color = [1.0, 1.0, 1.0, 1.0]

        self.meshgs_proxy = mesh2gs(self.mesh, opt.meshproxy_pitch)

        self.light_type = opt.light_type  # choose from ['envmap', 'GaussProbe']

        # load environmental gaussian (background)
        if opt.gs_path == '':
            self.GS = None
        else:
            self.GS = GaussianModel(3)
            self.GS.load_ply(opt.gs_path)

        self.probe_center, self.probe_scale = load_tensors_from_json(os.path.join(opt.probes_path, "probe.json"))
        #print("start loading probes")
        if self.num_probes == 1:
            self.probes = load_probes1(os.path.join(opt.probes_path, "probes"), self.probe_center, self.probe_scale)
        elif self.num_probes == 8:
            self.probes = load_probes8(os.path.join(opt.probes_path, "probes"), self.probe_center, self.probe_scale)
        elif self.num_probes == 64:
            self.probes = load_probes64(os.path.join(opt.probes_path, "probes"), self.probe_center, self.probe_scale)
        else:
            print("invalid num_probes")
            exit(-1)

        # normalize
        #self.center = self.mesh.vertices.mean(axis=0)
        #self.length = (self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)).max()
        #self.mesh.vertices = (self.mesh.vertices - self.center) / (self.length + 1e-5)
        #print("self.mesh.vertices.shape:", self.mesh.vertices.shape)
        #print("self.mesh.vertex_normals.shape:", self.mesh.vertex_normals.shape)
        print(f'[INFO] load mesh {self.mesh.vertices.shape}, {self.mesh.faces.shape}')

        
        # prepare raytracer
        self.RT = raytracing.RayTracer(self.mesh.vertices, self.mesh.vertex_normals, self.mesh.faces, True)

        dpg.create_context()
        self.register_dpg()
        self.step()

    def __del__(self):
        dpg.destroy_context()

    def render_gaussian(self, mode=["rgb", "label"]):
        with torch.no_grad():
            if self.cam_mode == 'pinhole':
                GSRenderer = simple_render
            elif self.cam_mode == 'panorama':
                GSRenderer = simple_render_panorama
            elif self.cam_mode == 'fisheye':
                GSRenderer = simple_render_fisheye

            gs_render = GSRenderer(self.GS, self.cam.center, self.cam.pose, 
                                                        self.cam.fovx, self.cam.fovy, self.H, self.W, scale_modifier=self.gs_scale).detach().cpu().numpy().copy()
            if "label" in mode:
                gs_label = GSRenderer(self.GS, self.cam.center, self.cam.pose, 
                                                            self.cam.fovx, self.cam.fovy, self.H, self.W, scale_modifier=self.gs_scale, proxy=self.meshgs_proxy).detach().cpu().numpy().copy()
            else:
                return gs_render
            
        return gs_render, gs_label

    def prepare_buffer(self, rays_o, rays_d, rays_m, outputs):
        if self.do_upsample:
            H = int(self.spp * self.H)
            W = int(self.spp * self.W)
        else:
            H = self.H
            W = self.W

        rays_m = rays_m.detach().cpu().numpy().reshape(H, W, 1)

        positions, flat_normals, normals, depth = outputs

        if self.norm_mode == 'raw':
            normals = flat_normals

        cam_pos = torch.from_numpy(self.cam.pose[:3, 3]).cuda()
        views = F.normalize(positions - cam_pos, p=2, dim=1)

        reflect_dir = reflect(-views, normals)
        refract_dir, attenuate, total_internal_reflection = refract(views, normals, 1.0, self.IOR)

        if self.mode == 'position':
            # outputs is the actual 3D point, how to visualize them ???
            # naive normalize...
            positions = positions.detach().cpu().numpy().reshape(H, W, 3)
            positions = (positions - positions.min(axis=0, keepdims=True)) / (
                        positions.max(axis=0, keepdims=True) - positions.min(axis=0, keepdims=True) + 1e-8)
            if self.do_upsample:
                positions = cv2.resize(positions, dsize=(self.W, self.H))
            return positions
        elif self.mode == 'normal':
            # already normalized to [-1, 1]
            mask_normals = normals.detach().cpu().numpy().reshape(H, W, 3) * rays_m
            normals = normals.detach().cpu().numpy().reshape(H, W, 3)
            normals = (normals + 1) * 0.5
            if self.do_upsample:
                normals = cv2.resize(normals, dsize=(self.W, self.H))

            if self.bkg == '3DGS':
                ##################################################################
                #### (+ Environment Gaussian )
                ##################################################################
                mask = (np.square(mask_normals).sum(-1) >= 1e-10)

                mask = mask.astype(np.float32)
                if self.do_upsample:
                    mask = cv2.resize(mask, dsize=(self.W, self.H))
                mask = mask[..., None]

                gs_render, gs_label = self.render_gaussian()
                bkg_color = gs_render
                mask = mask * (gs_label[..., :1]).astype(np.float32)

                normals = normals * mask + bkg_color * (1 - mask)

            return normals
        elif self.mode == 'depth':
            depth = depth.detach().cpu().numpy().reshape(H, W, 1).copy()
            # mask = depth >= 10

            normals = normals.detach().cpu().numpy().reshape(H, W, 3)
            mask = (np.square(normals).sum(-1) < 1e-10)

            mn = depth[~mask].min()
            mx = depth[~mask].max()
            depth = (depth - mn) / (mx - mn + 1e-5)

            depth[mask] = 0.0

            if self.do_upsample:
                depth = cv2.resize(depth, dsize=(self.W, self.H))

            depth = depth.repeat(3, -1)

            return depth
        elif self.mode == 'mask':
            normals = normals.detach().cpu().numpy().reshape(H, W, 3)
            mask = (np.square(normals).sum(-1) >= 1e-10)
            tmp = np.zeros_like(normals)
            tmp[mask] = 1.0
            mask = tmp

            if self.do_upsample:
                mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_AREA)
            return mask
        elif self.mode == 'reflect':
            if self.light_type == 'probes':
                valid_rays_mask = (torch.square(normals).sum(-1) >= 1e-10) # used to accelerate (Some redundant rays do not contribute to the final image.)

                rays_d = rays_d.reshape(1, H, W, 3)
                normals = normals.reshape(1, H, W, 3)

                final_color = torch.zeros_like(rays_o)
                chunk_size = 800 * 40

                
                query_rays = torch.cat([positions, reflect_dir], 1)
                query_rays = query_rays[valid_rays_mask]
                valid_N = len(query_rays)
                tmp_color = torch.zeros_like(query_rays[..., :3])
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()

                for j in range(0, valid_N, chunk_size):
                    rays_refl = query_rays[j:j + chunk_size]
                    if self.num_probes == 1:
                        refl_rgb, refl_depth = probes_sample_1(rays_refl, self.probes, self.iters)
                    elif self.num_probes == 8:
                        refl_rgb, refl_depth = probes_sample_8(rays_refl, self.probes, self.iters)
                    elif self.num_probes == 64:
                        refl_rgb, refl_depth = probes_sample_64(rays_refl, self.probes, self.iters)
                    tmp_color[j:j + chunk_size] = refl_rgb

                final_color[valid_rays_mask] = tmp_color

                ender.record()
                torch.cuda.synchronize()
                t = starter.elapsed_time(ender)

                dpg.set_value("_log_query_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')


                # final_color, _ = probes_sample_8(torch.cat([positions, reflect_dir], -1), self.probes, self.iters)

                final_color = torch.clamp(final_color, 0., 1.)
                final_color = final_color.view(H, W, 3).detach().cpu().numpy()
                if self.do_upsample:
                    final_color = cv2.resize(final_color, (self.W, self.H), interpolation=cv2.INTER_AREA)

                ##################################################################
                #### (+ Environment Gaussian )
                ##################################################################
                normals = normals.detach().cpu().numpy().reshape(H, W, 3) * rays_m
                mask = (np.square(normals).sum(-1) >= 1e-10)

                mask = mask.astype(np.float32)
                if self.do_upsample:
                    mask = cv2.resize(mask, dsize=(self.W, self.H))
                mask = mask[..., None]

                if self.bkg == '3DGS':
                    gs_render, gs_label = self.render_gaussian()
                    bkg_color = gs_render
                    mask = mask * (gs_label[..., :1]).astype(np.float32)
                elif self.bkg == 'black':
                    bkg_color = np.zeros_like(final_color)
                elif self.bkg == 'white':
                    bkg_color = np.ones_like(final_color)

                mesh_color = np.array([[self.mesh_color[:3]]]).astype(np.float32)
                final_color = final_color * mask * mesh_color + bkg_color * (1 - mask)

                return final_color
            else:
                raise NotImplementedError()
        elif self.mode == 'refract':
            #times = 2
            if self.light_type == 'probes':
                valid_rays_mask = (torch.square(normals).sum(-1) >= 1e-10) # used to accelerate (Some redundant rays do not contribute to the final image.)

                rays_d = rays_d.reshape(1, H, W, 3)
                positions_r = positions.reshape(1, H, W, 3)
                normals_r = normals.reshape(1, H, W, 3)

                rays_d_refract = refract_dir.contiguous().view(-1, 3)
                rays_o_refract = positions.contiguous().view(-1, 3)
                outputs_refract = self.RT.trace(rays_o_refract - normals * 1e-3, rays_d_refract, inplace=False)

                refract_positions, refract_flat_normals, refract_normals, refract_depth = outputs_refract
                rays_d_refract_2, attenuate2, total_internal_reflection2 = refract(rays_d_refract, -refract_normals, self.IOR, 1.0)

                final_color = torch.zeros_like(rays_o)
                chunk_size = 800 * 40

                query_rays = torch.cat([refract_positions, rays_d_refract_2], 1)
                query_rays = query_rays[valid_rays_mask]
                valid_N = len(query_rays)
                tmp_color = torch.zeros_like(query_rays[..., :3])
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()

                for j in range(0, valid_N, chunk_size):
                    rays_refr = query_rays[j:j + chunk_size]
                    if self.num_probes == 1:
                        refr_rgb, refr_depth = probes_sample_1(rays_refr, self.probes, self.iters)
                    elif self.num_probes == 8:
                        refr_rgb, refr_depth = probes_sample_8(rays_refr, self.probes, self.iters) # [1, -1, -1]
                    elif self.num_probes == 64:
                        refr_rgb, refr_depth = probes_sample_64(rays_refr, self.probes, self.iters)

                    tmp_color[j:j + chunk_size] = refr_rgb

                final_color[valid_rays_mask] = tmp_color

                ender.record()
                torch.cuda.synchronize()
                t = starter.elapsed_time(ender)

                dpg.set_value("_log_query_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')

                final_color = torch.clamp(final_color, 0., 1.)
                final_color = final_color.view(H, W, 3).detach().cpu().numpy()
                if self.do_upsample:
                    final_color = cv2.resize(final_color, (self.W, self.H), interpolation=cv2.INTER_AREA)

                ##################################################################
                #### (+ Environment Gaussian )
                ##################################################################
                normals = normals.detach().cpu().numpy().reshape(H, W, 3) * rays_m
                mask = (np.square(normals).sum(-1) >= 1e-10)

                mask = mask.astype(np.float32)
                if self.do_upsample:
                    mask = cv2.resize(mask, dsize=(self.W, self.H))
                mask = mask[..., None]

                if self.bkg == '3DGS':
                    gs_render, gs_label = self.render_gaussian()
                    bkg_color = gs_render
                    mask = mask * (gs_label[..., :1]).astype(np.float32)
                elif self.bkg == 'black':
                    bkg_color = np.zeros_like(final_color)
                elif self.bkg == 'white':
                    bkg_color = np.ones_like(final_color)

                mesh_color = np.array([[self.mesh_color[:3]]]).astype(np.float32)

                final_color = final_color * mask * mesh_color + bkg_color * (1 - mask)

                return final_color
            else:
                raise NotImplementedError()
        elif self.mode == 'render':
            if self.light_type == 'probes':
                valid_rays_mask = (torch.square(normals).sum(-1) >= 1e-10) # used to accelerate (Some redundant rays do not contribute to the final image.)

                rays_d_refract = refract_dir.contiguous().view(-1, 3)
                rays_o_refract = positions.contiguous().view(-1, 3)
                outputs_refract = self.RT.trace(rays_o_refract - normals * 1e-3, rays_d_refract, inplace=False)

                # Here, we provide a simple demo, which can be modified as needed to support total internal reflection 
                # and to determine the timing of probe sampling based on whether the ray intersects the mesh.
                refract_positions, refract_flat_normals, refract_normals, refract_depth = outputs_refract
                rays_d_refract_2, attenuate2, total_internal_reflection2 = refract(rays_d_refract, -refract_normals,
                                                                                   self.IOR, 1.0)

                final_refract_color = torch.zeros_like(rays_o)
                final_reflect_color = torch.zeros_like(rays_o)
                chunk_size = 800 * 40


                query_raysl = torch.cat([positions, reflect_dir], 1)
                query_raysl = query_raysl[valid_rays_mask]
                query_raysr = torch.cat([refract_positions, rays_d_refract_2], 1)
                query_raysr = query_raysr[valid_rays_mask]
                valid_N = len(query_raysr)

                tmp_colorl = torch.zeros_like(query_raysl[..., :3])
                tmp_colorr = torch.zeros_like(query_raysr[..., :3])
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                
                for j in range(0, valid_N, chunk_size):
                    rays_refl = query_raysl[j:j + chunk_size]
                    rays_refr = query_raysr[j:j + chunk_size]
                    if self.num_probes == 1:
                        refl_rgb, refl_depth = probes_sample_1(rays_refl, self.probes, self.iters)
                        refr_rgb, refr_depth = probes_sample_1(rays_refr, self.probes, self.iters)
                    elif self.num_probes == 8:
                        refl_rgb, refl_depth = probes_sample_8(rays_refl, self.probes, self.iters)
                        refr_rgb, refr_depth = probes_sample_8(rays_refr, self.probes, self.iters)
                    elif self.num_probes == 64:
                        refl_rgb, refl_depth = probes_sample_64(rays_refl, self.probes, self.iters)
                        refr_rgb, refr_depth = probes_sample_64(rays_refr, self.probes, self.iters)

                    tmp_colorl[j:j + chunk_size] = refl_rgb
                    tmp_colorr[j:j + chunk_size] = refr_rgb

                final_reflect_color[valid_rays_mask] = tmp_colorl
                final_refract_color[valid_rays_mask] = tmp_colorr

                ender.record()
                torch.cuda.synchronize()
                t = starter.elapsed_time(ender)

                dpg.set_value("_log_query_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')

                final_reflect_color = srgb_to_rgb(final_reflect_color.reshape(H, W, 3))
                final_refract_color = srgb_to_rgb(final_refract_color.reshape(H, W, 3))
                attenuate = attenuate.reshape(H, W, 1)
                final_color = rgb_to_srgb(final_refract_color * (1 - attenuate) + final_reflect_color * attenuate)

                final_color = torch.clamp(final_color, 0., 1.)

                final_color = final_color.view(H, W, 3).detach().cpu().numpy()
                if self.do_upsample:
                    final_color = cv2.resize(final_color, (self.W, self.H), interpolation=cv2.INTER_AREA)

                ##################################################################
                #### (+ Environment Gaussian )
                ##################################################################
                normals = normals.detach().cpu().numpy().reshape(H, W, 3) * rays_m
                mask = (np.square(normals).sum(-1) >= 1e-10)

                mask = mask.astype(np.float32)
                if self.do_upsample:
                    mask = cv2.resize(mask, dsize=(self.W, self.H))
                mask = mask[..., None]

                if self.bkg == '3DGS':
                    gs_render, gs_label = self.render_gaussian()
                    bkg_color = gs_render
                    mask = mask * (gs_label[..., :1]).astype(np.float32)
                elif self.bkg == 'black':
                    bkg_color = np.zeros_like(final_color)
                elif self.bkg == 'white':
                    bkg_color = np.ones_like(final_color)

                mesh_color = np.array([[self.mesh_color[:3]]]).astype(np.float32)

                final_color = final_color * mask * mesh_color + bkg_color * (1 - mask)

                return final_color
            else:
                raise NotImplementedError()
        elif self.mode == 'gs_render':
            gs_render = self.render_gaussian(mode=["rgb"])
            return gs_render
        elif self.mode == 'semantic':
            gs_render, gs_label = self.render_gaussian()

            normals = normals.detach().cpu().numpy().reshape(H, W, 3) * rays_m
            mask = (np.square(normals).sum(-1) >= 1e-10)

            mask = mask.astype(np.float32)
            if self.do_upsample:
                mask = cv2.resize(mask, dsize=(self.W, self.H))
            mask = mask[..., None]

            bkg_color = gs_render
            mask = mask * (gs_label[..., :1]).astype(np.float32)

            final_color = np.ones_like(bkg_color) * mask * np.array([165.0 / 255.0, 134.0 / 255.0, 192.0 / 255.0], dtype=np.float32) + bkg_color * (1 - mask)

            return final_color
        else:
            raise NotImplementedError()

    def step(self):
        if self.need_update:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            pose = torch.from_numpy(self.cam.pose).unsqueeze(0).cuda()

            if self.cam_mode == 'pinhole':
                MeshRays = get_rays
            elif self.cam_mode == 'panorama':
                MeshRays = get_rays_panorama
            elif self.cam_mode == 'fisheye':
                MeshRays = get_rays_fisheye

            if self.do_upsample:
                self.cam.H = int(self.cam.H * self.spp)
                self.cam.W = int(self.cam.W * self.spp)
                rays = MeshRays(pose, self.cam.intrinsics, self.cam.H, self.cam.W, -1)
                self.cam.H = self.H
                self.cam.W = self.W
            else:
                rays = MeshRays(pose, self.cam.intrinsics, self.H, self.W, -1)

            rays_o = rays['rays_o'].contiguous().view(-1, 3)
            rays_d = rays['rays_d'].contiguous().view(-1, 3)
            rays_m = rays['rays_m'].contiguous().view(-1, 1)
            outputs = self.RT.trace(rays_o, rays_d, inplace=False)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender) 

            if self.need_update:
                self.render_buffer = self.prepare_buffer(rays_o, rays_d, rays_m, outputs)
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(rays_o, rays_d, rays_m, outputs)) / (
                            self.spp + 1)

            # ender.record()
            # torch.cuda.synchronize()
            # t = starter.elapsed_time(ender)
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')
            dpg.set_value("_texture", self.render_buffer)

    def register_dpg(self):

        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="TransparentGS Control", tag="_control_window", width=330, height=530):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("G-buffer time: ")
                    dpg.add_text("no data", tag="_log_infer_time")
                with dpg.group(horizontal=True):
                    dpg.add_text("IterQuery time: ")
                    dpg.add_text("no data", tag="_log_query_time")
                    
                dpg.add_text("https://letianhuang.github.io/transparentgs/")

            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('position', 'normal', 'depth', 'mask', 'reflect', 'refract', 'render', 'gs_render', 'semantic'), label='gbuffers',
                              default_value=self.mode, callback=callback_change_mode)
                
                def callback_change_cammode(sender, app_data):
                    self.cam_mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('pinhole', 'panorama', 'fisheye'), label='camera', default_value=self.cam_mode, callback=callback_change_cammode)
                
                def callback_change_bkg(sender, app_data):
                    self.bkg = app_data
                    self.need_update = True

                dpg.add_combo(('black', 'white', '3DGS'), label='bkg', default_value=self.bkg, callback=callback_change_bkg)

                def callback_change_normmode(sender, app_data):
                    self.norm_mode = app_data
                    self.need_update = True

                dpg.add_combo(('smooth', 'raw'), label='normal mode', default_value=self.norm_mode, callback=callback_change_normmode)

                def callback_change_numprobes(sender, app_data):
                    self.num_probes = int(app_data)
                    if self.num_probes == 1:
                        self.probes = load_probes1(os.path.join(opt.probes_path, "probes"), self.probe_center, self.probe_scale)
                    elif self.num_probes == 8:
                        self.probes = load_probes8(os.path.join(opt.probes_path, "probes"), self.probe_center, self.probe_scale)
                    elif self.num_probes == 64:
                        self.probes = load_probes64(os.path.join(opt.probes_path, "probes"), self.probe_center, self.probe_scale)
                    else:
                        print("invalid num_probes")
                        exit(-1)
                    self.need_update = True

                dpg.add_combo((1, 8, 64), label='num probes', default_value=self.num_probes, callback=callback_change_numprobes)

                def callback_change_numiters(sender, app_data):
                    self.iters = int(app_data)
                    self.need_update = True

                dpg.add_combo((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), label='num iters', default_value=self.iters, callback=callback_change_numiters)

                # # bg_color picker
                # def callback_change_bg(sender, app_data):
                #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                #     self.need_update = True

                # dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.cam.fovx = focal2fov(fov2focal(app_data * math.pi / 180, self.cam.H), self.cam.W) * 180 / math.pi
                    self.need_update = True

                def callback_set_IOR(sender, app_data):
                    self.IOR = app_data
                    self.need_update = True

                def callback_set_gsscale(sender, app_data):
                    self.gs_scale = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (y)", min_value=1, max_value=180, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)
                dpg.add_slider_float(label="IOR", min_value=1.01, max_value=2.45, format="%f", default_value=self.IOR, callback=callback_set_IOR)
                dpg.add_slider_float(label="GS Scale", min_value=0.00000001, max_value=1.00, format="%f", default_value=self.gs_scale, callback=callback_set_gsscale)

                def callback_set_spp(sender, app_data):
                    self.spp = math.sqrt(app_data)
                    if self.spp > 1.0:
                        self.do_upsample = True
                    else:
                        self.do_upsample = False
                    self.need_update = True
                
                dpg.add_slider_float(label="spp", min_value=1.00, max_value=4.00, format="%f", default_value=self.spp, callback=callback_set_spp)

                def callback_set_meshcolor(sender, app_data):
                    self.mesh_color = app_data
                    self.need_update = True

                dpg.add_color_picker(label="Pick a color", default_value=self.mesh_color, tag="mesh color", callback=callback_set_meshcolor)
    
            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_rotate2(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit2(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_rotate2)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        dpg.create_viewport(title='TransparentGS_viewer', width=self.W, height=self.H, resizable=True)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', default='', type=str)
    parser.add_argument('--light_type', default='probes', type=str)
    parser.add_argument('--gs_path', default='', type=str)
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--probes_path', default='', type=str)
    parser.add_argument('--numProbes', type=int, default=8, help="choose 1/8/64")
    parser.add_argument('--iters', type=int, default=5, help="choose 0-10")
    parser.add_argument('--meshproxy_pitch', type=float, default=0.01)

    opt = parser.parse_args()

    gui = GUI(opt)
    gui.render()
