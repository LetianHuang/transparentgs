# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  lthuang@smail.nju.edu.cn or 1193897855@qq.com

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from os import makedirs
import numpy as np
import cv2
import imageio
import pyexr
from PIL import Image
from tqdm import tqdm
import argparse

from scene.dataset_readers import CameraInfo
from utils.camera_utils import cameraList_from_camInfos

import json
from typing import NamedTuple
import torch
from gaussian_renderer import probes_bake
import trimesh

from scene.gaussian_model import GaussianModel
from mesh_utils import mesh2gs, create_dodecahedron

class CameraArgs(NamedTuple):
    resolution : int
    data_device : torch.device

def load_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)

def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255
    
def save_image(fn, x : np.ndarray):
    try:
        pyexr.write(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)

def mapping_index_probes(x):
    a = x // 16
    b = x % 16 // 4
    c = x % 4
    return f"{a}{b}{c}"

def mapping_index_probes_abc(x):
    a = x // 16
    b = x % 16 // 4
    c = x % 4
    return a, b, c

def mapping_index_bkg(x):
    return f"{x + 1:04d}"

def load_numpy_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    center_str = data["center"]
    center_values = [float(x.strip()) for x in center_str.split(",")]
    center_numpy = np.array(center_values)

    scale_str = data["scale"]
    scale_values = [float(x.strip()) for x in scale_str.split(",")]
    scale_numpy = np.array(scale_values)

    return center_numpy, scale_numpy

def readCamerasFromJSON(json_path, dsize=(400, 400), num_probes=64, begin_id=0):
    cam_infos = []

    center, scale = load_numpy_from_json(json_path)

    aabb_min = center - scale
    aabb_max = center + scale
    offset = (aabb_max - aabb_min) / 3.

    c2ws = []

    for x in range(4):
        for y in range(4):
            for z in range(4):
                txx = aabb_min + offset * np.array([x, y, z])
                c2ws += [np.array([
                    [1, 0, 0, txx[0]],
                    [0, 1, 0, txx[1]],
                    [0, 0, 1, txx[2]],
                    [0, 0, 0, 1]
                ])]
                        
    fovx = 0.8456711594746306

    num_id = 0

    for idx in range(len(c2ws)):
        a, b, c = mapping_index_probes_abc(idx)

        if num_probes == 8:
            if a not in [0, 3] or b not in [0, 3] or c not in [0, 3]:
                continue
        
        if num_probes == 1:
            if idx != 0:
                continue

        num_id += 1

        if num_id - 1 < begin_id:
            continue

        c2w = c2ws[idx]

        c2w[:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3]) 
        T = w2c[:3, 3]

        W, H = dsize
        
        img = np.zeros((H, W, 3)).astype(np.uint8)
        image = Image.fromarray(img)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovx, FovX=fovx, image=image,
                        image_path="", image_name="", width=image.size[0], height=image.size[1]))
    
    views = cameraList_from_camInfos(cam_infos, resolution_scale=1.0, args=CameraArgs(1, "cuda"))
    return views

def render_set(model_path, views, gaussians, background : torch.Tensor):
    exr_path = os.path.join(model_path, "probes")

    makedirs(exr_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = probes_bake(view, gaussians, background, mode='rgb')
        other_render = probes_bake(view, gaussians, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), mode='depth')

        rendering = torch.nan_to_num(rendering, nan=background.mean().item())
        # other_render = torch.nan_to_num(other_render, nan=0.0)

        other_render = torch.where(other_render < 1e-8, torch.tensor(1000.0).cuda(), other_render)
        other_render = torch.where(other_render > 1e+3, torch.tensor(1000.0).cuda(), other_render)
        other_render = torch.nan_to_num(other_render, nan=1000)

        if True:
            rendering = rendering.detach().cpu().permute(1, 2, 0).numpy()
            other_render = other_render.detach().cpu().permute(1, 2, 0).numpy()
            w, h = 800, 800 
            rendering = cv2.resize(rendering, (w, h))
            other_render = cv2.resize(other_render, (w, h))[..., None]
            rendering = torch.tensor(rendering).float().permute(2, 0, 1)
            other_render = torch.tensor(other_render).float().permute(2, 0, 1)
        
        print(f"view.colmap_id: {view.colmap_id}")
        id_str = mapping_index_probes(view.colmap_id)

        save_image(os.path.join(exr_path, f'{id_str}.exr'), rendering.permute(1, 2, 0).detach().cpu().numpy().astype(np.float16))
        save_image(os.path.join(exr_path, f'{id_str}_depth.exr'), other_render.expand_as(rendering).permute(1, 2, 0).detach().cpu().numpy().astype(np.float16))

def store_numpy_to_json(mesh_name : str, mesh : trimesh.Trimesh, json_path, ratio=1.1, meshproxy_pitch=0.001):
    vertices = mesh.vertices
    gs = mesh2gs(mesh, meshproxy_pitch)
    os.makedirs("./models/meshgs_proxy", exist_ok=True)
    gs.save_ply(f"./models/meshgs_proxy/{mesh_name}")
    x = vertices[..., 0]
    y = vertices[..., 1]
    z = vertices[..., 2]

    minx, miny, minz = np.min(x), np.min(y), np.min(z)
    maxx, maxy, maxz = np.max(x), np.max(y), np.max(z)

    avg = lambda x, y : (x + y) * 0.5
    sub = lambda x, y : (y - x)

    cx, cy, cz = avg(minx, maxx), avg(miny, maxy), avg(minz, maxz)
    sx, sy, sz = sub(minx, maxx), sub(miny, maxy), sub(minz, maxz)

    with open(json_path, 'w') as f:
        meta = {
            "center": f"{cx}, {cy}, {cz}",
            "scale": f"{sx * ratio}, {sy * ratio}, {sz * ratio}"
        }
        json.dump(meta, f)



def render_sets(gs_path, probes_path, mesh_path, scale_ratio, meshproxy_pitch, dsize=(400, 400), white_background=True, num_probes=64, begin_id=0):
    with torch.no_grad():
        gaussians = GaussianModel(3)
        gaussians.load_ply(gs_path)

        json_path = os.path.join(probes_path, "probe.json")
        if mesh_path == '':
            mesh_name = "example.ply"
            mesh = create_dodecahedron()
        elif mesh_path != '':
            mesh_name = os.path.split(opt.mesh)[-1]
            mesh = trimesh.load(opt.mesh, force='mesh', skip_material=True)
        
        store_numpy_to_json(mesh_name, mesh, json_path, scale_ratio, meshproxy_pitch)
        
        views = readCamerasFromJSON(json_path, dsize, num_probes, begin_id=begin_id)

        bg_color = [1,1,1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(probes_path, views, gaussians, background)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gs_path', type=str)
    parser.add_argument('--probes_path', type=str)
    parser.add_argument('--mesh', type=str, default='')

    parser.add_argument('--W', type=int, default=400, help="width")
    parser.add_argument('--H', type=int, default=400, help="height")
    parser.add_argument('--numProbes', type=int, default=8, help="choose 1/8/64")
    parser.add_argument('--begin_id', type=int, default=0)
    parser.add_argument('--scale_ratio', type=float, default=1.001)
    parser.add_argument('--meshproxy_pitch', type=float, default=0.01)

    opt = parser.parse_args()

    if opt.numProbes not in [1, 8, 64]:
        raise NotImplementedError()
    
    makedirs(opt.probes_path, exist_ok=True)

    render_sets(opt.gs_path, opt.probes_path, opt.mesh, opt.scale_ratio, opt.meshproxy_pitch, dsize=(opt.W, opt.H), num_probes=opt.numProbes, begin_id=opt.begin_id)