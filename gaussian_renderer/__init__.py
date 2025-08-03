import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import diff_gaussian_rasterization_panorama
import diff_gaussian_rasterization_fisheye
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def simple_render(pc : GaussianModel, center, c2w : np.ndarray, FoVx, FoVy, H, W, bg_color=[0, 0, 0], scale_modifier=1.0, proxy : GaussianModel = None, pc_label=0, proxy_label=1):
    return _raw_simple_render(pc, center, c2w, FoVx, FoVy, H, W, bg_color, scale_modifier, proxy, pc_label, proxy_label)

def simple_render_fisheye(pc : GaussianModel, center, c2w : np.ndarray, FoVx, FoVy, H, W, bg_color=[0, 0, 0], scale_modifier=1.0, proxy : GaussianModel = None, pc_label=0, proxy_label=1):
    return _raw_simple_render(pc, center, c2w, FoVx, FoVy, H, W, bg_color, scale_modifier, proxy, pc_label, proxy_label, 
                             GRSetttings=diff_gaussian_rasterization_fisheye.GaussianRasterizationSettings, 
                             GRasterizer=diff_gaussian_rasterization_fisheye.GaussianRasterizer)

def simple_render_panorama(pc : GaussianModel, center, c2w : np.ndarray, FoVx, FoVy, H, W, bg_color=[0, 0, 0], scale_modifier=1.0, proxy : GaussianModel = None, pc_label=0, proxy_label=1):
    return _raw_simple_render(pc, center, c2w, FoVx, FoVy, H, W, bg_color, scale_modifier, proxy, pc_label, proxy_label, 
                             GRSetttings=diff_gaussian_rasterization_panorama.GaussianRasterizationSettings, 
                             GRasterizer=diff_gaussian_rasterization_panorama.GaussianRasterizer)


def _raw_simple_render(pc : GaussianModel, center, c2w : np.ndarray, FoVx, FoVy, H, W, bg_color=[0, 0, 0], scale_modifier=1.0, proxy : GaussianModel = None, pc_label=0, proxy_label=1,
                      GRSetttings=GaussianRasterizationSettings, GRasterizer=GaussianRasterizer):

    if proxy is None:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    else:
        screenspace_points = torch.zeros_like(torch.cat([pc.get_xyz, proxy.get_xyz], 0), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    FoVx = FoVx / 180 * math.pi
    FoVy = FoVy / 180 * math.pi
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    center = torch.tensor(center, dtype=torch.float32, device="cuda")
    
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    
    raster_settings = GRSetttings(
        image_height=int(H),
        image_width=int(W),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scale_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GRasterizer(raster_settings=raster_settings)
    
    if proxy is None:
        means3D = pc.get_xyz
    else:
        means3D = torch.cat([pc.get_xyz, proxy.get_xyz], 0)
    means2D = screenspace_points

    if proxy is None:
        opacity = pc.get_opacity
    else:
        opacity = torch.cat([pc.get_opacity, proxy.get_opacity], 0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if proxy is None:
        scales = pc.get_scaling
    else:
        scales = torch.cat([pc.get_scaling, proxy.get_scaling / min(scale_modifier, 1e-5)], 0)

    if rotations is None:
        rotations = pc.get_rotation
    else:
        rotations = torch.cat([pc.get_rotation, proxy.get_rotation], 0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    
    if proxy is None:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = torch.ones_like(pc.get_xyz) * pc_label
        colors_precomp = torch.cat([colors_precomp, torch.ones_like(proxy.get_xyz) * proxy_label], 0)

    # return colors_precomp
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return rendered_image.permute(1, 2, 0)

def probes_bake(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, mode = 'rgb'):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = diff_gaussian_rasterization_panorama.GaussianRasterizationSettings(
        image_height=viewpoint_camera.image_height,
        image_width=viewpoint_camera.image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = diff_gaussian_rasterization_panorama.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if False:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    
    if mode == 'rgb':
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        depth = torch.norm(means3D - viewpoint_camera.camera_center, dim=-1, keepdim=True)
        colors_precomp = torch.cat([depth, depth, depth], -1)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    if mode == 'rgb':
        return rendered_image
    else:
        return rendered_image[:1, ...]