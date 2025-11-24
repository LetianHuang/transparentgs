import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import diff_gaussian_rasterization_panorama
import diff_gaussian_rasterization_fisheye
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

##########################################################################
## Thanks to 2DGS: SIGGRAPH 2024
##########################################################################
import diff_surfel_rasterization
from scene.gaussian_model2d import GaussianModel2D
from utils.point_utils import depths_to_points, CameraTempView

def surfel_splatting(pc : GaussianModel2D, center, c2w : np.ndarray, FoVx, FoVy, H, W, bg_color=[0, 0, 0], scale_modifier=1.0, depth_ratio=1.0, colors_extension=None):
    GRSetttings=diff_surfel_rasterization.GaussianRasterizationSettings
    GRasterizer=diff_surfel_rasterization.GaussianRasterizer
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0

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

    means3D = pc.get_xyz
    means2D = screenspace_points

    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc.get_scaling
    if pc.get_scaling.shape[1] == 3:
        # TODO
        scales = scales[..., :2]
        ####### 1. To be compatible with 3DGS. ##########################
        ####### 2. Support a variety of complex camera models. ##########################
        

    rotations = pc.get_rotation

    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None


    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    if colors_extension is not None:
        colors_precomp = colors_precomp * colors_extension

    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median

    surf_points = depths_to_points(CameraTempView(world_view_transform, W, H, full_proj_transform), surf_depth)

    surf_points = surf_points.reshape(*surf_depth.shape[1:], 3)

    # # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    # surf_normal = surf_normal.permute(2,0,1)
    # # remember to multiply with accum_alpha since render_normal is unnormalized.
    # surf_normal = surf_normal * (render_alpha).detach()
    return rendered_image.permute(1, 2, 0), render_normal.permute(1, 2, 0), surf_depth.permute(1, 2, 0), surf_points

################### 2DGS #################################################

##########################################################################
## Thanks to GS^3: SIGGRAPH 2024
##########################################################################
import diff_gaussian_rasterization_light

def shadow_splatting(lit_pos, pc : GaussianModel, H, W, bg_color=[0, 0, 0], scale_modifier=1, shadow_color=0.6): # 0.3
    FoVx = 0.5 * math.pi
    FoVy = 0.5 * math.pi
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    c2w = np.array([
                    [1, 0, 0, lit_pos[0]],
                    [0, 1, 0, lit_pos[1]],
                    [0, 0, 1, lit_pos[2]],
                    [0, 0, 0, 1]
                ])
    
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    raster_settings_light = diff_gaussian_rasterization_light.GaussianRasterizationSettings(
        image_height = int(H),
        image_width = int(W),
        tanfovx = tanfovx,
        tanfovy = tanfovy,
        bg = bg_color[:3],
        scale_modifier = scale_modifier,
        viewmatrix = world_view_transform,
        projmatrix = full_proj_transform,
        sh_degree = pc.active_sh_degree,
        campos = torch.tensor(lit_pos, device="cuda", dtype=torch.float32),
        prefiltered = False,
        debug = False,
    )
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    if pc.get_scaling.shape[1] == 2:
        ####### To be compatible with 2DGS. ##########################
        third_axis_4_2dgs = 1e-5
        scales = torch.cat([pc.get_scaling, torch.ones_like(pc.get_scaling[..., :1]) * third_axis_4_2dgs], -1)

    rotations = pc.get_rotation

    # shadow splatting
    with torch.no_grad():
        rasterizer_light = diff_gaussian_rasterization_light.GaussianRasterizer(raster_settings=raster_settings_light)
        opcacity_light = torch.zeros(scales.shape[0], dtype=torch.float32, device=scales.device)
        _, out_weight, _, shadow = rasterizer_light(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = torch.zeros((2, 3), dtype=torch.float32, device=scales.device),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            non_trans = opcacity_light,
            offset = 0.015,
            thres = 4,
            is_train = False
        )
        
        opcacity_light = torch.clamp_min(opcacity_light, 1e-10)
        shadow = shadow / opcacity_light
        
        
        shadow = shadow[..., None]
        shadow = torch.where(shadow > 1e-6, torch.ones_like(shadow), torch.ones_like(shadow) * shadow_color)
        shadow = torch.cat([shadow, shadow, shadow], -1)
    
    return shadow            
#################################### GS^3 #############################################

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
    
    if pc.get_scaling.shape[1] == 2:
        ####### To be compatible with 2DGS. ##########################
        third_axis_4_2dgs = 1e-5
        pc_get_scaling = torch.cat([pc.get_scaling, torch.ones_like(pc.get_scaling[..., :1]) * third_axis_4_2dgs], -1)
    else:
        pc_get_scaling = pc.get_scaling
    if proxy is None:
        scales = pc_get_scaling
    else:
        scales = torch.cat([pc_get_scaling, proxy.get_scaling / min(scale_modifier, 1e-5)], 0)

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
        if pc.get_scaling.shape[1] == 2:
            ####### To be compatible with 2DGS. ##########################
            third_axis_4_2dgs = 1e-5
            scales = torch.cat([pc.get_scaling, torch.ones_like(pc.get_scaling[..., :1]) * third_axis_4_2dgs], -1)
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