#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import gbuffer_pass
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
import uuid
from shader.NVDIFFREC.util import reflect, refract, rgb_to_srgb, srgb_to_rgb
from util import load_probes1, load_probes8, load_probes64, probes_sample_1, probes_sample_8, probes_sample_64,  load_tensors_from_json
import numpy as np

import trimesh
from rtproxy import DiffMeshGSUnification

def unification(cam_pos, kd, positions, normals, alpha, 
                probes, num_probes, iters, 
                RT : DiffMeshGSUnification,
                IOR, 
                iteration=0,
                inplace=True):
    positions = positions.permute(1, 2, 0).reshape(-1, 3)
    normals = normals.permute(1, 2, 0).reshape(-1, 3) # [H, W, 3]

    valid_rays_mask = (alpha[0] > 0.5).detach()
    valid_rays_mask = valid_rays_mask.reshape(-1)

    views = F.normalize(positions - cam_pos, p=2, dim=-1).detach()

    rt_pos, _, rt_normal, _ = RT.trace(torch.zeros_like(views) + cam_pos, views)
    # print(f"rt_normal.shape: {rt_normal.shape}, normals.shape: {normals.shape}, valid_rays_mask.shape")
    extra_losses = {
        "rt_normal" : 0.2 * (1 - (rt_normal[valid_rays_mask] * normals[valid_rays_mask]).sum(dim=0))[None].mean()
    }
    extra_losses = {}

    return extra_losses

def deferred_refract(cam_pos, kd, positions, normals, alpha, 
                probes, num_probes, iters, 
                RT : DiffMeshGSUnification, 
                IOR, # spatially varying?
                iteration=0,
                transparency=None,
                inplace=True
                ):
    kd = kd.permute(1, 2, 0) # [H, W, 3]
    if True:
        positions = positions.permute(1, 2, 0) # [H, W, 3]
        normals = normals.permute(1, 2, 0) # [H, W, 3]
        valid_rays_mask = (alpha[0] > 0.5).detach()

        H, W = kd.shape[:2]
        positions = positions.reshape(-1, 3)
        normals = normals.reshape(-1, 3)
        valid_rays_mask = valid_rays_mask.reshape(-1)

        views = F.normalize(positions - cam_pos, p=2, dim=-1).detach()

        if True:
            rt_pos, _, rt_normal, _ = RT.trace(torch.zeros_like(views) + cam_pos, views)
            extra_losses = {
                "rt_normal" : 0.2 * (1 - (rt_normal[valid_rays_mask] * normals[valid_rays_mask]).sum(dim=0))[None].mean()
            }
            positions = rt_pos
            normals = rt_normal
    
        reflect_dir = reflect(-views, normals)
        refract_dir, attenuate, total_internal_reflection = refract(views, normals, 1.0, IOR)

        rays_d_refract = refract_dir.contiguous().view(-1, 3)
        rays_o_refract = positions.contiguous().view(-1, 3)

        refract_positions, _, refract_normals, _ = RT.trace(rays_o_refract - normals.detach() * 1e-3, rays_d_refract)

        rays_d_refract_2, attenuate2, total_internal_reflection2 = refract(rays_d_refract, -refract_normals,
                                                                            IOR, 1.0)

        final_refract_color = torch.zeros_like(normals)
        final_reflect_color = torch.zeros_like(normals)
        chunk_size = 800 * 40 # 800 * 40

        query_raysl = torch.cat([positions, reflect_dir], -1)
        query_raysl = query_raysl[valid_rays_mask]
        query_raysr = torch.cat([refract_positions, rays_d_refract_2], -1)
        query_raysr = query_raysr[valid_rays_mask]
        valid_N = len(query_raysr)

        tmp_colorl = torch.zeros_like(query_raysl[..., :3])
        tmp_colorr = torch.zeros_like(query_raysr[..., :3])
        
        for j in range(0, valid_N, chunk_size):
            rays_refl = query_raysl[j:j + chunk_size]
            rays_refr = query_raysr[j:j + chunk_size]
            if num_probes == 1:
                refl_rgb, refl_depth = probes_sample_1(rays_refl, probes, iters)
                refr_rgb, refr_depth = probes_sample_1(rays_refr, probes, iters)
            elif num_probes == 8:
                refl_rgb, refl_depth = probes_sample_8(rays_refl, probes, iters)
                refr_rgb, refr_depth = probes_sample_8(rays_refr, probes, iters)
            elif num_probes == 64:
                refl_rgb, refl_depth = probes_sample_64(rays_refl, probes, iters)
                refr_rgb, refr_depth = probes_sample_64(rays_refr, probes, iters)

            tmp_colorl[j:j + chunk_size] = refl_rgb
            tmp_colorr[j:j + chunk_size] = refr_rgb

        final_reflect_color[valid_rays_mask] = tmp_colorl
        final_refract_color[valid_rays_mask] = tmp_colorr

        final_reflect_color = srgb_to_rgb(final_reflect_color.reshape(H, W, 3))
        final_refract_color = srgb_to_rgb(final_refract_color.reshape(H, W, 3))

        attenuate = attenuate.reshape(H, W, 1)

    if transparency is None:
        final_color = rgb_to_srgb(((final_refract_color * (1 - attenuate) + final_reflect_color * attenuate)) * kd)
    else:
        final_color = rgb_to_srgb(((final_refract_color * (1 - attenuate) * transparency + final_reflect_color * attenuate * (1 - transparency))) * kd)

    final_color = final_color.view(H, W, 3).permute(2, 0, 1)

    return final_color, extra_losses
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    if opt.stage > 1:
        scene = Scene(dataset, gaussians, load_iteration=7000) 
    else:
        scene = Scene(dataset, gaussians)
    
    if opt.stage > 1:

        num_probes = dataset.num_probes
        probes_iters = dataset.query_iters

        probe_center, probe_scale = load_tensors_from_json(os.path.join(dataset.probes_path, "probe.json"))

        if num_probes == 1:
            probes = load_probes1(os.path.join(dataset.probes_path, "probes"), probe_center, probe_scale)
        elif num_probes == 8:
            probes = load_probes8(os.path.join(dataset.probes_path, "probes"), probe_center, probe_scale)
        elif num_probes == 64:
            probes = load_probes64(os.path.join(dataset.probes_path, "probes"), probe_center, probe_scale)
        else:
            print("invalid num_probes")
            exit(-1)

        meshgs_proxy = DiffMeshGSUnification(trimesh.load(opt.mesh))
        # meshgs_proxy.mergeGaussian(gaussians)

    gaussians.training_setup(opt)
    if opt.stage > 1:
        meshgs_proxy.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if opt.stage > 1:
            meshgs_proxy.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
        
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_alpha_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if opt.stage > 1:
            meshgs_proxy.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            if opt.stage > 1:
                meshgs_proxy.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = gbuffer_pass(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        rend_alpha = render_pkg['rend_alpha']
        surf_points = render_pkg['surf_points']
        surf_depth = render_pkg['surf_depth']
        gt_image = viewpoint_cam.original_image.cuda()
        gt_mask = viewpoint_cam.gt_alpha_mask.cuda()

        extra_losses = {}
        if opt.stage > 1:
            if iteration >= 3000: # 
                shading_result, extra_losses = deferred_refract(
                    cam_pos=(viewpoint_cam.world_view_transform.T).inverse()[:3,3],
                    kd=image,
                    positions=surf_points,
                    normals=rend_normal,
                    alpha=gt_mask,
                    probes=probes,
                    num_probes=num_probes,
                    iters=probes_iters,
                    RT=meshgs_proxy,
                    IOR=gaussians.get_IOR,
                    iteration=iteration
                )
                image = shading_result
            elif iteration % 500 == 0:
                with torch.no_grad():
                    shading_result, _ = deferred_refract(
                        cam_pos=(viewpoint_cam.world_view_transform.T).inverse()[:3,3],
                        kd=image,
                        positions=surf_points,
                        normals=rend_normal,
                        alpha=rend_alpha,
                        probes=probes,
                        num_probes=num_probes,
                        iters=probes_iters,
                        RT=meshgs_proxy,
                        IOR=gaussians.get_IOR,
                        iteration=iteration
                    )
            else:
                extra_losses = unification(
                    cam_pos=(viewpoint_cam.world_view_transform.T).inverse()[:3,3],
                    kd=image,
                    positions=surf_points,
                    normals=rend_normal,
                    alpha=gt_mask,
                    probes=probes,
                    num_probes=num_probes,
                    iters=probes_iters,
                    RT=meshgs_proxy,
                    IOR=gaussians.get_IOR,
                    iteration=iteration
                )

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal # if iteration > 7000 else 0.0 xxxxxxxxxxxxxxxxx
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        lambda_mask = opt.lambda_mask

        alpha_loss = lambda_mask * l1_loss(rend_alpha, gt_mask)
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss + alpha_loss

        for key in extra_losses:
            total_loss += extra_losses[key] * 0.2

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_alpha_for_log = 0.4 * alpha_loss.item() + 0.6 * ema_alpha_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "alpha" : f"{ema_alpha_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # save

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                if opt.stage > 1:
                    meshgs_proxy.save_ply(iteration)

            if opt.stage > 1 and iteration % 500 == 0:
                mesh_path = os.path.join(scene.model_path, "mesh", f"iteration_{iteration}")
                os.makedirs(mesh_path, exist_ok=True)
                meshgs_proxy.extract_mesh(os.path.join(mesh_path, "mesh.ply"), 
                                        object_bbox_min=np.array(probe_center) - np.array(probe_scale), 
                                        object_bbox_max=np.array(probe_center) + np.array(probe_scale))


            # Densification
            if iteration < opt.densify_until_iter and opt.stage <= 1:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if opt.stage > 1:
                    meshgs_proxy.get_optimizer.step()
                    meshgs_proxy.get_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")