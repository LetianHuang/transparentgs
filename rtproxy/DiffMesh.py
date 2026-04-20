import trimesh
from scene import GaussianModel
from .rt_proxy import RayTracingProxy
import raytracing
import torch
from utils.sh_utils import eval_sh
import numpy as np

class UnifyOptimizer:
    def __init__(self, lst):
        self.lst = lst
    
    def step(self):
        for x in self.lst:
            x.step()

    def zero_grad(self, set_to_none = True):
        for x in self.lst:
            x.zero_grad(set_to_none)

class DiffMeshGSUnification(RayTracingProxy):
    def __init__(self, mesh : trimesh.Trimesh):
        self.mesh = mesh

        self.meshgs_proxy = GaussianModel(3)
        self.meshgs_proxy.create_from_mesh(mesh, spatial_lr_scale=0)

        self.gs_list = [self.meshgs_proxy]

        with torch.no_grad():
            self.RT = raytracing.RayTracer(
                self.meshgs_proxy.get_xyz.detach().cpu().numpy(),
                self.meshgs_proxy.get_normal.detach().cpu().numpy(),
                self.mesh.faces,
                have_vertexnormal=True,
                get_index=True
            )

    @property
    def get_optimizer(self):
        return self.meshgs_proxy.optimizer

    @property  
    def active_sh_degree(self):
        self.meshgs_proxy.active_sh_degree

    def trace(self, rays_o, rays_d, inplace=False, update_bvh=False):
        if update_bvh:
            with torch.no_grad():
                self.RT = raytracing.RayTracer(
                    self.meshgs_proxy.get_xyz.detach().cpu().numpy(),
                    self.meshgs_proxy.get_normal.detach().cpu().numpy(),
                    self.mesh.faces,
                    have_vertexnormal=True,
                    get_index=True
                )
        outputs = self.RT.trace_api(rays_o.contiguous().view(-1, 3), rays_d.contiguous().view(-1, 3), 
                                                                            gbuffer={
                                                                                "vertices" : self.meshgs_proxy.get_xyz,
                                                                                "normals" : self.meshgs_proxy.get_normal
                                                                            }, inplace=False)
        positions = outputs["positions"]
        normals = outputs["normals"]
        depth = outputs["depth"]

        return positions, normals, normals, depth
        
    @torch.no_grad()
    def extract_mesh(self, path, object_bbox_min, object_bbox_max):
        vertices = (self.meshgs_proxy.get_xyz).detach().cpu().numpy()
        normals = (self.meshgs_proxy.get_normal).detach().cpu().numpy()
        triangles = self.mesh.faces
        self.mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=normals)
        # pc = self.meshgs_proxy
        # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        # dir_pp = (pc.get_xyz)
        # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # colors_precomp = torch.cat([colors_precomp, torch.ones_like(colors_precomp[..., :1])], -1)
        # self.mesh.visual.vertex_colors = np.array(colors_precomp.detach().cpu().numpy() * 255, dtype=np.uint8)
        self.mesh.export(path)

    def mergeGaussian(self, GS : GaussianModel):
        self.gs_list += [GS]

    def training_setup(self, training_args):
        for gs in self.gs_list:
            gs.training_setup(training_args)

    
    def restore(self, model_args, training_args):
        for gs in self.gs_list:
            gs.restore(model_args, training_args)

    def update_learning_rate(self, iteration):
        for gs in self.gs_list:
            gs.update_learning_rate(iteration)

    def oneupSHdegree(self):
        for gs in self.gs_list:
            gs.oneupSHdegree()
    
    @property
    def get_IOR(self):
        return self.gs_list[-1].get_IOR
    
    @property
    def get_xyz(self):
        return torch.cat([gs.get_xyz for gs in self.gs_list], 0)
    
    @property
    def get_opacity(self):
        return torch.cat([gs.get_opacity for gs in self.gs_list], 0)
    
    @property
    def get_scaling(self):
        return torch.cat([gs.get_scaling for gs in self.gs_list], 0)
    
    @property
    def get_rotation(self):
        return torch.cat([gs.get_rotation for gs in self.gs_list], 0)
    
    @property
    def get_features(self):
        return torch.cat([gs.get_features for gs in self.gs_list], 0)
    
    def get_covariance(self, scaling_modifier = 1):
        return torch.cat([gs.get_covariance(scaling_modifier) for gs in self.gs_list], 0)
    
    @property
    def max_radii2D(self):
        return self.gs_list[-1].max_radii2D
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.gs_list[-1].add_densification_stats(viewspace_point_tensor, update_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        self.gs_list[-1].densify_and_prune(max_grad, min_opacity, extent, max_screen_size)

    def reset_opacity(self):
        self.gs_list[-1].reset_opacity()

    @property
    def optimizer(self):
        return UnifyOptimizer([x.optimizer for x in self.gs_list])
    
    def save_ply(self, path):
        self.meshgs_proxy.save_ply(path[:-4] + f"proxy.ply")