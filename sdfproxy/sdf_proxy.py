import os

import torch
from pyhocon import ConfigFactory

import sys
sys.path.append("./sdfproxy")


from models.fields import SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import numpy as np
import trimesh



import os

import sys

import torch
from pyhocon import ConfigFactory



class RayTracingProxy:

    def trace(self, rays_o, rays_d, iteration):
        """
        return positions, normals, alpha, oth_loss
        """
        pass

    @property
    def get_optimizer(self):
        """
        return optimizer (e.g., Adam)
        """
        pass

    def extract_mesh(self, path, object_bbox_min, object_bbox_max):
        """
        save mesh (.ply or .obj) in the path
        """
        pass

    def update_learning_rate(self, iter_step):
        pass

class NeuralSDFRayTracingProxy(RayTracingProxy):

    def __init__(self, conf_path=None, conf=None, case=None, device=None, end_iter=30_000):
        super().__init__()

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.conf_path = conf_path or os.path.join(base_dir, 'confs', 'base.conf')

        if conf is None:
            with open(self.conf_path, 'r', encoding='utf-8') as conf_file:
                conf_text = conf_file.read()
            if case is not None:
                conf_text = conf_text.replace('CASE_NAME', case)
            self.conf = ConfigFactory.parse_string(conf_text)
        else:
            self.conf = conf

        # Training parameters
        self.end_iter = end_iter
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')

        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.uncertain_map = self.conf.get_bool('train.uncertain_map')

        self.views = self.conf.get_float('train.views', default=72)

        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.extIOR =  self.conf.get_float('train.extIOR', default=1.0003)
        self.intIOR = self.conf.get_float('train.intIOR', default=1.4723)
        self.decay_rate  = self.conf.get_float('train.decay_rate', default=0.1)
        self.n_samples = self.conf.get_int('model.neus_renderer.n_samples', default=0.1)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.refract_weight = self.conf.get_float('train.refract_weight')

        is_continue = False
        self.is_continue = is_continue
        self.mode = 'train'
        self.model_list = []
        self.uncertain_masks = []

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     **self.conf['model.neus_renderer'])

        if self.use_white_bkgd:
            self.background_rgb = torch.ones([1, 3], device=self.device)
        else:
            self.background_rgb = torch.zeros([1, 3], device=self.device)

        self.update_learning_rate(0)

    def _flatten_rays(self, rays_o, rays_d):
        original_shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        return rays_o, rays_d, original_shape

    def _restore_ray_shape(self, tensor, original_shape):
        return tensor.reshape(*original_shape, tensor.shape[-1])

    def _near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / torch.clamp(a, min=1e-8)
        near = mid - 1.0
        far = mid + 1.0
        return near, far
    
    @property
    def get_optimizer(self):
        return self.optimizer

    def trace(self, rays_o, rays_d, iter_step):
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)
        rays_o, rays_d, original_shape = self._flatten_rays(rays_o, rays_d)
        near, far = self._near_far_from_sphere(rays_o, rays_d)

        final_render_out = {}

        chunk_size = 512
        valid_N = len(rays_o)

        for j in range(0, valid_N, chunk_size):
            render_out = self.renderer.render(
                rays_o[j:j + chunk_size],
                rays_d[j:j + chunk_size],
                near[j:j + chunk_size],
                far[j:j + chunk_size],
                background_rgb=self.background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio(iter_step)
            )
            for key in render_out:
                if key not in final_render_out:
                    if key in ["inter_point", "gradients", "weight_sum", "gradient_error"]:
                        final_render_out[key] = render_out[key]
                else:
                    if key in ["inter_point", "gradients", "weight_sum", "gradient_error"]:
                        # print(f"final_render_out: {final_render_out[key].shape}, render_out: {render_out[key].shape}")
                        final_render_out[key] = torch.cat([final_render_out[key], render_out[key]], 0)

        render_out = final_render_out

        positions = self._restore_ray_shape(render_out["inter_point"], original_shape)
        normals = self._restore_ray_shape(render_out["gradients"], original_shape)
        alpha = self._restore_ray_shape(render_out["weight_sum"], original_shape)
        oth_loss = render_out["gradient_error"].mean() * self.igr_weight

        return positions, normals, alpha, oth_loss

    def extract_mesh(self, path, object_bbox_min, object_bbox_max):
        bound_min = torch.as_tensor(object_bbox_min, dtype=torch.float32, device=self.device)
        bound_max = torch.as_tensor(object_bbox_max, dtype=torch.float32, device=self.device)

        resolution = 512
        threshold = 0
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

        mesh = trimesh.Trimesh(vertices, triangles)

        mesh.export(path)

    def update_learning_rate(self, iter_step):
        if self.warm_up_end > 0 and iter_step < self.warm_up_end:
            learning_factor = iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            denom = max(self.end_iter - self.warm_up_end, 1)
            progress = np.clip((iter_step - self.warm_up_end) / denom, 0.0, 1.0)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def get_cos_anneal_ratio(self, iter_step):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, iter_step / self.anneal_end])