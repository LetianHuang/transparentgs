from sdfproxy import NeuralSDFRayTracingProxy
from rtproxy import MeshRayTracingProxy
import torch
import numpy as np
import os

import torch
from pyhocon import ConfigFactory
import trimesh

from sdfproxy.models.dataset import Dataset
import argparse
import logging
from tqdm import tqdm
import torch.nn.functional as F
from shader.NVDIFFREC.util import safe_normalize
from utils.loss_utils import l1_loss

from scene.gaussian_primitive import GaussianModel


def mesh_query_func(mesh: trimesh.Trimesh, pts: torch.Tensor):
    pts_np = pts.detach().cpu().numpy()
    
    closest, dist, face_id = trimesh.proximity.closest_point(mesh, pts_np)
    
    normals = mesh.face_normals[face_id]
    vectors = pts_np - closest
    
    sign = torch.sign(torch.sum(vectors * normals, dim=-1))  
    sdf = torch.tensor(dist).to(pts.device).float().unsqueeze(-1) 
    sdf = sign * sdf 

    return sdf


def fast_initialization_mesh(bound_min, bound_max, resolution, sdf_query_func, mesh : trimesh.Trimesh):
    N = 64
    device = bound_min.device
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    loss = 0

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                with torch.no_grad():
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val1 = mesh_query_func(mesh, pts)
                val0 = sdf_query_func(pts)
                loss += l1_loss(val0, val1)
    return loss / (len(X) + len(Y) + len(Z))

def fast_initialization_gs(bound_min, bound_max, resolution, sdf_query_func, gs : GaussianModel):
    N = 64
    device = bound_min.device
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    loss = 0

    gs_query_func = None # TODO

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                with torch.no_grad():
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val1 = gs_query_func(gs, pts)
                val0 = sdf_query_func(pts)
                loss += l1_loss(val0, val1)
    return loss / (len(X) + len(Y) + len(Z))

class StandaloneSDFTrainer:

    def __init__(self, conf_path, case, data_dir=None, mesh=None, base_exp_dir=None, train_end_iter=-1, mcube_threshold=0.0, is_continue=False, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.conf_path = os.path.abspath(conf_path)
        self.case = case
        self.mcube_threshold = mcube_threshold
        self.is_continue = is_continue

        with open(self.conf_path, 'r', encoding='utf-8') as conf_file:
            conf_text = conf_file.read().replace('CASE_NAME', case)
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        if data_dir != '':
            self.conf['dataset.data_dir'] = data_dir
            self.conf['dataset']['data_dir'] = data_dir

        print(f"self.conf['dataset]: {self.conf['dataset']}, data_dir: {data_dir}")

        self.base_exp_dir = self.conf['general.base_exp_dir']
        if base_exp_dir != '':
            self.base_exp_dir = base_exp_dir
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset = Dataset(self.conf['dataset'])
        self.proxy = NeuralSDFRayTracingProxy(
            conf_path=self.conf_path,
            conf=self.conf,
            case=case,
            device=str(self.device),
            end_iter=self.conf.get_int('train.end_iter')
        )

        self.iter_step = 0
        self.end_iter = self.conf.get_int('train.end_iter')
        if train_end_iter != -1:
            self.end_iter = train_end_iter
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.mask_only_training = self.conf.get_bool('train.mask_only', default=False)
        if getattr(self.dataset, 'mask_only', False):
            self.mask_only_training = True
            logging.warning('Dataset has no screen_point supervision. Falling back to mask-only training.')

        self.views = int(self.conf.get_float('train.views', default=72))
        self.writer = None

        if self.is_continue:
            self._load_latest_checkpoint()

        self.RT = MeshRayTracingProxy(mesh_path=mesh)
        if True: # scale
            self.RT.scale(lambda vertices : (vertices - self.dataset.scale_mats_np[0][:3, 3][None]) / self.dataset.scale_mats_np[0][0, 0])

    def _load_latest_checkpoint(self):
        checkpoint_dir = os.path.join(self.base_exp_dir, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            return

        candidates = []
        for name in os.listdir(checkpoint_dir):
            if not name.endswith('.pth'):
                continue
            if name.startswith('ckpt_'):
                candidates.append(name)

        if not candidates:
            return

        candidates.sort()
        self.load_checkpoint(os.path.join(checkpoint_dir, candidates[-1]))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.proxy.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.proxy.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.proxy.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.proxy.optimizer.load_state_dict(checkpoint['optimizerNoColor'])
        self.iter_step = checkpoint['iter_step']
        logging.info('Loaded checkpoint %s', checkpoint_path)

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.proxy.nerf_outside.state_dict(),
            'sdf_network_fine': self.proxy.sdf_network.state_dict(),
            'variance_network_fine': self.proxy.deviation_network.state_dict(),
            'optimizerNoColor': self.proxy.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        checkpoint_dir = os.path.join(self.base_exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{self.iter_step:0>6d}.pth')
        torch.save(checkpoint, checkpoint_path)

    def refraction(self, l, normal, eta1, eta2):
        cos_theta = torch.sum(l * (-normal), dim=1).unsqueeze(1)
        i_p = l + normal * cos_theta
        t_p = eta1 / eta2 * i_p

        t_p_norm = torch.sum(t_p * t_p, dim=1)
        total_reflect_mask = (t_p_norm.detach() > 0.999999).unsqueeze(1)

        t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(1).expand_as(normal) * (-normal)
        t = t_i + t_p
        t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=1), min=1e-10)).unsqueeze(1)

        cos_theta_t = torch.sum(t * (-normal), dim=1).unsqueeze(1)
        e_i = (cos_theta_t * eta2 - cos_theta * eta1) / torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
        e_p = (cos_theta_t * eta1 - cos_theta * eta2) / torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)
        attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach()

        return t, attenuate, total_reflect_mask

    def check_sdf_val(self, intersection1, intersection2):
        with torch.no_grad():
            first_second = intersection2 - intersection1
            rays_d = first_second / torch.linalg.norm(first_second, ord=2, dim=-1, keepdim=True)
            z_vals = torch.linspace(0.0, 1.0, self.proxy.renderer.n_samples, device=self.device)
            check_z_vals = torch.linalg.norm(intersection2 - intersection1, ord=2, dim=-1, keepdim=True) * z_vals[None, :]
            pts = intersection1[:, None, :] + rays_d[:, None, :] * check_z_vals[..., :, None]
            check_sdf = self.proxy.sdf_network.sdf(pts.reshape(-1, 3)).reshape(-1, self.proxy.renderer.n_samples)

            for i in range(self.proxy.renderer.up_sample_steps // 2):
                new_check_z_vals = self.proxy.renderer.up_sample_occulsion(
                    intersection1,
                    rays_d,
                    check_z_vals,
                    check_sdf,
                    self.proxy.renderer.n_importance // (self.proxy.renderer.up_sample_steps // 2),
                    64 * 2 ** i,
                )
                check_z_vals, check_sdf = self.proxy.renderer.cat_z_vals(
                    intersection1,
                    rays_d,
                    check_z_vals,
                    new_check_z_vals,
                    check_sdf,
                    last=False,
                )

            occlusion_sign = check_sdf.sign().detach()
        return occlusion_sign, check_sdf

    def get_image_perm(self):
        if self.views <= 0:
            raise ValueError('train.views must be a positive integer')

        if self.views >= self.dataset.n_images:
            return torch.randperm(self.dataset.n_images)
        if self.dataset.n_images % self.views == 0:
            return torch.linspace(0, self.dataset.n_images - 1, self.dataset.n_images)[::int(self.dataset.n_images // self.views)].int()
        if self.views == 20:
            return torch.tensor([0, 4, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 54, 56, 60, 64, 68])

        indices = np.linspace(0, self.dataset.n_images - 1, self.views)
        indices = np.clip(np.round(indices).astype(np.int64), 0, self.dataset.n_images - 1)
        indices = np.unique(indices)
        return torch.from_numpy(indices).int()

    def _background_rgb(self):
        return self.proxy.background_rgb if self.proxy.use_white_bkgd else None

    def initial_model(self, data):
        rays_o, rays_d, _, mask, _ = data[:, :3], data[:, 3:6], data[:, 6:9], data[:, 9:10], data[:, 10:11][..., 0]
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        with torch.no_grad():
            mesh_pos, mesh_normal, _, _ = self.RT.trace(rays_o, rays_d)
            est_mid = torch.mean((mesh_pos - rays_o) / rays_d, dim=-1, keepdim=True)
            eps = 1e-1 # other value ? accelerate ?
            # print(f"near: {near[0:10, 0]}")
            # print(f"far: {far[0:10, 0]}")
            # print(f"est_mid: {est_mid[0:10, 0]}")
            fine_near = torch.where((near < est_mid) & (est_mid < far), est_mid - (est_mid - near) * eps, near)
            fine_far = torch.where((near < est_mid) & (est_mid < far), est_mid + (far - est_mid) * eps, far)
            near, far = fine_near, fine_far
        if self.proxy.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        render_out = self.proxy.renderer.render(
            rays_o,
            rays_d,
            near,
            far,
            background_rgb=self._background_rgb(),
            cos_anneal_ratio=self.proxy.get_cos_anneal_ratio(self.iter_step),
        )
        
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']

        rend_normal = safe_normalize(render_out['gradients'])

        eikonal_loss = gradient_error
        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
        normal_loss = (1 - (rend_normal[mask[:, 0].detach() > 0.5] * mesh_normal[mask[:, 0].detach() > 0.5]).sum(dim=-1))[None].mean()
        loss = mask_loss * self.proxy.mask_weight + eikonal_loss * self.proxy.igr_weight + normal_loss * 0.2

        self.proxy.optimizer.zero_grad()
        loss.backward()
        self.proxy.optimizer.step()
        self.iter_step += 1

        if self.writer is not None:
            mask_sum = mask.sum() + 1e-5
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)

        return loss

    def detail_reconstruction(self, data):
        rays_o, rays_d, ray_point, mask, valid_mask = data[:, :3], data[:, 3:6], data[:, 6:9], data[:, 9:10], data[:, 10:11][..., 0]
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        valid_mask = valid_mask.bool()

        if self.proxy.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        render_out = self.proxy.renderer.render(
            rays_o,
            rays_d,
            near,
            far,
            background_rgb=self._background_rgb(),
            cos_anneal_ratio=self.proxy.get_cos_anneal_ratio(self.iter_step),
        )
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        normal_1 = render_out['gradients']
        inter_point = render_out['inter_point']

        l_t1, _, total_reflect_mask1 = self.refraction(rays_d, normal_1, eta1=self.proxy.extIOR, eta2=self.proxy.intIOR)
        rays_o = inter_point + l_t1 * 2
        rays_d = -l_t1
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        render_out = self.proxy.renderer.render(
            rays_o,
            rays_d,
            near,
            far,
            background_rgb=self._background_rgb(),
            cos_anneal_ratio=self.proxy.get_cos_anneal_ratio(self.iter_step),
        )
        normal_2 = render_out['gradients']
        inter_point2 = render_out['inter_point']
        render_out_dir2, _, total_reflect_mask2 = self.refraction(-rays_d, -normal_2, eta1=self.proxy.intIOR, eta2=self.proxy.extIOR)

        _, check_sdf = self.check_sdf_val(inter_point, inter_point2)
        occlusion_mask = (check_sdf > 1e-3).sum(1) == 0
        valid_mask = valid_mask & (~total_reflect_mask1[:, 0]) & (~total_reflect_mask2[:, 0]) & occlusion_mask

        target = ray_point - inter_point2.detach()
        target = target / target.norm(dim=1, keepdim=True)
        diff = render_out_dir2 - target
        ray_loss = (diff[valid_mask]).pow(2).sum() if valid_mask.any() else diff.sum() * 0.0

        eikonal_loss = gradient_error
        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
        loss = mask_loss * self.proxy.mask_weight + eikonal_loss * self.proxy.igr_weight + ray_loss * self.proxy.refract_weight

        self.proxy.optimizer.zero_grad()
        loss.backward()
        self.proxy.optimizer.step()
        self.iter_step += 1

        if self.writer is not None:
            mask_sum = mask.sum() + 1e-5
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Loss/ray_loss', ray_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)

        return loss

    def train(self, init_epoch):
        self.proxy.update_learning_rate(self.iter_step)

        remaining_steps = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        for _ in tqdm(range(remaining_steps), desc='Standalone SDF training'):
            image_index = image_perm[self.iter_step % len(image_perm)]
            if (not self.mask_only_training) and self.iter_step >= init_epoch:
                data, _, _ = self.dataset.gen_ray_masks_near(image_index, self.batch_size)
                loss = self.detail_reconstruction(data)
            else:
                data, _, _ = self.dataset.gen_random_rays_at(image_index, self.batch_size)
                loss = self.initial_model(data)

            if self.iter_step % self.report_freq == 0:
                print(f'iter:{self.iter_step:8d} loss = {loss} lr={self.proxy.optimizer.param_groups[0]["lr"]}')

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(world_space=True, resolution=512, threshold=self.mcube_threshold)

            self.proxy.update_learning_rate(self.iter_step)

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

        self.save_checkpoint()
        self.validate_mesh(world_space=True, resolution=512, threshold=self.mcube_threshold)

        if self.writer is not None:
            self.writer.close()

    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0, mesh_path=None):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32, device=self.device)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32, device=self.device)
        vertices, triangles = self.proxy.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        if mesh_path is None:
            mesh_dir = os.path.join(self.base_exp_dir, 'meshes')
            os.makedirs(mesh_dir, exist_ok=True)
            mesh_path = os.path.join(mesh_dir, f'{self.iter_step:0>8d}.ply')
        trimesh.Trimesh(vertices, triangles).export(mesh_path)
        print(mesh_path)


def main():
    parser = argparse.ArgumentParser(description='Standalone NeuralSDFRayTracingProxy trainer')
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate_mesh'])
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--init_epoch', type=int, default=50001)
    parser.add_argument('--mesh', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--base_exp_dir', type=str, default='')
    parser.add_argument('--train_end_iter', type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    trainer = StandaloneSDFTrainer(
        conf_path=args.conf,
        case=args.case,
        mesh=args.mesh,
        data_dir=args.data_dir,
        base_exp_dir=args.base_exp_dir,
        train_end_iter=args.train_end_iter,
        mcube_threshold=args.mcube_threshold,
        is_continue=args.is_continue,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    if args.mode == 'train':
        trainer.train(args.init_epoch)
    else:
        trainer.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)


if __name__ == '__main__':
    format_string = '[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=format_string)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    main()