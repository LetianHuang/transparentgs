import collections
import json
import os
import struct
from glob import glob
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


ColmapCamera = collections.namedtuple(
    'ColmapCamera', ['id', 'model', 'width', 'height', 'params']
)
ColmapImage = collections.namedtuple(
    'ColmapImage', ['id', 'qvec', 'tvec', 'camera_id', 'name']
)

COLMAP_CAMERA_MODELS = {
    0: ('SIMPLE_PINHOLE', 3),
    1: ('PINHOLE', 4),
    2: ('SIMPLE_RADIAL', 4),
    3: ('RADIAL', 5),
    4: ('OPENCV', 8),
    5: ('OPENCV_FISHEYE', 8),
    6: ('FULL_OPENCV', 12),
    7: ('FOV', 5),
    8: ('SIMPLE_RADIAL_FISHEYE', 4),
    9: ('RADIAL_FISHEYE', 5),
    10: ('THIN_PRISM_FISHEYE', 12),
}


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character='<'):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ],
    ], dtype=np.float32)


def read_colmap_intrinsics_text(path):
    cameras = {}
    with open(path, 'r', encoding='utf-8') as fid:
        for raw_line in fid:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])), dtype=np.float32)
            cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)
    return cameras


def read_colmap_intrinsics_binary(path):
    cameras = {}
    with open(path, 'rb') as fid:
        num_cameras = read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, 'iiQQ')
            model_name, num_params = COLMAP_CAMERA_MODELS[model_id]
            params = np.array(read_next_bytes(fid, 8 * num_params, 'd' * num_params), dtype=np.float32)
            cameras[camera_id] = ColmapCamera(camera_id, model_name, width, height, params)
    return cameras


def read_colmap_extrinsics_text(path):
    images = {}
    with open(path, 'r', encoding='utf-8') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])), dtype=np.float32)
            tvec = np.array(tuple(map(float, elems[5:8])), dtype=np.float32)
            camera_id = int(elems[8])
            name = elems[9]
            fid.readline()
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name)
    return images


def read_colmap_extrinsics_binary(path):
    images = {}
    with open(path, 'rb') as fid:
        num_images = read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_images):
            props = read_next_bytes(fid, 64, 'idddddddi')
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float32)
            tvec = np.array(props[5:8], dtype=np.float32)
            camera_id = props[8]
            name_chars = []
            current_char = read_next_bytes(fid, 1, 'c')[0]
            while current_char != b'\x00':
                name_chars.append(current_char.decode('utf-8'))
                current_char = read_next_bytes(fid, 1, 'c')[0]
            num_points2d = read_next_bytes(fid, 8, 'Q')[0]
            if num_points2d > 0:
                read_next_bytes(fid, 24 * num_points2d, 'ddq' * num_points2d)
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, ''.join(name_chars))
    return images


def read_colmap_points_text(path):
    points = []
    with open(path, 'r', encoding='utf-8') as fid:
        for raw_line in fid:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            elems = line.split()
            points.append([float(elems[1]), float(elems[2]), float(elems[3])])
    if not points:
        return None
    return np.asarray(points, dtype=np.float32)


def read_colmap_points_binary(path):
    points = []
    with open(path, 'rb') as fid:
        num_points = read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 43, 'QdddBBBd')
            points.append(props[1:4])
            track_length = read_next_bytes(fid, 8, 'Q')[0]
            if track_length > 0:
                read_next_bytes(fid, 8 * track_length, 'ii' * track_length)
    if not points:
        return None
    return np.asarray(points, dtype=np.float32)


def focal2fov(focal, pixels):
    return 2.0 * np.arctan(pixels / (2.0 * focal))


def resolve_existing_path(root, candidates):
    for candidate in candidates:
        path = Path(root) / candidate
        if path.exists():
            return path
    return None


def resolve_image_path(data_dir, relative_path):
    rel_path = Path(relative_path)
    candidates = []
    if rel_path.suffix:
        candidates.append(rel_path)
    else:
        for suffix in ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'):
            candidates.append(rel_path.with_suffix(suffix))
    for candidate in candidates:
        full_path = Path(data_dir) / candidate
        if full_path.exists():
            return full_path
    raise FileNotFoundError(f'Unable to resolve image path for {relative_path}')


def build_intrinsics_matrix(fx, fy, cx, cy):
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    return intrinsics


def build_scale_mat(center, radius):
    radius = max(float(radius), 1e-4)
    scale_mat = np.eye(4, dtype=np.float32)
    scale_mat[0, 0] = radius
    scale_mat[1, 1] = radius
    scale_mat[2, 2] = radius
    scale_mat[:3, 3] = np.asarray(center, dtype=np.float32)
    return scale_mat


def compute_normalization_from_points(points, scale_factor):
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    radius = (bbox_max - bbox_min).max() * 0.5 * scale_factor
    return center.astype(np.float32), max(radius, 1e-4)


def compute_normalization_from_camera_centers(centers, scale_factor):
    centers = np.asarray(centers, dtype=np.float32)
    center = centers.mean(axis=0)
    radius = np.linalg.norm(centers - center[None, :], axis=1).max()
    if radius < 1e-4:
        radius = 1.0
    radius *= scale_factor
    return center.astype(np.float32), float(radius)


def colmap_camera_to_intrinsics(camera):
    params = camera.params
    model = camera.model
    if model in {'SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'}:
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model in {'PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'THIN_PRISM_FISHEYE', 'FOV'}:
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        raise ValueError(f'Unsupported COLMAP camera model: {model}')
    return build_intrinsics_matrix(fx, fy, cx, cy)


def load_mask_from_file(mask_path, height, width):
    mask_image = cv.imread(str(mask_path), cv.IMREAD_UNCHANGED)
    if mask_image is None:
        raise FileNotFoundError(f'Unable to read mask: {mask_path}')
    if mask_image.ndim == 3:
        mask_image = mask_image[:, :, 0]
    if mask_image.shape[:2] != (height, width):
        mask_image = cv.resize(mask_image, (width, height), interpolation=cv.INTER_NEAREST)
    return (mask_image > 127).astype(np.float32)[..., None]


def load_alpha_mask_from_image(image_path):
    with Image.open(image_path) as image:
        rgba = image.convert('RGBA')
        alpha = np.asarray(rgba, dtype=np.uint8)[:, :, 3]
    return (alpha > 0).astype(np.float32)[..., None]


def build_default_mask(height, width):
    return np.ones((height, width, 1), dtype=np.float32)


def compute_mask_bounds(masks_np):
    mask_bounds = []
    for mask in masks_np:
        coords_y, coords_x = np.where(mask[:, :, 0] > 0.5)
        if coords_y.size == 0 or coords_x.size == 0:
            h, w = mask.shape[:2]
            mask_bounds.append([0, h - 1, 0, w - 1])
        else:
            mask_bounds.append([coords_y.min(), coords_y.max(), coords_x.min(), coords_x.max()])
    return np.asarray(mask_bounds, dtype=np.float32)


def load_colmap_frames(data_dir, images_dir):
    sparse_dir = Path(data_dir) / 'sparse' / '0'
    if (sparse_dir / 'images.bin').exists() and (sparse_dir / 'cameras.bin').exists():
        extrinsics = read_colmap_extrinsics_binary(sparse_dir / 'images.bin')
        intrinsics = read_colmap_intrinsics_binary(sparse_dir / 'cameras.bin')
    elif (sparse_dir / 'images.txt').exists() and (sparse_dir / 'cameras.txt').exists():
        extrinsics = read_colmap_extrinsics_text(sparse_dir / 'images.txt')
        intrinsics = read_colmap_intrinsics_text(sparse_dir / 'cameras.txt')
    else:
        raise FileNotFoundError('COLMAP sparse model not found under sparse/0')

    point_cloud = None
    if (sparse_dir / 'points3D.bin').exists():
        point_cloud = read_colmap_points_binary(sparse_dir / 'points3D.bin')
    elif (sparse_dir / 'points3D.txt').exists():
        point_cloud = read_colmap_points_text(sparse_dir / 'points3D.txt')

    frames = []
    image_root = Path(data_dir) / images_dir
    for image_id, image in sorted(extrinsics.items(), key=lambda item: item[1].name):
        camera = intrinsics[image.camera_id]
        intrinsics_mat = colmap_camera_to_intrinsics(camera)
        rotation = qvec2rotmat(image.qvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = image.tvec
        relative_image_path = Path(image.name)
        image_path = image_root / relative_image_path
        if not image_path.exists():
            image_path = image_root / relative_image_path.name
        frames.append({
            'image_id': image_id,
            'image_name': Path(image.name).stem,
            'image_path': str(image_path),
            'intrinsics': intrinsics_mat,
            'w2c': w2c,
            'height': int(camera.height),
            'width': int(camera.width),
        })
    return frames, point_cloud


def load_nerf_frames(data_dir):
    transform_files = [
        'transforms_train.json',
        'transforms_val.json',
        'transforms_test.json',
        'transforms.json',
    ]
    frames = []
    seen_paths = set()
    for transform_name in transform_files:
        transform_path = Path(data_dir) / transform_name
        if not transform_path.exists():
            continue
        with open(transform_path, 'r', encoding='utf-8') as file:
            transform_data = json.load(file)
        for frame_idx, frame in enumerate(transform_data.get('frames', [])):
            image_path = resolve_image_path(data_dir, frame['file_path'])
            image_path_key = str(image_path.resolve())
            if image_path_key in seen_paths:
                continue
            seen_paths.add(image_path_key)
            with Image.open(image_path) as image:
                width, height = image.size
            if 'fl_x' in frame or 'fl_x' in transform_data:
                fx = float(frame.get('fl_x', transform_data.get('fl_x')))
                fy = float(frame.get('fl_y', transform_data.get('fl_y', fx)))
                cx = float(frame.get('cx', transform_data.get('cx', width * 0.5)))
                cy = float(frame.get('cy', transform_data.get('cy', height * 0.5)))
            else:
                camera_angle_x = float(frame.get('camera_angle_x', transform_data.get('camera_angle_x')))
                fx = width / (2.0 * np.tan(camera_angle_x * 0.5))
                if 'camera_angle_y' in frame or 'camera_angle_y' in transform_data:
                    camera_angle_y = float(frame.get('camera_angle_y', transform_data.get('camera_angle_y')))
                    fy = height / (2.0 * np.tan(camera_angle_y * 0.5))
                else:
                    fy = fx
                cx = float(frame.get('cx', transform_data.get('cx', width * 0.5)))
                cy = float(frame.get('cy', transform_data.get('cy', height * 0.5)))

            c2w = np.asarray(frame['transform_matrix'], dtype=np.float32)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w).astype(np.float32)
            frames.append({
                'image_id': len(frames),
                'image_name': image_path.stem,
                'image_path': str(image_path),
                'intrinsics': build_intrinsics_matrix(fx, fy, cx, cy),
                'w2c': w2c,
                'height': int(height),
                'width': int(width),
            })
    if not frames:
        raise FileNotFoundError('No NeRF transform files found in dataset directory')
    return frames


def load_masks_for_frames(data_dir, frames):
    mask_dir = Path(data_dir) / 'mask'
    masks = []
    mask_paths = []
    for frame in frames:
        if mask_dir.exists():
            mask_path = resolve_existing_path(mask_dir, [
                frame['image_name'] + '.png',
                frame['image_name'] + '.jpg',
                frame['image_name'] + '.jpeg',
                Path(frame['image_path']).name,
            ])
        else:
            mask_path = None

        if mask_path is not None:
            mask = load_mask_from_file(mask_path, frame['height'], frame['width'])
            mask_paths.append(str(mask_path))
        else:
            try:
                mask = load_alpha_mask_from_image(frame['image_path'])
            except Exception:
                mask = build_default_mask(frame['height'], frame['width'])
            mask_paths.append(frame['image_path'])
        masks.append(mask)
    return np.stack(masks, axis=0), mask_paths


def load_screen_points_or_default(screen_point_path, n_images, height, width):
    if screen_point_path and Path(screen_point_path).exists():
        screen_points = np.load(screen_point_path).astype(np.float32)
        if screen_points.shape[0] != n_images:
            raise ValueError('screen_point.npy does not match image count')
        return screen_points, True
    return np.zeros((n_images, height, width, 3), dtype=np.float32), False


def load_light_masks_or_default(data_dir, frames, screen_points, has_screen_points):
    light_mask_dir = Path(data_dir) / 'light_mask'
    light_masks = []
    for frame_idx, frame in enumerate(frames):
        light_mask_path = None
        if light_mask_dir.exists():
            light_mask_path = resolve_existing_path(light_mask_dir, [
                frame['image_name'] + '.png',
                frame['image_name'] + '.jpg',
                frame['image_name'] + '.jpeg',
                Path(frame['image_path']).name,
            ])
        if light_mask_path is not None:
            light_mask = load_mask_from_file(light_mask_path, frame['height'], frame['width'])
        elif has_screen_points:
            light_mask = (np.abs(screen_points[frame_idx, :, :, :1]).sum(axis=-1, keepdims=True) > 0).astype(np.float32)
        else:
            light_mask = np.ones((frame['height'], frame['width'], 1), dtype=np.float32)
        light_masks.append(light_mask)
    return np.stack(light_masks, axis=0)

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.dataset_type = conf.get_string('dataset_type', default='idr').lower()
        self.render_cameras_name = conf.get_string('render_cameras_name', default='cameras_sphere.npz')
        self.object_cameras_name = conf.get_string('object_cameras_name', default=self.render_cameras_name)
        self.screen_point_name = conf.get_string('screen_point_name', default='screen_point.npy')
        self.images_dir = conf.get_string('images_dir', default='images')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        if self.dataset_type == 'idr':
            self._load_idr_dataset()
        elif self.dataset_type == 'colmap':
            self._load_colmap_dataset()
        elif self.dataset_type == 'nerf':
            self._load_nerf_dataset()
        else:
            raise ValueError(f'Unsupported dataset_type: {self.dataset_type}')

        self.B, self.H, self.W = self.masks_np.shape[0], self.masks_np.shape[1], self.masks_np.shape[2]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.light_masks = torch.from_numpy(self.light_mask_np.astype(np.float32)).cpu() # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.masks_bound = torch.from_numpy(self.masks_bound_np.astype(np.float32)).cpu()
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

        self.image_pixels = self.H * self.W
        self.screen_point = torch.from_numpy(self.screen_point_np.astype(np.float32)).cpu()#[n_image, H*W, 3]

        object_bbox_min = np.array([-1.01, -1.01, -1.01], dtype=np.float32)
        object_bbox_max = np.array([1.01, 1.01, 1.01], dtype=np.float32)
        self.object_bbox_min = object_bbox_min
        self.object_bbox_max = object_bbox_max

        if self.dataset_type == 'idr':
            object_bbox_min_h = np.array([-1.01, -1.01, -1.01, 1.0], dtype=np.float32)
            object_bbox_max_h = np.array([1.01, 1.01, 1.01, 1.0], dtype=np.float32)
            object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            object_bbox_min_h = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min_h[:, None]
            object_bbox_max_h = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max_h[:, None]
            self.object_bbox_min = object_bbox_min_h[:3, 0]
            self.object_bbox_max = object_bbox_max_h[:3, 0]

        print('Load data: End')

    def _load_idr_dataset(self):
        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.screen_point_np = np.load(os.path.join(self.data_dir, self.screen_point_name)).astype(np.float32)
        self.has_screen_points = True
        self.mask_only = False

        self.masks_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'mask')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
        self.masks_np = (self.masks_np[:, :, :, :1] > 0.8).astype(np.float32)
        self.n_images = len(self.masks_lis)
        self.masks_bound_np = compute_mask_bounds(self.masks_np)

        self.light_mask_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'light_mask')))
        self.light_mask_np = np.stack([cv.imread(im_name) for im_name in self.light_mask_lis]) / 256.0
        self.light_mask_np = (self.light_mask_np[:, :, :, :1] > 0.8).astype(np.float32)

        self.world_mats_np = [camera_dict[f'world_mat_{idx}'].astype(np.float32) for idx in range(self.n_images)]
        self.scale_mats_np = [camera_dict[f'scale_mat_{idx}'].astype(np.float32) for idx in range(self.n_images)]

    def _load_colmap_dataset(self):
        frames, sparse_points = load_colmap_frames(self.data_dir, self.images_dir)
        self._load_generic_dataset(frames, sparse_points=sparse_points)

    def _load_nerf_dataset(self):
        frames = load_nerf_frames(self.data_dir)
        self._load_generic_dataset(frames, sparse_points=None)

    def _load_generic_dataset(self, frames, sparse_points=None):
        self.n_images = len(frames)
        self.masks_np, self.masks_lis = load_masks_for_frames(self.data_dir, frames)
        self.masks_bound_np = compute_mask_bounds(self.masks_np)

        camera_centers = [np.linalg.inv(frame['w2c'])[:3, 3] for frame in frames]
        if sparse_points is not None and len(sparse_points) > 0:
            center, radius = compute_normalization_from_points(sparse_points, self.scale_mat_scale)
        else:
            center, radius = compute_normalization_from_camera_centers(camera_centers, self.scale_mat_scale)
        scale_mat = build_scale_mat(center, radius)

        self.world_mats_np = []
        self.scale_mats_np = []
        for frame in frames:
            world_mat = (frame['intrinsics'] @ frame['w2c']).astype(np.float32)
            self.world_mats_np.append(world_mat)
            self.scale_mats_np.append(scale_mat.copy())

        screen_point_path = Path(self.data_dir) / self.screen_point_name
        self.screen_point_np, self.has_screen_points = load_screen_points_or_default(
            screen_point_path,
            self.n_images,
            frames[0]['height'],
            frames[0]['width'],
        )
        self.light_mask_np = load_light_masks_or_default(
            self.data_dir,
            frames,
            self.screen_point_np,
            self.has_screen_points,
        )
        self.mask_only = not self.has_screen_points
        self.camera_dict = None

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        mask = self.masks[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        # load screen point

        ray_point = self.screen_point[img_idx][(pixels_y, pixels_x)].to(self.device)
        valid_mask = self.light_masks[img_idx][(pixels_y, pixels_x)].to(self.device)  #[bacthsize,1]
        mask = mask.to(self.device)

        return torch.cat([rays_o, rays_v, ray_point, mask, valid_mask], dim=-1), pixels_x, pixels_y

        # batch_size, 10

    def gen_random_rays_at_mask(self, img_idx, uncertain_map, batch_size):
        """
        Generate mask image rays at world space from one camera.
        """
        pixels_y, pixels_x  = torch.where(uncertain_map)# torch.where return {H, W }
        num = (uncertain_map).sum()
        index = torch.randint(low=0, high=num, size=[batch_size], dtype=int)
        pixels_x, pixels_y = pixels_x[index], pixels_y[index]

        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        # load screen point

        ray_point = self.screen_point[img_idx][(pixels_y, pixels_x)].to(self.device)
        valid_mask = self.light_masks[img_idx][(pixels_y, pixels_x)].to(self.device)  # [bacthsize,1]
        mask = mask.to(self.device)

        return torch.cat([rays_o, rays_v, ray_point, mask, valid_mask], dim=-1), pixels_x, pixels_y


    def gen_ray_masks_near(self, img_idx, batch_size):
        pixels_y = torch.randint(low=np.max([self.masks_bound_np[img_idx][0]-150, 0]),
                                 high=np.min([self.masks_bound_np[img_idx][1]+150, self.H]),
                                 size=[batch_size], dtype=int)#heigh
        pixels_x = torch.randint(low=np.max([self.masks_bound_np[img_idx][2]-150, 0]),
                                 high= np.min([self.masks_bound_np[img_idx][3]+150, self.W]),
                                 size=[batch_size], dtype=int)#wifth

        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        # load screen point
        # index = pixels_y * self.W + pixels_x

        ray_point = self.screen_point[img_idx][(pixels_y, pixels_x)].to(self.device)
        valid_mask = self.light_masks[img_idx][(pixels_y, pixels_x)].to(self.device)  # [bacthsize,1]
        mask = mask.to(self.device)

        return torch.cat([rays_o, rays_v, ray_point, mask, valid_mask], dim=-1), pixels_x, pixels_y

    def gen_ray_at_mask(self, img_idx, batch_size):
        pixels_y, pixels_x, _ = torch.where(self.light_masks[img_idx] > 0.9) #torch.where return {H, W }
        num = (self.light_masks[img_idx] > 0.9).sum()
        index = torch.randint(low=0, high=num, size=[batch_size])
        pixels_x, pixels_y = pixels_x[index], pixels_y[index]

        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        # load screen point
        ray_point = self.screen_point[img_idx][(pixels_y, pixels_x)].to(self.device)
        valid_mask = self.light_masks[img_idx][(pixels_y, pixels_x)].to(self.device)  # [bacthsize,1]
        mask = mask.to(self.device)
        return torch.cat([rays_o, rays_v, ray_point, mask, valid_mask], dim=-1), pixels_x, pixels_y


    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        mask = (self.masks_np[idx] * 255.0).astype(np.uint8)
        mask = np.repeat(mask, 3, axis=2)
        return cv.resize(mask, (self.W // resolution_level, self.H // resolution_level)).clip(0, 255)