
import sys
sys.path.append("..")
from Python_EnvMatt.EM.EnvMatt import *
from Python_EnvMatt.calibration.zhang_calibration import *
from glob import glob

import numpy as np
import os.path as osp
import argparse
import h5py
import trimesh
import os
import cv2

def save_data_drt(K, projections, screen_projection, corresp_path, masks_path, views, name, h5_dir='/mnt/data3/lzc/DRT_data/'):

    screen_points = []
    pixel_distance = 0.24825
    H = pixel_distance * 1080
    W = pixel_distance * 1920
    corresp_list = glob(osp.join(corresp_path, '*.npy'))
    corresp_list.sort(key= natural_keys)
    light_mask = glob(osp.join(corresp_path, '*.png'))
    light_mask.sort(key=natural_keys)
    mask_lists = glob(osp.join(masks_path, '*.jpg'))
    mask_lists.sort(key= natural_keys)
    masks = []
    for i in range(views):
        """
            calculate the 3D position of the light source point on the screen space
            transform the 3D position into the camera 1 coordinate system
            and then transform it into the turntable coordinate system.
        """
        masks.append(cv2.imread(mask_lists[i])[:,:,0])
        corresp = np.load(corresp_list[i])        # corresp[i] 由像素坐标系转换到图像坐标系
        valid_pixel_index = np.array(load_light(light_mask[i]), dtype=bool)
        pixel_x = corresp[:, :, 0] * pixel_distance - W / 2
        pixel_y = H / 2 - corresp[:, :, 1] * pixel_distance

        corr_point = np.stack([pixel_x, pixel_y, np.ones_like(pixel_y)], axis=2)
        valid_point = corr_point[valid_pixel_index]
        valid_point = (screen_projection[:3, :3] @ valid_point.T + screen_projection[:3, 3:4]).T
        projection = projections[i]
        valid_point = (projection[:3, :3].transpose() @ (valid_point.T - projection[:3, 3:4])).T


        corr_point[valid_pixel_index] = valid_point
        corr_point[~valid_pixel_index] = 0

        index_x = corresp[:, :, 0] >= 1920
        index_y = corresp[:, :, 1] >= 1080
        corr_point[index_x] = 0
        corr_point[index_y] = 0
        volid_screen_point(corr_point[valid_pixel_index].reshape(-1,3), corresp_path, i, 'scal_screen_point')
        screen_points.append(corr_point)
        print(i)
    screen_points = np.array(screen_points)
    print(screen_points.shape)
    with h5py.File(osp.join(h5_dir, name + ".h5"), 'w') as f:
        f.create_dataset('num_view', data = [views])
        f.create_dataset('cam_proj', data = projections[:views])
        f.create_dataset('cam_k', data = K)
        f.create_dataset('screen_position', data = screen_points)
        f.create_dataset('mask', data = np.array(masks))
    print(osp.join(h5_dir, name + ".h5"))
    print('sucess save h5 data !!!!!!')
    return osp.join(h5_dir, name + ".h5"), screen_points.shape[1], screen_points.shape[2]


def volid_screen_point(pts, work_dir, index, name='pose'):
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(work_dir, name + str(index) + '.ply'))

def save_neus(h5_path, mesh_path, work_dir, name, resy, resx, n_images=32):

    h5data = h5py.File(h5_path, 'r')
    print('loading data..............')
    projections = []
    screen_points = []
    masks = []
    light_masks = []
    # resy = 3472//2  # the resolution of the calibrate image
    # resx = 4624//2 # 1280 1920 4624 2312
    for i in range(n_images):
        projections.append(h5data['cam_proj'][i])
        K = h5data['cam_k'][:]
        screen_pixel = h5data['screen_position'][i].reshape(resy, resx, -1)
        mask = h5data['mask'][i]
        valid = screen_pixel[:, :, 0] != 0
        light_masks.append(valid)
        masks.append(mask)
        screen_points.append(screen_pixel)
    print('final load data')

    out_dir = os.path.join(work_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'light_mask'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'ray_point'), exist_ok=True)
    for i, img in enumerate(masks):
        img = np.tile(img[:, :, None], (3))
        image_shape = img.shape
        imgs = np.zeros((image_shape[0] + 100, image_shape[1], image_shape[2]))
        imgs[:image_shape[0], :image_shape[1], :] = img

        cv2.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), imgs)
        cv2.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), imgs)

        light_mask = np.tile(light_masks[i][:, :, None], (3))
        imgs = np.zeros((image_shape[0] + 100, image_shape[1], image_shape[2]))
        imgs[:image_shape[0], :image_shape[1], :] = light_mask * 255
        cv2.imwrite(os.path.join(out_dir, 'light_mask', '{:0>3d}.png'.format(i)), imgs)
    print('final write image ')
    cam_dict = dict()
    intrinsic = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    intrinsic[:3, :3] = K
    for i in range(n_images):
        world_mat = intrinsic @ projections[i]
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)

    pcd = trimesh.load(mesh_path)

    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)

    center = (bbox_max + bbox_min) * 0.5

    box = 1.3 * (bbox_max - bbox_min).max() * 0.5
    scale_mat = np.diag([box, box, box, 1.0]).astype(np.float32)

    scale_mat[:3, 3] = center  # 缩放和平移


    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    #scale the screen point
    scale_screen_points = []
    for i in range(len(screen_points)):
        screen_point = screen_points[i]
        screen_point = screen_point - scale_mat[:3, 3]
        screen_point = screen_point / scale_mat[0][0]
        s_shape = screen_point.shape
        tem = np.zeros((s_shape[0] + 100, s_shape[1], s_shape[2]))
        tem[:s_shape[0], :s_shape[1], :] = screen_point
        scale_screen_points.append(tem)
    print(work_dir)
    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    np.save(os.path.join(out_dir, 'screen_point.npy'), np.array(scale_screen_points))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/data4/Lzc/DRT_data/our/1028')
    parser.add_argument('--mode', type=str, default='cock')
    parser.add_argument('--model', type=str, default='mesh')#neus mesh
    parser.add_argument('--views', type=int, default=31)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--recompute_cam', type=bool, default=False)
    parser.add_argument('--recompute_mask', type=bool, default=False)
    parser.add_argument('--testRT', type=bool, default=True)
    parser.add_argument('--recompute_corr', type=bool, default=False)

    #The checkerboard length displayed on the display, in mm
    parser.add_argument('--world_len_screen', type=float, default=14.9)#
    parser.add_argument('--save_path', type=str, default='/mnt/data2/lzc/data/trans/our/')
    args = parser.parse_args()


    model_path = osp.join(args.path, args.mode)
    mask_path = osp.join(model_path, 'mask')

    if osp.exists(osp.join(model_path, 'K.npy')) and\
        osp.exists(osp.join(model_path, 'w2c_mats.npy')) and (not args.recompute_cam):
        K1 = np.load(osp.join(model_path, 'K.npy'))
        projections = np.load(osp.join(model_path, 'w2c_mats.npy'))
        screen_projection = np.load(osp.join(model_path, 'screen_w2c_mats.npy'))
    else:
        K1, projections, screen_projection = compute_R_T(args, basedir=model_path,
                           imgC1_files='camera1T',
                           imgC2_files='camera2', imgscreen_files='camera1S'
                           , K_file='k.npy', recompute= args.recompute_cam)

    if args.recompute_mask or len(glob(osp.join(model_path, 'mask', "*.jpg"))) != args.views:
        mask_path = compute_mask(args, recompute=True)

    if args.testRT:
        compute_visualhull(args, projections, K1 , box=130, plot3Dcam = True)

    if args.recompute_corr or len(glob(osp.join(model_path, 'corresp', "*.npy"))) != args.views:
        corresp_path = compute_corresp(args,  tag=False)
    else:
        corresp_path = osp.join(model_path, 'corr/')

    save_data_drt(K1, projections, screen_projection, corresp_path, mask_path,
                      args.views, name=args.mode, h5_dir=model_path)
    save_neus(osp.join(model_path, args.mode + ".h5"),
              osp.join(model_path, args.mode + '_vh.ply'),
              '/mnt/data4/Lzc/DRT_data/our/',
              args.mode,
              resy=3472//2,
              resx=4624//2,
              n_images=args.views)


