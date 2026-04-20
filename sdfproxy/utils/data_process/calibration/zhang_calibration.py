import os
import cv2
import numpy as np
from glob import glob
from skimage import measure
import trimesh as trm
import os.path as osp
import matplotlib.pyplot as plt
import torch
import re

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def calibrate(camera_file, K_file=None, w=5, h=5, world_len=23.2, fscale=0.25, kernel=21, tag='fixed', flag='camera'):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    world_point = np.zeros((w * h, 3), np.float32)
    world_points = []
    world_point[:, :2] = np.mgrid[0:w * world_len:world_len,
                     0:h * world_len:world_len].T.reshape(-1, 2)
    if tag=='center':
        world_point[:, 0] = world_point[:, 0] - (w - 1) * world_len / 2.0
        world_point[:, 1] = world_point[:, 1] - (h - 1) * world_len / 2.0
        world_point[:, 1] = -world_point[:, 1]

    imgpoints = []
    image_list = sorted(glob(osp.join(camera_file, '*.jpg')))
    img_names = []
    for fname in image_list:
        img = cv2.imread(fname)
        img = cv2.resize(img, None, fx=fscale, fy=fscale, interpolation=cv2.INTER_CUBIC)
        img = cv2.GaussianBlur(img, (kernel, kernel), 0, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w, h), flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret is True:
            img_names.append(fname.split('_')[-1].split('.')[0])
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # ?
            imgpoints.append(corners)
            world_points.append(world_point)
            print(fname + " success\n")
        else:
            print(fname + ' error\n')

    if K_file is None:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, imgpoints,
                                                           gray.shape[::-1], None, None)
        np.save(osp.join(camera_file, 'k.npy'), mtx)
        np.save(osp.join(camera_file, 'dist.npy'), dist)
        #in
        R = []
        T = []
        for i in range(len(rvecs)):
            r,_ = cv2.Rodrigues(rvecs[i].ravel())
            R.append(r)
            T.append(tvecs[i])
        np.save(osp.join(camera_file, 'R.npy'), np.array(R))
        np.save(osp.join(camera_file, 'T.npy'), np.array(T))
        print('sucess!')
        return np.array(R), np.array(T), img_names
    else:
        K = np.load(K_file)
        dist = np.load(K_file.split('k.npy')[0] + 'dist.npy')
        R = []
        T = []
        total_error = 0
        extrinsic = []
        for i in range(len(world_points)):
            world_point = np.array(world_points[i]).reshape(-1,3)
            img_point = np.array(imgpoints[i]).reshape(-1, 2)
            _, R_exp, T_ = cv2.solvePnP(world_point,
                                        img_point,
                                        K,
                                        dist)
            img_points_repro, _ = cv2.projectPoints(world_points[i], R_exp, T_, K, dist)
            error = cv2.norm(imgpoints[i], np.float32(img_points_repro),
                             cv2.NORM_L2) / len(img_points_repro)

            total_error += error
            R_, _ = cv2.Rodrigues(R_exp)
            R.append(R_)
            T.append(T_)
        print(("Average Error of Reproject: "), total_error / len(world_points))
        return np.array(R), np.array(T), img_names

def rotationMatrixToEulerAngles(R):
    import math
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    x = x * 180 / np.pi
    y = y * 180 / np.pi
    z = z * 180 / np.pi
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta):
    theta = theta*np.pi/180
    import math
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def load_light(file):
    img = cv2.cvtColor(cv2.imread(file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img)
    mask[img>127] = 1
    return mask

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def load_mask(file, fscale=1):
    # img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    img = cv2.resize(img, None, fx=fscale, fy=fscale, interpolation=cv2.INTER_CUBIC)

    mask = np.zeros_like(img)
    mask[img>127] = 1
    return mask

def compute_visualhull(args, projections, K , box=200):

    # initialize visual hull voxels
    """
    R  [num,3,3]
    K [3,3]
    T [num,3]
    """
    resolution = 256

    minX, maxX = -box, box
    minY, maxY = -box, box
    minZ, maxZ = -box, box

    y, x, z = np.meshgrid(
        np.linspace(minX, maxX, resolution),
        np.linspace(minY, maxY, resolution),
        np.linspace(minZ, maxZ, resolution)
    )

    x = x[:, :, :, np.newaxis]
    y = y[:, :, :, np.newaxis]
    z = z[:, :, :, np.newaxis]
    coord = np.concatenate([x, y, z], axis=3) #[res,res,res,3]

    volume = -np.ones(x.shape).squeeze()#[res,res,res,1] ==>[res,res,res]


    f = K[0][0]
    cxp = K[0][2]
    cyp = K[1][2]
    base_dir = osp.join(args.path, args.mode)
    exp_dir = osp.join(base_dir, 'exp')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    meshName = os.path.join(exp_dir, 'visualHull_{}.ply'.format(len(projections)+1))
    mask_lists = glob(osp.join(base_dir, 'mask', "*.jpg"))
    mask_lists.sort(key=natural_keys)
    print(len(mask_lists))
    origins =[]
    for i in range(len(mask_lists)):
        projection = projections[i]
        R = projection[:3, :3]
        T = projection[:3, 3]
        print("{}:{}".format(i+1, len(mask_lists)))
        print("{}:\n{}".format(mask_lists[i], rotationMatrixToEulerAngles(R)))
        mask  = cv2.imread(mask_lists[i], flags=0) #1 8 3 0 8 1
        # mask = load_mask(mask_lists[i])
        imgH = mask.shape[0]
        imgW = mask.shape[1]
        mask = mask.reshape(imgH * imgW)
        origin = -np.matmul(np.transpose(R), T)
        target = R[2, :] + origin
        origins.append(origin)
        up = -R[1, :]
        print("{}:{}".format(origin, target))
        yAxis = up / np.sqrt(np.sum(up * up))
        zAxis = target - origin
        zAxis = zAxis / np.sqrt(np.sum(zAxis * zAxis))
        xAxis = np.cross(zAxis, yAxis)
        xAxis = xAxis / np.sqrt(np.sum(xAxis * xAxis))
        Rot = np.stack([xAxis, yAxis, zAxis], axis=0)

        coordCam = np.matmul(Rot, np.expand_dims(coord - origin, axis=4))

        coordCam = coordCam.squeeze(4)
        xCam = coordCam[:, :, :, 0] / coordCam[:, :, :, 2]
        yCam = coordCam[:, :, :, 1] / coordCam[:, :, :, 2]

        xId = xCam * f + cxp
        yId = -yCam * f + cyp

        xInd = np.logical_and(xId >= 0, xId < imgW - 0.5)
        yInd = np.logical_and(yId >= 0, yId < imgH - 0.5)

        imInd = np.logical_and(xInd, yInd)

        xImId = np.round(xId[imInd]).astype(np.int32)
        yImId = np.round(yId[imInd]).astype(np.int32)

        maskInd = mask[yImId * imgW + xImId]

        volumeInd = imInd.copy()

        volumeInd[imInd == 1] = maskInd

        volume[volumeInd == 0] = 1
        print('Occupied voxel: %d' % np.sum((volume > 0).astype(np.float32)))

        verts, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0)

        print('Vertices Num: %d' % verts.shape[0])
        print('Normals Num: %d' % normals.shape[0])
        print('Faces Num: %d' % faces.shape[0])

        axisLen = float(resolution - 1) / 2.0

        verts = (verts - axisLen) / axisLen * box
        mesh = trm.Trimesh(vertices=verts, vertex_normals=normals, faces=faces)

        mesh.export(osp.join(exp_dir, os.path.basename(meshName.replace('.ply', '_{}.ply'.format(i)))))


    print('Export final mesh !')
    meshName = osp.join(base_dir, args.mode+'_vh.ply')
    mesh.export(meshName)

def compute_R_T(args=None, basedir=None, imgC1_files=None, imgC2_files=None, K_file=None, recompute=False, imgscreen_files=None ):
    projection_file = osp.join(basedir, 'w2c_mats.npy')
    if not osp.exists(projection_file) or recompute:

        imgC2_files = osp.join(basedir, imgC2_files)
        camera_K_file = osp.join(imgC2_files, 'k.npy')
        r2, t2, img_name2 = calibrate(camera_file=imgC2_files,
                                      K_file=None, w=11, h=8,
                                      world_len=6, fscale=1,
                                      tag='center', kernel=1)
        rk = r2[1:]
        tk = t2[1:]
        r2_ = r2[0]
        t2_ = t2[0]
        for i in range(len(rk)):
            print(rotationMatrixToEulerAngles(rk[i]))

        K_file = osp.join(basedir, imgC1_files, K_file)
        imgC1_files = osp.join(basedir, imgC1_files)

        r1, t1, img_name1 = calibrate(camera_file=imgC1_files,
                                      K_file=K_file, w=11, h=8,
                                      world_len=6, fscale=0.5, tag='center', kernel=1)
        r1 = r1[0]
        t1_ = t1[0]

        K_file = osp.join(basedir, imgC1_files, K_file)
        imgscreen_files = osp.join(basedir, imgscreen_files)
        rs, ts, img_names = calibrate(camera_file=imgscreen_files,
                                      K_file=K_file, w=13, h=6,
                                      world_len=args.world_len_screen, fscale=0.5, tag='center', kernel=1)
        rs = rs[0]
        ts = ts[0]

        len_p = len(rk)
        R1 = np.matmul(np.tile(r1[None, :, :, ], (len_p, 1, 1)), np.matmul(np.tile(r2_.transpose()[None, :, :], (
            len_p, 1, 1)), rk))

        T1 = np.matmul(np.tile(r1[None, :, :,], (len_p, 1, 1)), np.matmul(np.tile(r2_.transpose()[None, :, :], (
            len_p, 1, 1)),
             tk - np.tile(t2_[None, :, :], (len_p, 1, 1)))) \
             + np.tile(t1_[None, :, :], (len_p, 1, 1))
        T1 = T1[..., 0]
        print("---------------------------------------------------------")
        for i in range(len(R1)):
            origin = -np.matmul(np.transpose(R1[i]), T1[i])
            target = R1[i][2, :] + origin
            print("{}:{}".format(origin, target))
        print("---------------------------------------------------------")
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c_mats = []
        for i in range(len(R1)):
            m = np.concatenate([np.concatenate([R1[i], T1[i][:,None]], 1), bottom], 0)
            w2c_mats.append(m)

        poses = np.linalg.inv(w2c_mats)
        for i in range(len(R1)):
            print(rotationMatrixToEulerAngles(R1[i]))

        np.save(osp.join(basedir, 'w2c_mats.npy'), np.array(w2c_mats))
        K1 = np.load(K_file)
        np.save(osp.join(basedir, 'K.npy'), K1)

        screen_w2c_mats = np.concatenate([np.concatenate([rs, ts], 1), bottom], 0)
        np.save(osp.join(basedir, 'screen_w2c_mats.npy'), screen_w2c_mats)
        screen_projection = screen_w2c_mats
        projections = w2c_mats
        np.save(osp.join(basedir, 'poses.npy'), poses)
    else:
        K1 = np.load(osp.join(basedir, 'K.npy'))
        projections = np.load(osp.join(basedir, 'w2c_mats.npy'))
        screen_projection = np.load(osp.join(basedir, 'screen_w2c_mats.npy'))
    return K1, projections, screen_projection

