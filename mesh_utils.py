import trimesh
import numpy as np


from scene.gaussian_model import GaussianModel
from scene.dataset_readers import BasicPointCloud


def mesh2gs(mesh : trimesh.Trimesh, pitch=0.05): # 0.01
    voxelized = mesh.voxelized(pitch=pitch) # 
    xyz = voxelized.points
    
    num_pts = len(xyz)
    pcd = BasicPointCloud(points=xyz, colors=np.ones((num_pts, 3)), normals=np.zeros((num_pts, 3)))
    
    meshGS = GaussianModel(3)
    meshGS.create_from_pcd(pcd, 0, init_opac=1.0)

    return meshGS

def create_dodecahedron(radius=1, center=np.array([0, 0, 0])):
    vertices = np.array([
        -0.57735, -0.57735, 0.57735,
        0.934172, 0.356822, 0,
        0.934172, -0.356822, 0,
        -0.934172, 0.356822, 0,
        -0.934172, -0.356822, 0,
        0, 0.934172, 0.356822,
        0, 0.934172, -0.356822,
        0.356822, 0, -0.934172,
        -0.356822, 0, -0.934172,
        0, -0.934172, -0.356822,
        0, -0.934172, 0.356822,
        0.356822, 0, 0.934172,
        -0.356822, 0, 0.934172,
        0.57735, 0.57735, -0.57735,
        0.57735, 0.57735, 0.57735,
        -0.57735, 0.57735, -0.57735,
        -0.57735, 0.57735, 0.57735,
        0.57735, -0.57735, -0.57735,
        0.57735, -0.57735, 0.57735,
        -0.57735, -0.57735, -0.57735,
    ]).reshape((-1, 3), order="C")

    faces = np.array([
        19, 3, 2,
        12, 19, 2,
        15, 12, 2,
        8, 14, 2,
        18, 8, 2,
        3, 18, 2,
        20, 5, 4,
        9, 20, 4,
        16, 9, 4,
        13, 17, 4,
        1, 13, 4,
        5, 1, 4,
        7, 16, 4,
        6, 7, 4,
        17, 6, 4,
        6, 15, 2,
        7, 6, 2,
        14, 7, 2,
        10, 18, 3,
        11, 10, 3,
        19, 11, 3,
        11, 1, 5,
        10, 11, 5,
        20, 10, 5,
        20, 9, 8,
        10, 20, 8,
        18, 10, 8,
        9, 16, 7,
        8, 9, 7,
        14, 8, 7,
        12, 15, 6,
        13, 12, 6,
        17, 13, 6,
        13, 1, 11,
        12, 13, 11,
        19, 12, 11,
    ]).reshape((-1, 3), order="C") - 1

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    return trimesh.Trimesh(vertices=vertices, faces=faces)