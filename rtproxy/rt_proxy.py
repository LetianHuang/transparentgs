import trimesh
import raytracing
import torch

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

class MeshRayTracingProxy(RayTracingProxy):

    def __init__(self, mesh_path):
        super().__init__()

        self.mesh = trimesh.load(mesh_path, force='mesh', skip_material=True)
        self.RT = raytracing.RayTracer(self.mesh.vertices, self.mesh.vertex_normals, self.mesh.faces, True)

    def scale(self, func):
        self.mesh.vertices = func(self.mesh.vertices)
        self.RT = raytracing.RayTracer(self.mesh.vertices, self.mesh.vertex_normals, self.mesh.faces, True)
    
    def trace(self, rays_o, rays_d, iteration=0):
        outputs = self.RT.trace(rays_o.contiguous().view(-1, 3), rays_d.contiguous().view(-1, 3), inplace=False)
        positions, flat_normals, normals, depth = outputs

        alpha = (torch.square(normals).sum(-1) >= 1e-10)

        return positions, normals, alpha, 0.0