
import numpy as np
import torch

# CUDA extension
import _raytracing as _backend

import torch

def compute_barycentric_weights(vertices, points):
    a = vertices[:, 0]
    v0 = vertices[:, 1] - a
    v1 = vertices[:, 2] - a
    v2 = points - a

    d00 = torch.einsum('ij,ij->i', v0, v0)
    d01 = torch.einsum('ij,ij->i', v0, v1)
    d11 = torch.einsum('ij,ij->i', v1, v1)
    d20 = torch.einsum('ij,ij->i', v2, v0)
    d21 = torch.einsum('ij,ij->i', v2, v1)

    inv_denom = 1.0 / (d00 * d11 - d01 * d01 + 1e-8)

    w1 = (d11 * d20 - d01 * d21) * inv_denom
    w2 = (d00 * d21 - d01 * d20) * inv_denom
    w0 = 1.0 - w1 - w2

    return torch.stack((w0, w1, w2), dim=-1)

def interp_anything(anything, weight):
    # Interpolation based on barycentric weights
    out = (anything * weight.unsqueeze(-1)).sum(dim=1)

    return out

class RayTracer():
    def __init__(self, vertices, vertex_normals, triangles, have_vertexnormal=False, get_index=False):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]

        if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
        if have_vertexnormal:
            if torch.is_tensor(vertex_normals): vertex_normals = vertex_normals.detach().cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.detach().cpu().numpy()

        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        self.get_index = get_index
        self.vertices = vertices
        self.vertex_normals = vertex_normals

        if get_index:
            self.impl = _backend.create_raytracer_getindex(vertices, vertex_normals, triangles, True)
        else:
            # implementation
            if have_vertexnormal:
                self.impl = _backend.create_raytracer_withnormal(vertices, vertex_normals, triangles)
            else:
                self.impl = _backend.create_raytracer(vertices, triangles)

    def trace(self, rays_o, rays_d, inplace=False):
        # rays_o: torch.Tensor, cuda, float, [N, 3]
        # rays_d: torch.Tensor, cuda, float, [N, 3]
        # inplace: write positions to rays_o, face_normals to rays_d

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        if not inplace:
            # allocate
            positions = torch.empty_like(rays_o)
            normals = torch.empty_like(rays_d)
            face_normals = torch.empty_like(rays_d)
        else:
            positions = rays_o
            normals = rays_d
            face_normals = torch.empty_like(rays_d).cuda()

        depth = torch.empty_like(rays_o[:, 0])
        proxy_index = torch.empty_like(rays_o.int())
        
        # inplace write intersections back to rays_o
        self.impl.trace(rays_o, rays_d, positions, normals, face_normals, depth, proxy_index) # [N, 3]

        positions = positions.view(*prefix, 3)
        normals = normals.view(*prefix, 3)
        face_normals = face_normals.view(*prefix, 3)
        depth = depth.view(*prefix)

        if self.get_index:
            proxy_index = proxy_index.view(*prefix, 3)
            return positions, normals, face_normals, depth, proxy_index

        return positions, normals, face_normals, depth
    
    def trace_api(self, rays_o, rays_d, gbuffer={}, inplace=False):
        assert (self.get_index)
        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()
        prefix = rays_o.detach().shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        with torch.no_grad():
            if not inplace:
                # allocate
                positions = torch.empty_like(rays_o)
                normals = torch.empty_like(rays_d)
                face_normals = torch.empty_like(rays_d)
            else:
                positions = rays_o
                normals = rays_d
                face_normals = torch.empty_like(rays_d).cuda()

            depth = torch.empty_like(rays_o[:, 0])
            proxy_index = torch.empty_like(rays_o.int())
            
            # inplace write intersections back to rays_o
            self.impl.trace(rays_o, rays_d, positions, normals, face_normals, depth, proxy_index) # [N, 3]
            proxy_index = proxy_index.detach().long()

        # differentiable
        positions = rays_o + rays_d * depth[..., None]
        vertices = gbuffer["vertices"]

        weight = compute_barycentric_weights(vertices[proxy_index], positions) # detach ?

        results = {}

        for key in gbuffer:
            if key == "vertices":
                continue
            results[key] = interp_anything(gbuffer[key][proxy_index], weight).view(*prefix, 3)

        results["positions"] = positions.view(*prefix, 3)
        results["depth"] = depth.view(*prefix)

        return results