#include "trilinear_weights.h"

__global__ void compute_trilinear_weights_kernel(
    const float* positions, const float* p0, const float* p1,
    int grid_size, int num_positions, float* weights) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_positions) return;

    float3 pos = make_float3(positions[idx * 3 + 0], positions[idx * 3 + 1], positions[idx * 3 + 2]);
    float3 p0_val = make_float3(p0[0], p0[1], p0[2]);
    float3 p1_val = make_float3(p1[0], p1[1], p1[2]);

    float3 positions_grid;
    positions_grid.x = (pos.x - p0_val.x) / (p1_val.x - p0_val.x) * (grid_size - 1);
    positions_grid.y = (pos.y - p0_val.y) / (p1_val.y - p0_val.y) * (grid_size - 1);
    positions_grid.z = (pos.z - p0_val.z) / (p1_val.z - p0_val.z) * (grid_size - 1);

    int3 base_idx;
    base_idx.x = floorf(positions_grid.x);
    base_idx.y = floorf(positions_grid.y);
    base_idx.z = floorf(positions_grid.z);
    
    float3 offset;
    offset.x = positions_grid.x - base_idx.x;
    offset.y = positions_grid.y - base_idx.y;
    offset.z = positions_grid.z - base_idx.z;
    
    float total_weights = 0;

    for (int dx = 0; dx < 2; ++dx) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dz = 0; dz < 2; ++dz) {
                int3 neighbor_idx = make_int3(base_idx.x + dx, base_idx.y + dy, base_idx.z + dz);
                if (neighbor_idx.x >= 0 && neighbor_idx.x < grid_size &&
                    neighbor_idx.y >= 0 && neighbor_idx.y < grid_size &&
                    neighbor_idx.z >= 0 && neighbor_idx.z < grid_size) {

                    float weight = ((dx == 0 ? 1.0f - offset.x : offset.x) *
                                    (dy == 0 ? 1.0f - offset.y : offset.y) *
                                    (dz == 0 ? 1.0f - offset.z : offset.z));

                    total_weights += weight;
                    

                    int weight_idx = (neighbor_idx.x * grid_size + neighbor_idx.y) * grid_size + neighbor_idx.z;
                    atomicAdd(&weights[idx * grid_size * grid_size * grid_size + weight_idx], weight);
                }
            }
        }
    }
    //printf("%f", total_weights);
}

torch::Tensor compute_trilinear_weights(
    const torch::Tensor& positions, const torch::Tensor& p0, const torch::Tensor& p1,
    int grid_size) {
    auto weights = torch::zeros({positions.size(0), grid_size, grid_size, grid_size}, positions.options());

    int num_positions = positions.size(0);

    
    dim3 block(256);
    dim3 grid((num_positions + block.x - 1) / block.x);

    
    compute_trilinear_weights_kernel<<<grid, block>>>(
        positions.contiguous().data_ptr<float>(),
        p0.contiguous().data_ptr<float>(),
        p1.contiguous().data_ptr<float>(),
        grid_size,
        num_positions,
        weights.contiguous().data_ptr<float>()
    );

    return weights;
}

