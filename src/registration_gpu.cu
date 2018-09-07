#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>

namespace cuda = stk::cuda;

__global__ void ssd_kernel(
    cuda::VolumePtr<float> fixed,
    cuda::VolumePtr<float> moving,
    cuda::VolumePtr<float4> df,
    dim3 dims,
    float3 fixed_origin,
    float3 fixed_spacing,
    float3 moving_origin,
    float3 inv_moving_spacing,
    float3 delta,
    cuda::VolumePtr<float2> cost_acc
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z)
    {
        return;
    }

    float3 d0 { df(x,y,z).x, df(x,y,z).y, df(x,y,z).z };
    float3 d1 = d0 + delta;

    float3 world_p = fixed_origin + float3{float(x),float(y),float(z)} * fixed_spacing; 
    
    float3 moving_p0 = (world_p + d0 - moving_origin) * inv_moving_spacing; 
    float3 moving_p1 = (world_p + d1 - moving_origin) * inv_moving_spacing; 

    float f0 = fixed(x,y,z) - cuda::linear_at_border<float>(
        moving, dims, moving_p0.x, moving_p0.y, moving_p0.z);
    
    float f1 = fixed(x,y,z) - cuda::linear_at_border<float>(
        moving, dims, moving_p1.x, moving_p1.y, moving_p1.z);
    
    cost_acc(x,y,z).x = cost_acc(x,y,z).x + f0*f0;
    cost_acc(x,y,z).y = cost_acc(x,y,z).y + f1*f1;
}

void gpu_compute_unary_cost(
    const stk::GpuVolume& fixed,
    const stk::GpuVolume& moving,
    const stk::GpuVolume& df,
    const float3& delta,
    stk::GpuVolume& unary_cost, // float2
    cuda::Stream& stream
)
{
    dim3 dims = fixed.size();
    float3 inv_moving_spacing = {
        1.0f / moving.spacing().x,
        1.0f / moving.spacing().y,
        1.0f / moving.spacing().z
    };

    dim3 block_size {16,16,1};
    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    ssd_kernel<<<grid_size, block_size, 0, stream>>>(
        fixed,
        moving,
        df,
        dims,
        fixed.origin(),
        fixed.spacing(),
        moving.origin(),
        inv_moving_spacing,
        delta,
        unary_cost
    );
}

__global__ void regularizer_kernel(
    cuda::VolumePtr<float4> df,
    dim3 dims,
    float3 delta,
    float weight,
    cuda::VolumePtr<float4> cost_x,
    cuda::VolumePtr<float4> cost_y,
    cuda::VolumePtr<float4> cost_z
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z)
    {
        return;
    }

    // Assume spacing is 1,1,1

    // Cost ordered as f_same, f01, f10, f_same

    float4 o_x = {0, 0, 0, 0};
    float4 o_y = {0, 0, 0, 0};
    float4 o_z = {0, 0, 0, 0};

    float3 d = {df(x,y,z).x, df(x,y,z).y, df(x,y,z).z};
    float3 dx = {df(x+1,y,z).x, df(x+1,y,z).y, df(x+1,y,z).z};
    float3 dy = {df(x,y+1,z).x, df(x,y+1,z).y, df(x,y+1,z).z};
    float3 dz = {df(x,y,z+1).x, df(x,y,z+1).y, df(x,y,z+1).z};

    if (x + 1 < dims.x) {
        float3 diff_00 = d - dx;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
        float3 diff_01 = d - (dx+delta);
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
        float3 diff_10 = (d+delta) - dx;
        float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;
        
        o_x.x = dist2_00;
        o_x.y = dist2_01;
        o_x.z = dist2_10;
    }
    if (y + 1 < dims.y) {
        float3 diff_00 = d - dy;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
        float3 diff_01 = d - (dy+delta);
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
        float3 diff_10 = (d+delta) - dy;
        float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;
        
        o_y.x = dist2_00;
        o_y.y = dist2_01;
        o_y.z = dist2_10;
    }
    if (z + 1 < dims.z) {
        float3 diff_00 = d - dz;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
        float3 diff_01 = d - (dz+delta);
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
        float3 diff_10 = (d+delta) - dz;
        float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;
        
        o_z.x = dist2_00;
        o_z.y = dist2_01;
        o_z.z = dist2_10;
    }

    cost_x(x,y,z) = o_x;
    cost_y(x,y,z) = o_y;
    cost_z(x,y,z) = o_z;
}


void gpu_compute_binary_cost(
    const stk::GpuVolume& df,
    const float3& delta,
    float weight,
    stk::GpuVolume& cost_x, // float4
    stk::GpuVolume& cost_y, // float4
    stk::GpuVolume& cost_z, // float4
    cuda::Stream& stream
)
{
    dim3 dims = df.size();
    dim3 block_size {16,16,1};
    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    regularizer_kernel<<<grid_size, block_size, 0, stream>>>(
        df,
        dims,
        delta,
        weight,
        cost_x,
        cost_y,
        cost_z
    );
}

__global__ void apply_displacement_delta(
    cuda::VolumePtr<float4> df,
    cuda::VolumePtr<uint8_t> labels,
    dim3 dims,
    float4 delta
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z)
    {
        return;
    }

    df(x,y,z) = df(x,y,z) + delta * labels(x,y,z);
}

void gpu_apply_displacement_delta(
    stk::GpuVolume& df, 
    const stk::GpuVolume& labels, 
    const float3& delta,
    cuda::Stream& stream
)
{
    dim3 dims = df.size();
    dim3 block_size {16,16,1};
    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    apply_displacement_delta<<<grid_size, block_size, 0, stream>>>(
        df, 
        labels, 
        dims, 
        float4{delta.x, delta.y, delta.z, 0.0f}
    );
}
