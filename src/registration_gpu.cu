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
    
    cost_acc(x,y,z).x = f0*f0;
    cost_acc(x,y,z).y = f1*f1;
}
inline __device__ bool is_inside(const dim3& dims, const int3& p)
{
    return (p.x >= 0 && p.x < int(dims.x) && p.y >= 0 && p.y < int(dims.y) && p.z >= 0 && p.z < int(dims.z));
}
__global__ void ncc_kernel(
    cuda::VolumePtr<float> fixed,
    cuda::VolumePtr<float> moving,
    cuda::VolumePtr<float4> df,
    dim3 fixed_dims, // Full fixed volume dims
    dim3 moving_dims, // Full moving volume dims
    int3 block_offset,
    int3 block_dims,
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

    if (x >= block_dims.x ||
        y >= block_dims.y ||
        z >= block_dims.z)
    {
        return;
    }

    int gx = x + block_offset.x;
    int gy = y + block_offset.y;
    int gz = z + block_offset.z;

    if (gx >= fixed_dims.x ||
        gy >= fixed_dims.y ||
        gz >= fixed_dims.z)
    {
        return;
    }

    float3 d0 { df(gx,gy,gz).x, df(gx, gy, gz).y, df(gx, gy, gz).z };
    float3 d1 = d0 + delta;

    float3 world_p = fixed_origin + float3{float(gx),float(gy),float(gz)} * fixed_spacing; 
    
    float3 moving_p0 = (world_p + d0 - moving_origin) * inv_moving_spacing; 
    float3 moving_p1 = (world_p + d1 - moving_origin) * inv_moving_spacing; 

    float sff = 0.0f;
    float sf = 0.0f;
    
    float smm0 = 0.0f;
    float sfm0 = 0.0f;
    float sm0 = 0.0f;
    float smm1 = 0.0f;
    float sfm1 = 0.0f;
    float sm1 = 0.0f;
    unsigned int n = 0;

    int radius = 2;

    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                // TODO: Does not account for anisotropic volumes
                int r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > radius * radius)
                    continue;

                int3 fp{gx + dx, gy + dy, gz + dz};
                
                if (!is_inside(fixed_dims, fp))
                    continue;

                float3 mp0{moving_p0.x + dx, moving_p0.y + dy, moving_p0.z + dz};
                float3 mp1{moving_p1.x + dx, moving_p1.y + dy, moving_p1.z + dz};

                float fixed_v = fixed(fp.x, fp.y, fp.z);

                float moving_v0 = cuda::linear_at_border<float>(moving, moving_dims, mp0.x, mp0.y, mp0.z);
                float moving_v1 = cuda::linear_at_border<float>(moving, moving_dims, mp1.x, mp1.y, mp1.z);

                sff += fixed_v * fixed_v;

                smm0 += moving_v0 * moving_v0;
                smm1 += moving_v1 * moving_v1;
                
                sfm0 += fixed_v*moving_v0;
                sfm1 += fixed_v*moving_v1;

                sm0 += moving_v0;
                sm1 += moving_v1;

                sf += fixed_v;

                ++n;
            }
        }
    }

    if (n == 0)
        return;

    // Subtract mean
    sff -= (sf * sf / n);
    smm0 -= (sm0 * sm0 / n);
    sfm0 -= (sf * sm0 / n);
    
    smm1 -= (sm1 * sm1 / n);
    sfm1 -= (sf * sm1 / n);

    float denom0 = sqrt(sff*smm0);
    float denom1 = sqrt(sff*smm1);

    float2 out = {0, 0};
    if(denom0 > 1e-14) {
        out.x = 0.5f*(1.0f-float(sfm0 / denom0));
    }
    if(denom1 > 1e-14) {
        out.y = 0.5f*(1.0f-float(sfm1 / denom1));
    }

    cost_acc(gx,gy,gz) = out;
}

void gpu_compute_unary_cost(
    const stk::GpuVolume& fixed,
    const stk::GpuVolume& moving,
    const stk::GpuVolume& df,
    const int3& block_offset,
    const int3& block_dims,
    const float3& delta,
    stk::GpuVolume& unary_cost, // float2
    cuda::Stream& stream
)
{
    float3 inv_moving_spacing = {
        1.0f / moving.spacing().x,
        1.0f / moving.spacing().y,
        1.0f / moving.spacing().z
    };

    dim3 block_size {16,16,1};
    dim3 grid_size {
        (block_dims.x + block_size.x - 1) / block_size.x,
        (block_dims.y + block_size.y - 1) / block_size.y,
        (block_dims.z + block_size.z - 1) / block_size.z
    };

    ncc_kernel<<<grid_size, block_size, 0, stream>>>(
        fixed,
        moving,
        df,
        fixed.size(),
        moving.size(),
        block_offset,
        block_dims,
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
    int3 block_offset,
    int3 block_dims,
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

    if (x >= block_dims.x ||
        y >= block_dims.y ||
        z >= block_dims.z)
    {
        return;
    }

    int gx = x + block_offset.x;
    int gy = y + block_offset.y;
    int gz = z + block_offset.z;

    if (gx >= dims.x ||
        gy >= dims.y ||
        gz >= dims.z)
    {
        return;
    }

    // Assume spacing is 1,1,1

    // Cost ordered as f_same, f01, f10, f_same

    float3 d = {df(gx,gy,gz).x, df(gx,gy,gz).y, df(gx,gy,gz).z};
    {
        float4 o_x = {0, 0, 0, 0};
        float4 o_y = {0, 0, 0, 0};
        float4 o_z = {0, 0, 0, 0};

        float3 dx = {df(gx+1,gy,gz).x, df(gx+1,gy,gz).y, df(gx+1,gy,gz).z};
        float3 dy = {df(gx,gy+1,gz).x, df(gx,gy+1,gz).y, df(gx,gy+1,gz).z};
        float3 dz = {df(gx,gy,gz+1).x, df(gx,gy,gz+1).y, df(gx,gy,gz+1).z};

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
        cost_x(gx,gy,gz) = weight*o_x;
        cost_y(gx,gy,gz) = weight*o_y;
        cost_z(gx,gy,gz) = weight*o_z;
    }

    // TODO:
    // Compute cost at block border
    
    if (x == 0 && gx != 0) {
        float3 dx = {df(gx-1,gy,gz).x, df(gx-1,gy,gz).y, df(gx-1,gy,gz).z};
        
        float3 diff_00 = d - dx;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
        float3 diff_01 = (d+delta) - dx;
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
        cost_x(gx-1,gy,gz).x = weight*dist2_00;
        cost_x(gx-1,gy,gz).y = weight*dist2_01;
        cost_x(gx-1,gy,gz).z = weight*dist2_00; // border nodes can't move
    }
    
    if (y == 0 && gy != 0) {
        float3 dy = {df(gx,gy-1,gz).x, df(gx,gy-1,gz).y, df(gx,gy-1,gz).z};
        
        float3 diff_00 = d - dy;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
        float3 diff_01 = (d+delta) - dy;
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
        cost_y(gx,gy-1,gz).x = weight*dist2_00;
        cost_y(gx,gy-1,gz).y = weight*dist2_01;
        cost_y(gx,gy-1,gz).z = weight*dist2_00; // border nodes can't move
    }

    if (z == 0 && gz != 0) {
        float3 dz = {df(gx,gy,gz-1).x, df(gx,gy,gz-1).y, df(gx,gy,gz-1).z};
        
        float3 diff_00 = d - dz;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
        float3 diff_01 = (d+delta) - dz;
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
        cost_z(gx,gy,gz-1).x = weight*dist2_00;
        cost_z(gx,gy,gz-1).y = weight*dist2_01;
        cost_z(gx,gy,gz-1).z = weight*dist2_00; // border nodes can't move
    }

}


void gpu_compute_binary_cost(
    const stk::GpuVolume& df,
    const int3& block_offset,
    const int3& block_dims,
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
        block_offset,
        block_dims,
        delta,
        weight,
        cost_x,
        cost_y,
        cost_z
    );
    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
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
    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}
