#pragma once

#include "registration.h"

#include <stk/cuda/stream.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>
#include <stk/math/types.h>

#include <deque>

void gpu_compute_unary_cost(
    const stk::GpuVolume& fixed,
    const stk::GpuVolume& moving,
    const stk::GpuVolume& df,
    const float3& delta,
    stk::GpuVolume& unary_cost, // float2
    stk::cuda::Stream& stream
);
void gpu_compute_binary_cost(
    const stk::GpuVolume& df,
    const float3& delta,
    float weight,
    stk::GpuVolume& cost_x, // float4
    stk::GpuVolume& cost_y, // float4
    stk::GpuVolume& cost_z,  // float4
    stk::cuda::Stream& stream
);

// Applies the specified delta to all voxels where label != 0
void gpu_apply_displacement_delta(
    stk::GpuVolume& df, 
    const stk::GpuVolume& labels, 
    const float3& delta,
    stk::cuda::Stream& stream
);

bool do_block(
    const stk::VolumeFloat2& unary_cost,
    const stk::VolumeFloat4& binary_cost_x,
    const stk::VolumeFloat4& binary_cost_y,
    const stk::VolumeFloat4& binary_cost_z,
    const int3& block_p,
    const int3& block_dims,
    const int3& block_offset,
    stk::VolumeUChar& labels
);


class GpuPipeline
{
public:
    struct Block
    {
        int3 p;
        int3 dims;
        int3 offset;
    };

    GpuPipeline(
        stk::GpuVolume fixed,
        stk::GpuVolume moving,
        stk::GpuVolume df
    ) :
        _fixed(fixed),
        _moving(moving),
        _df(df)
    {
        dim3 dims = _df.size();

        _unary_cost = stk::Volume(dims, stk::Type_Float2, nullptr, stk::Usage_Pinned);
        _unary_cost.fill(float2{0});
        _gpu_unary_cost = _unary_cost;

        _binary_cost_x = stk::Volume(dims, stk::Type_Float4, nullptr, stk::Usage_Pinned);
        _binary_cost_y = stk::Volume(dims, stk::Type_Float4, nullptr, stk::Usage_Pinned);
        _binary_cost_z = stk::Volume(dims, stk::Type_Float4, nullptr, stk::Usage_Pinned);
        
        _gpu_binary_cost_x = _binary_cost_x;
        _gpu_binary_cost_y = _binary_cost_y;
        _gpu_binary_cost_z = _binary_cost_z;

        _labels = stk::Volume(dims, stk::Type_UChar, nullptr, stk::Usage_Pinned);
        _gpu_labels = _labels;
    }
    ~GpuPipeline()
    {
    }

    void enqueue_block(const Block& block)
    {
        _gpu_queue.push_back(block);
    }

    void print(const stk::GpuVolume& vol)
    {
        stk::VolumeFloat cvol = vol.download();
        for (int z = 0; z < (int)cvol.size().z; ++z) {
        for (int y = 0; y < (int)cvol.size().y; ++y) {
        for (int x = 0; x < (int)cvol.size().x; ++x) {
            printf("%f, ", cvol(x,y,z));

        }
        printf("\n");
        }
        printf("\n");
        }

    }

    void print2(const stk::GpuVolume& vol)
    {
        stk::VolumeFloat2 cvol = vol.download();
        for (int z = 0; z < (int)cvol.size().z; ++z) {
        for (int y = 0; y < (int)cvol.size().y; ++y) {
        for (int x = 0; x < (int)cvol.size().x; ++x) {
            printf("(%f, %f), ", cvol(x,y,z).x, cvol(x,y,z).y);

        }
        printf("\n");
        }
        printf("\n");
        }

    }
    void print4(const stk::GpuVolume& vol)
    {
        stk::VolumeFloat4 cvol = vol.download();
        for (int z = 0; z < (int)cvol.size().z; ++z) {
        for (int y = 0; y < (int)cvol.size().y; ++y) {
        for (int x = 0; x < (int)cvol.size().x; ++x) {
            printf("(%f, %f, %f, %f), ", cvol(x,y,z).x, cvol(x,y,z).y, cvol(x,y,z).z, cvol(x,y,z).w);

        }
        printf("\n");
        }
        printf("\n");
        }

    }

    void dispatch(const float3& delta)
    {
        _labels.fill(0);
        
        while(!_gpu_queue.empty()) {
            Block block = _gpu_queue.front();
            _gpu_queue.pop_front();

            int gx = block.p.x * block.dims.x - block.offset.x;
            int gy = block.p.y * block.dims.y - block.offset.y;
            int gz = block.p.z * block.dims.z - block.offset.z;

            stk::GpuVolume sub_f(_fixed, 
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );
            print(sub_f);
            print(_fixed);

            stk::GpuVolume sub_m(_moving, 
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );

            stk::GpuVolume sub_df(_df, 
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );

            stk::GpuVolume sub_unary_gpu(_gpu_unary_cost,
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );

            gpu_compute_unary_cost(
                sub_f,
                sub_m,
                sub_df,
                delta,
                sub_unary_gpu,
                _streams[0]
            );
            
            stk::VolumeFloat2 sub_unary(_unary_cost,
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );
            sub_unary_gpu.download(sub_unary, _streams[0]);
            _gpu_unary_cost.download(_unary_cost, _streams[0]);

            stk::GpuVolume sub_binary_x_gpu(_gpu_binary_cost_x,
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );
            stk::GpuVolume sub_binary_y_gpu(_gpu_binary_cost_y,
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );

            stk::GpuVolume sub_binary_z_gpu(_gpu_binary_cost_z,
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );

            gpu_compute_binary_cost(
                sub_df,
                delta,
                _regularization_weight,
                sub_binary_x_gpu,
                sub_binary_y_gpu,
                sub_binary_z_gpu,
                _streams[0]
            );
            
            stk::VolumeFloat4 sub_binary_x(_binary_cost_x,
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );
            sub_binary_x_gpu.download(sub_binary_x, _streams[0]);

            stk::VolumeFloat4 sub_binary_y(_binary_cost_y,
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );
            sub_binary_y_gpu.download(sub_binary_y, _streams[0]);
            
            stk::VolumeFloat4 sub_binary_z(_binary_cost_z,
                { gx, gx + block.dims.x },
                { gy, gy + block.dims.y },
                { gz, gz + block.dims.z }
            );
            sub_binary_z_gpu.download(sub_binary_z, _streams[0]);

            _streams[0].synchronize();

            _cpu_queue.push_back(block);
        }

        while(!_cpu_queue.empty()) {
            Block block = _cpu_queue.front();
            _cpu_queue.pop_front();

            do_block(
                _unary_cost,
                _binary_cost_x,
                _binary_cost_y,
                _binary_cost_z,
                block.p,
                block.dims,
                block.offset,
                _labels
            );
        }
        
        _gpu_labels.upload(_labels);

        gpu_apply_displacement_delta(_df, _gpu_labels, delta, stk::cuda::Stream::null());
    }

    stk::GpuVolume _fixed;
    stk::GpuVolume _moving;
    stk::GpuVolume _df;
    
    stk::VolumeFloat2 _unary_cost;
    stk::GpuVolume _gpu_unary_cost;
    
    stk::VolumeFloat4 _binary_cost_x;
    stk::VolumeFloat4 _binary_cost_y;
    stk::VolumeFloat4 _binary_cost_z;

    stk::GpuVolume _gpu_binary_cost_x;
    stk::GpuVolume _gpu_binary_cost_y;
    stk::GpuVolume _gpu_binary_cost_z;

    stk::VolumeUChar _labels;
    stk::GpuVolume _gpu_labels;

    stk::cuda::Stream _streams[4];

    std::deque<Block> _gpu_queue;
    std::deque<Block> _cpu_queue;
};
