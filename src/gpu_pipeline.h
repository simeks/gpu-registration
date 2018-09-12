#pragma once

#include "profiler/profiler.h"
#include "registration.h"

#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>
#include <stk/math/types.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

void gpu_compute_unary_cost(
    const stk::GpuVolume& fixed,
    const stk::GpuVolume& moving,
    const stk::GpuVolume& df,
    const int3& block_offset,
    const int3& block_dims,
    const float3& delta,
    stk::GpuVolume& unary_cost, // float2
    stk::cuda::Stream& stream
);
void gpu_compute_binary_cost(
    const stk::GpuVolume& df,
    const int3& block_offset,
    const int3& block_dims,
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

void stream_callback(cudaStream_t , cudaError_t , void* user_data);
void worker_thread(class GpuPipeline* pipeline);

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

        _stop = false;
        for (int i = 0; i < 8; ++i) {
            _threads[i] = std::thread(worker_thread, this);
        }

    }
    ~GpuPipeline()
    {
        _stop = true;
        for (int i = 0; i < 8; ++i) {
            _threads[i].join();
        }
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

    void download_subvolume(
        const stk::GpuVolume& src, 
        stk::Volume& tgt, 
        const int3& offset,
        const int3& dims,
        bool pad, // Pad all axes by 1 in negative direction for binary cost
        stk::cuda::Stream& stream)
    {
        int3 padded_offset = offset;
        int3 padded_dims = dims;

        if (pad) {
            if (offset.x > 0) {
                padded_offset.x -= 1;
                padded_dims.x += 1;
            }
            if (offset.y > 0) {
                padded_offset.y -= 1;
                padded_dims.y += 1;
            }
            if (offset.z > 0) {
                padded_offset.z -= 1;
                padded_dims.z += 1;
            }
        }

        stk::GpuVolume sub_src(src,
            { padded_offset.x, padded_offset.x + padded_dims.x },
            { padded_offset.y, padded_offset.y + padded_dims.y },
            { padded_offset.z, padded_offset.z + padded_dims.z }
        );

        stk::Volume sub_tgt(tgt,
            { padded_offset.x, padded_offset.x + padded_dims.x },
            { padded_offset.y, padded_offset.y + padded_dims.y },
            { padded_offset.z, padded_offset.z + padded_dims.z }
        );

        sub_src.download(sub_tgt, stream);
    }

    void dispatch(const float3& delta)
    {
        _blocks_remaining = _gpu_queue.size();

        _labels.fill(0);
        
        int si = 0;
        {
            PROFILER_SCOPE("gpucost", 0xFF492343);
            while(!_gpu_queue.empty()) {
                Block block = _gpu_queue.front();
                _gpu_queue.pop_front();

                int gx = block.p.x * block.dims.x - block.offset.x;
                int gy = block.p.y * block.dims.y - block.offset.y;
                int gz = block.p.z * block.dims.z - block.offset.z;

                _streams[si].stream.synchronize();

                gpu_compute_unary_cost(
                    _fixed,
                    _moving,
                    _df,
                    {gx, gy, gz},
                    block.dims,
                    delta,
                    _gpu_unary_cost,
                    _streams[si].stream
                );
                
                download_subvolume(
                    _gpu_unary_cost, 
                    _unary_cost,
                    {gx,gy,gz},
                    block.dims,
                    false,
                    _streams[si].stream
                );

                gpu_compute_binary_cost(
                    _df,
                    {gx, gy, gz},
                    block.dims,
                    delta,
                    _regularization_weight,
                    _gpu_binary_cost_x,
                    _gpu_binary_cost_y,
                    _gpu_binary_cost_z,
                    _streams[si].stream
                );
                
                download_subvolume(
                    _gpu_binary_cost_x, 
                    _binary_cost_x,
                    {gx,gy,gz},
                    block.dims,
                    true,
                    _streams[si].stream
                );
                download_subvolume(
                    _gpu_binary_cost_y, 
                    _binary_cost_y,
                    {gx,gy,gz},
                    block.dims,
                    true,
                    _streams[si].stream
                );
                download_subvolume(
                    _gpu_binary_cost_z, 
                    _binary_cost_z,
                    int3{gx,gy,gz},
                    block.dims,
                    true,
                    _streams[si].stream
                );

                _streams[si].pipe = this;
                _streams[si].block = block;
                _streams[si].stream.add_callback(stream_callback, &_streams[si]);

                si = (si + 1) % 4;
            }
            for (si = 0; si < 4; ++si)
                _streams[si].stream.synchronize();
        }

        while (_blocks_remaining > 0) {
            std::this_thread::yield();
        }

        // for (int i = 0; i < 8; ++i) {
        //     _threads[i].join();
        // }

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

    struct Stream {
        class GpuPipeline* pipe;
        stk::cuda::Stream stream;

        Block block;

        Stream() : pipe(nullptr) {}
    };
    Stream _streams[4];

    std::thread _threads[8];

    std::deque<Block> _gpu_queue;
    std::atomic<uint32_t> _blocks_remaining;
    std::atomic<bool> _stop;

    std::mutex _queue_lock;
    std::deque<Block> _cpu_queue;

};

void stream_callback(cudaStream_t , cudaError_t , void* user_data)
{
    GpuPipeline::Stream* stream = static_cast<GpuPipeline::Stream*>(user_data);
    stream->pipe->_queue_lock.lock();
    stream->pipe->_cpu_queue.push_back(stream->block);
    stream->pipe->_queue_lock.unlock();
}
void worker_thread(class GpuPipeline* pipeline)
{
    while (!pipeline->_stop) {
        GpuPipeline::Block b{0};
        b.dims = {-1,-1,-1};

        pipeline->_queue_lock.lock();
        if (!pipeline->_cpu_queue.empty()) {
            b = pipeline->_cpu_queue.front();
            pipeline->_cpu_queue.pop_front();
        }
        pipeline->_queue_lock.unlock();

        if (b.dims.x != -1) {
            do_block(
                pipeline->_unary_cost,
                pipeline->_binary_cost_x,
                pipeline->_binary_cost_y,
                pipeline->_binary_cost_z,
                b.p,
                b.dims,
                b.offset,
                pipeline->_labels
            );
            --pipeline->_blocks_remaining;
        }
        else {
            std::this_thread::yield();
        }
    }
}
