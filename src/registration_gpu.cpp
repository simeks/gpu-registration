#include "gpu_pipeline.h"
#include "graph_cut.h"
#include "profiler/profiler.h"
#include "registration.h"

#include <stk/cuda/stream.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include <iostream>
// Pads vectorfield of type float3 to float4
stk::VolumeFloat4 df_float3_to_float4(const stk::VolumeFloat3& df)
{
    dim3 dims = df.size();
    stk::VolumeFloat4 out(dims);

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        out(x,y,z) = float4{df(x,y,z).x, df(x,y,z).y, df(x,y,z).z, 0.0f};
    }}}
    
    return out;
}
stk::VolumeFloat3 df_float4_to_float3(const stk::VolumeFloat4& df, stk::VolumeFloat3& out)
{
    dim3 dims = df.size();

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        out(x,y,z) = float3{df(x,y,z).x, df(x,y,z).y, df(x,y,z).z};
    }}}
    
    return out;
}

// returns current energy
double build_graph(
    GraphCut<double>& graph, 
    const stk::VolumeFloat2 unary_cost,
    const stk::VolumeFloat4 binary_cost_x,
    const stk::VolumeFloat4 binary_cost_y,
    const stk::VolumeFloat4 binary_cost_z
)
{
    dim3 dims = unary_cost.size();

    double current_energy = 0;

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        double f0 = unary_cost(x,y,z).x;
        double f1 = unary_cost(x,y,z).y;

        graph.add_term1(x, y, z, f0, f1);

        current_energy += f0;
    // }
    // }
    // }
    // for (int z = 0; z < (int)dims.z; ++z) {
    // for (int y = 0; y < (int)dims.y; ++y) {
    // for (int x = 0; x < (int)dims.x; ++x) {

        if (x + 1 < int(dims.x)) {
            double f_same = binary_cost_x(x,y,z).x;
            double f01 = binary_cost_x(x,y,z).y;
            double f10 = binary_cost_x(x,y,z).z;

            graph.add_term2(
                x, y, z,
                x + 1, y, z, 
                f_same, f01, f10, f_same);

            current_energy += f_same;
        }
        if (y + 1 < int(dims.y)) {
            double f_same = binary_cost_y(x,y,z).x;
            double f01 = binary_cost_y(x,y,z).y;
            double f10 = binary_cost_y(x,y,z).z;

            graph.add_term2(
                x, y, z,
                x, y + 1, z,
                f_same, f01, f10, f_same);

            current_energy += f_same;
        }
        if (z + 1 < int(dims.z)) {
            double f_same = binary_cost_z(x,y,z).x;
            double f01 = binary_cost_z(x,y,z).y;
            double f10 = binary_cost_z(x,y,z).z;

            graph.add_term2(
                x, y, z,
                x, y, z + 1,
                f_same, f01, f10, f_same);

            current_energy += f_same;
        }
    }}}

    return current_energy;
}

bool do_block(
    const stk::VolumeFloat2& unary_cost,
    const stk::VolumeFloat4& binary_cost_x,
    const stk::VolumeFloat4& binary_cost_y,
    const stk::VolumeFloat4& binary_cost_z,
    const int3& block_p,
    const int3& block_dims,
    const int3& block_offset,
    stk::VolumeUChar& labels)
{
    dim3 dims = unary_cost.size();

    GraphCut<double> graph(block_dims);

    double current_energy = 0;
    {
        PROFILER_SCOPE("build", 0xFF228844);
        
        for (int sub_z = 0; sub_z < block_dims.z; ++sub_z) {
            for (int sub_y = 0; sub_y < block_dims.y; ++sub_y) {
                for (int sub_x = 0; sub_x < block_dims.x; ++sub_x) {
                    // Global coordinates
                    int gx = block_p.x * block_dims.x - block_offset.x + sub_x;
                    int gy = block_p.y * block_dims.y - block_offset.y + sub_y;
                    int gz = block_p.z * block_dims.z - block_offset.z + sub_z;

                    // Skip voxels outside volume
                    if (gx < 0 || gx >= int(dims.x) ||
                        gy < 0 || gy >= int(dims.y) ||
                        gz < 0 || gz >= int(dims.z)) {
                        graph.add_term1(sub_x, sub_y, sub_z, 0, 0);
                        continue;
                    }

                    double f0 = unary_cost(gx, gy, gz).x;
                    double f1 = unary_cost(gx, gy, gz).y;

                    // Block borders (excl image borders) (T-weights with binary term for neighboring voxels)

                    if (sub_x == 0 && gx != 0) {
                        f0 += binary_cost_x(gx-1,gy,gz).x;
                        f1 += binary_cost_x(gx-1,gy,gz).y;
                    }
                    else if (sub_x == block_dims.x - 1 && gx < int(dims.x) - 1) {
                        f0 += binary_cost_x(gx,gy,gz).x;
                        f1 += binary_cost_x(gx,gy,gz).z;
                    }

                    if (sub_y == 0 && gy != 0) {
                        f0 += binary_cost_y(gx,gy-1,gz).x;
                        f1 += binary_cost_y(gx,gy-1,gz).y;
                    }
                    else if (sub_y == block_dims.y - 1 && gy < int(dims.y) - 1) {
                        f0 += binary_cost_y(gx,gy,gz).x;
                        f1 += binary_cost_y(gx,gy,gz).z;
                    }

                    if (sub_z == 0 && gz != 0) {
                        f0 += binary_cost_z(gx,gy,gz-1).x;
                        f1 += binary_cost_z(gx,gy,gz-1).y;
                    }
                    else if (sub_z == block_dims.z - 1 && gz < int(dims.z) - 1) {
                        f0 += binary_cost_z(gx,gy,gz).x;
                        f1 += binary_cost_z(gx,gy,gz).z;
                    }

                    graph.add_term1(sub_x, sub_y, sub_z, f0, f1);

                    current_energy += f0;

                    if (sub_x + 1 < block_dims.x && gx + 1 < int(dims.x)) {
                        double f_same = binary_cost_x(gx,gy,gz).x;
                        double f01 = binary_cost_x(gx,gy,gz).y;
                        double f10 = binary_cost_x(gx,gy,gz).z;

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x + 1, sub_y, sub_z, 
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_y + 1 < block_dims.y && gy + 1 < int(dims.y)) {
                        double f_same = binary_cost_y(gx,gy,gz).x;
                        double f01 = binary_cost_y(gx,gy,gz).y;
                        double f10 = binary_cost_y(gx,gy,gz).z;

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x, sub_y + 1, sub_z, 
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_z + 1 < block_dims.z && gz + 1 < int(dims.z)) {
                        double f_same = binary_cost_z(gx,gy,gz).x;
                        double f01 = binary_cost_z(gx,gy,gz).y;
                        double f10 = binary_cost_z(gx,gy,gz).z;

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x, sub_y, sub_z + 1, 
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                }
            }
        }
    }


    double current_emin;
    {
        PROFILER_SCOPE("minimize", 0xFF985423);
        current_emin = graph.minimize();
    }

    bool changed_flag = false;

    if (1.0 - current_emin / current_energy > 0.001f) // Accept solution
    {
        PROFILER_SCOPE("apply", 0xFF767323);
        for (int sub_z = 0; sub_z < block_dims.z; sub_z++) {
            for (int sub_y = 0; sub_y < block_dims.y; sub_y++) {
                for (int sub_x = 0; sub_x < block_dims.x; sub_x++) {
                    // Global coordinates
                    int gx = block_p.x * block_dims.x - block_offset.x + sub_x;
                    int gy = block_p.y * block_dims.y - block_offset.y + sub_y;
                    int gz = block_p.z * block_dims.z - block_offset.z + sub_z;

                    // Skip voxels outside volume
                    if (gx < 0 || gx >= int(dims.x) ||
                        gy < 0 || gy >= int(dims.y) ||
                        gz < 0 || gz >= int(dims.z))
                    {
                        continue;
                    }

                    labels(gx,gy,gz) = (uint8_t)graph.get_var(sub_x, sub_y, sub_z);
                    if (labels(gx,gy,gz) == 1)
                    {
                        changed_flag = true;
                    }
                }
            }
        }
    }
    return changed_flag;
}

void run_registration_gpu(
    const stk::VolumeFloat& fixed, 
    const stk::VolumeFloat& moving,
    stk::VolumeFloat3 initial_df
)
{
    stk::cuda::Stream stream_1;
    stk::cuda::Stream stream_2;

    dim3 dims = initial_df.size();

    stk::GpuVolume gpu_fixed(fixed);
    stk::GpuVolume gpu_moving(moving);

    stk::VolumeFloat4 df = df_float3_to_float4(initial_df);
    stk::GpuVolume gpu_df = stk::GpuVolume(df);
    
    GpuPipeline pipeline(
        gpu_fixed,
        gpu_moving,
        gpu_df
    );

    float3 step_size {0.5f, 0.5f, 0.5f};

    // Setting the block size to (0, 0, 0) will disable blocking and run the whole volume
    int3 block_dims = _block_size;
    if (block_dims.x == 0)
        block_dims.x = dims.x;
    if (block_dims.y == 0)
        block_dims.y = dims.y;
    if (block_dims.z == 0)
        block_dims.z = dims.z;

    int3 block_count {
        int(dims.x + block_dims.x - 1) / block_dims.x,
        int(dims.y + block_dims.y - 1) / block_dims.y,
        int(dims.z + block_dims.z - 1) / block_dims.z
    };
    
    int num_iterations = 0;

    bool done = false;
    while (!done) {
        PROFILER_SCOPE("iteration", 0xFF39842A);

        done = true;
        
        size_t num_blocks_changed = 0;

        for (int use_shift = 0; use_shift < 1; ++use_shift) {
            PROFILER_SCOPE("shift", 0xFF766952);
            if (use_shift == 1 && (block_count.x * block_count.y * block_count.z) <= 1)
                continue;

            // We only do shifting in the directions that requires it
            int3 block_offset{0, 0, 0};
            int3 real_block_count = block_count;
            if (use_shift == 1) {
                block_offset.x = block_count.x == 1 ? 0 : (block_dims.x / 2);
                block_offset.y = block_count.y == 1 ? 0 : (block_dims.y / 2);
                block_offset.z = block_count.z == 1 ? 0 : (block_dims.z / 2);
                
                if (block_count.x > 1) real_block_count.x += 1;
                if (block_count.y > 1) real_block_count.y += 1;
                if (block_count.z > 1) real_block_count.z += 1;
            }
            
            for (int black_or_red = 0; black_or_red < 2; black_or_red++) {
                PROFILER_SCOPE("red_black", 0xFF339955);
                int num_blocks = real_block_count.x * real_block_count.y * real_block_count.z;
                
                bool block_changed = false;
                for (int n = 0; n < 6; ++n) {
                    PROFILER_SCOPE("step", 0xFFAA6FE2);

                    // delta in [mm]
                    float3 delta {
                        step_size.x * _neighbors[n].x,
                        step_size.y * _neighbors[n].y,
                        step_size.z * _neighbors[n].z
                    };
    
                    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
                        PROFILER_SCOPE("block", 0xFFAA623D);
                        int block_x = block_idx % real_block_count.x;
                        int block_y = (block_idx / real_block_count.x) % real_block_count.y;
                        int block_z = block_idx / (real_block_count.x*real_block_count.y);

                        int off = (block_z) % 2;
                        off = (block_y + off) % 2;
                        off = (block_x + off) % 2;

                        if (off != black_or_red) {
                            continue;
                        }
                        int3 block_p{block_x, block_y, block_z};

                        pipeline.enqueue_block({block_p, block_dims, block_offset});
                    }
                    pipeline.dispatch(delta);
                }
            }
        }

        done = num_blocks_changed == 0;

        ++num_iterations;

        // A max_iteration_count of -1 means we run until we converge
        if (_max_iteration_count != -1 && num_iterations >= _max_iteration_count)
            break;
        
        PROFILER_FLIP();
    }
    gpu_df.download(df);
    df_float4_to_float3(df, initial_df);
}