#include "registration_cpu.h"

stk::VolumeFloat3 run_registration_cpu(
    const VolumeFloat& fixed, 
    const VolumeFloat& moving
)
{
    bool done = false;
    while (!done) {
        done = true;

        size_t num_blocks_changed = 0;

        for (int use_shift = 0; use_shift < 2; ++use_shift) {
            PROFILER_SCOPE("shift", 0xFF766952);
            if (use_shift == 1 && (block_count.x * block_count.y * block_count.z) <= 1)
                continue;

            /*
                We only do shifting in the directions that requires it
            */

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
                
                #pragma omp parallel for schedule(dynamic) reduction(+:num_blocks_changed)
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

                    bool need_update = change_flags.is_block_set(block_p, use_shift == 1);
                    int n_count = 6; // Neighbors
                    for (int n = 0; n < n_count; ++n) {
                        int3 neighbor = block_p + _neighbors[n];
                        if (0 <= neighbor.x && neighbor.x < real_block_count.x &&
                            0 <= neighbor.y && neighbor.y < real_block_count.y &&
                            0 <= neighbor.z && neighbor.z < real_block_count.z) {
                            need_update = need_update || change_flags.is_block_set(neighbor, use_shift == 1);
                        }
                    }

                    if (!need_update) {
                        continue;
                    }

                    bool block_changed = false;
                    for (int n = 0; n < n_count; ++n) {
                        // delta in [mm]
                        float3 delta {
                            step_size.x * _neighbors[n].x,
                            step_size.y * _neighbors[n].y,
                            step_size.z * _neighbors[n].z
                        };

                        block_changed |= do_block(
                            unary_fn,
                            binary_fn,
                            block_p,
                            block_dims,
                            block_offset,
                            delta,
                            def
                        );
                    }

                    if (block_changed)
                        ++num_blocks_changed;

                    change_flags.set_block(block_p, block_changed, use_shift == 1);
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
}
