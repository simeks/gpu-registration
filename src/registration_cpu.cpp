#include "block_change_flags.h"
#include "graph_cut.h"
#include "profiler/profiler.h"
#include "registration.h"

#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include <iostream>


struct Unary_SSD
{
    Unary_SSD() {}
    Unary_SSD(const stk::VolumeFloat& fixed, const stk::VolumeFloat& moving) :
        _fixed(fixed), _moving(moving)
    {
    }

    inline double operator()(const int3& p, const float3& d)
    {
        float3 fixed_p{
            float(p.x),
            float(p.y),
            float(p.z)
        };
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p + d - _moving.origin()) / _moving.spacing();

        float moving_v = _moving.linear_at(moving_p, stk::Border_Constant);

        float f = float(_fixed(p) - moving_v);
        return f*f;
    }

    stk::VolumeFloat _fixed;
    stk::VolumeFloat _moving;
};
struct Unary_NCC
{
    Unary_NCC() {}
    Unary_NCC(const stk::VolumeFloat& fixed, const stk::VolumeFloat& moving) :
        _fixed(fixed), _moving(moving)
    {
    }

    inline double operator()(const int3& p, const float3& d)
    {
        int _radius = 2;

        float3 fixed_p{
            float(p.x),
            float(p.y),
            float(p.z)
        }; 
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p + d - _moving.origin()) / _moving.spacing();

        double sff = 0.0;
        double smm = 0.0;
        double sfm = 0.0;
        double sf = 0.0;
        double sm = 0.0;
        size_t n = 0;

        for (int dz = -_radius; dz <= _radius; ++dz) {
            for (int dy = -_radius; dy <= _radius; ++dy) {
                for (int dx = -_radius; dx <= _radius; ++dx) {
                    // TODO: Does not account for anisotropic volumes
                    int r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 > _radius * _radius)
                        continue;
                    int3 fp{p.x + dx, p.y + dy, p.z + dz};
                    
                    if (!stk::is_inside(_fixed.size(), fp))
                        continue;


                    float3 mp{moving_p.x + dx, moving_p.y + dy, moving_p.z + dz};

                    float fixed_v = _fixed(fp);
                    float moving_v = _moving.linear_at(mp, stk::Border_Constant);

                    sff += fixed_v * fixed_v;
                    smm += moving_v * moving_v;
                    sfm += fixed_v*moving_v;
                    sm += moving_v;
                    sf += fixed_v;

                    ++n;
                }
            }
        }

        if (n == 0)
            return 0.0f;

        // Subtract mean
        sff -= (sf * sf / n);
        smm -= (sm * sm / n);
        sfm -= (sf * sm / n);
        
        double denom = sqrt(sff*smm);

        if(denom > 1e-14) {
            return 0.5f*(1.0f-float(sfm / denom));
        }
        return 0.0f;
    }

    stk::VolumeFloat _fixed;
    stk::VolumeFloat _moving;
};


struct Binary_Regularizer
{
    Binary_Regularizer() {}
    Binary_Regularizer(float3 spacing, float weight) : _spacing(spacing), _weight(weight) {}

    inline double operator()(const int3& /*p*/, const float3& d0, const float3& d1, const int3& step)
    {
        float3 step_in_mm {
            step.x*_spacing.x, 
            step.y*_spacing.y, 
            step.z*_spacing.z
        };
        
        float3 diff = d0 - d1;
        
        float dist_squared = stk::norm2(diff);
        float step_squared = stk::norm2(step_in_mm);
        
        return _weight * dist_squared / step_squared;
    }
    
    float3 _spacing;
    float _weight;
};

Unary_NCC make_unary_fn(const stk::VolumeFloat& fixed, const stk::VolumeFloat& moving)
{
    return Unary_NCC(fixed, moving);
}
Binary_Regularizer make_binary_fn()
{
    return Binary_Regularizer(float3{1,1,1}, _regularization_weight);
}


template<typename UnaryFn, typename BinaryFn>
double calculate_energy_(UnaryFn& unary_fn, BinaryFn& binary_fn, stk::VolumeFloat3 df)
{
    dim3 dims = df.size();

    double total_energy = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:total_energy)
    for (int gz = 0; gz < int(dims.z); ++gz) {
        for (int gy = 0; gy < int(dims.y); ++gy) {
            for (int gx = 0; gx < int(dims.x); ++gx) {
                int3 p{gx, gy, gz};
                float3 d1 = df(p);

                total_energy += unary_fn(p, d1);

                if (gx + 1 < int(dims.x)) {
                    int3 step{1, 0, 0};
                    float3 d2 = df(p + step);
                    total_energy += binary_fn(p, d1, d2, step);
                }
                if (gy + 1 < int(dims.y)) {
                    int3 step{0, 1, 0};
                    float3 d2 = df(p + step);
                    total_energy += binary_fn(p, d1, d2, step);
                }
                if (gz + 1 < int(dims.z)) {
                    int3 step{0, 0, 1};
                    float3 d2 = df(p + step);
                    total_energy += binary_fn(p, d1, d2, step);
                }
            }
        }
    }
    return total_energy;
}
double calculate_energy(
    const stk::VolumeFloat& fixed, 
    const stk::VolumeFloat& moving, 
    const stk::VolumeFloat3& df
)
{
    auto unary_fn = make_unary_fn(fixed, moving);
    Binary_Regularizer binary_fn = make_binary_fn();

    return calculate_energy_(unary_fn, binary_fn, df);
}

template<typename UnaryFn, typename BinaryFn>
bool do_block(
    UnaryFn& unary_fn,
    BinaryFn& binary_fn,
    const int3& block_p,
    const int3& block_dims,
    const int3& block_offset,
    const float3& delta, // delta in mm
    stk::VolumeFloat3& df
)
{
    dim3 dims = df.size();

    typedef double FlowType;

    GraphCut<FlowType> graph(block_dims);

    FlowType current_energy = 0;
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

                    int3 p{gx, gy, gz};
                    float3 def1 = df(p);
            
                    double f0 = unary_fn(p, def1);
                    double f1 = unary_fn(p, def1 + delta);

                    // Block borders (excl image borders) (T-weights with binary term for neighboring voxels)

                    if (sub_x == 0 && gx != 0) {
                        int3 step{-1, 0, 0};
                        float3 def2 = df(gx - 1, gy, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_x == block_dims.x - 1 && gx < int(dims.x) - 1) {
                        int3 step{1, 0, 0};
                        float3 def2 = df(gx + 1, gy, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    if (sub_y == 0 && gy != 0) {
                        int3 step{0, -1, 0};
                        float3 def2 = df(gx, gy - 1, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_y == block_dims.y - 1 && gy < int(dims.y) - 1) {
                        int3 step{0, 1, 0};
                        float3 def2 = df(gx, gy + 1, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    if (sub_z == 0 && gz != 0) {
                        int3 step{0, 0, -1};
                        float3 def2 = df(gx, gy, gz - 1);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_z == block_dims.z - 1 && gz < int(dims.z) - 1) {
                        int3 step{0, 0, 1};
                        float3 def2 = df(gx, gy, gz + 1);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    graph.add_term1(sub_x, sub_y, sub_z, f0, f1);

                    current_energy += f0;

                    if (sub_x + 1 < block_dims.x && gx + 1 < int(dims.x)) {
                        int3 step{1, 0, 0};
                        float3 def2 = df(p + step);
                        double f_same = binary_fn(p, def1, def2, step);
                        double f01 = binary_fn(p, def1, def2 + delta, step);
                        double f10 = binary_fn(p, def1 + delta, def2, step);

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x + 1, sub_y, sub_z, 
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_y + 1 < block_dims.y && gy + 1 < int(dims.y)) {
                        int3 step{0, 1, 0};
                        float3 def2 = df(p + step);
                        double f_same = binary_fn(p, def1, def2, step);
                        double f01 = binary_fn(p, def1, def2 + delta, step);
                        double f10 = binary_fn(p, def1 + delta, def2, step);

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x, sub_y + 1, sub_z,
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_z + 1 < block_dims.z && gz + 1 < int(dims.z)) {
                        int3 step{0, 0, 1};
                        float3 def2 = df(p + step);
                        double f_same = binary_fn(p, def1, def2, step);
                        double f01 = binary_fn(p, def1, def2 + delta, step);
                        double f10 = binary_fn(p, def1 + delta, def2, step);

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


    FlowType current_emin;
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

                    if (graph.get_var(sub_x, sub_y, sub_z) == 1)
                    {
                        df(gx, gy, gz) = df(gx, gy, gz) + delta;
                        changed_flag = true;
                    }
                }
            }
        }
    }
    return changed_flag;
}

void run_registration_cpu(
    const stk::VolumeFloat& fixed, 
    const stk::VolumeFloat& moving,
    stk::VolumeFloat3 df
)
{
    auto unary_fn = make_unary_fn(fixed, moving);
    Binary_Regularizer binary_fn = make_binary_fn();

    dim3 dims = df.size();

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

        for (int use_shift = 0; use_shift < 2; ++use_shift) {
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

                        block_changed |= do_block(
                            unary_fn,
                            binary_fn,
                            block_p,
                            block_dims,
                            block_offset,
                            delta,
                            df
                        );

                        if (block_changed)
                            ++num_blocks_changed;
                    }
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
