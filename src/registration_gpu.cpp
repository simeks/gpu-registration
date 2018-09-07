#include "graph_cut.h"
#include "profiler/profiler.h"
#include "registration.h"

#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include <iostream>

void gpu_compute_unary_cost(
    const stk::GpuVolume& fixed,
    const stk::GpuVolume& moving,
    const stk::GpuVolume& df,
    const float3& delta,
    stk::GpuVolume& unary_cost // float2
);
void gpu_compute_binary_cost(
    const stk::GpuVolume& df,
    const float3& delta,
    float weight,
    stk::GpuVolume& cost_x, // float4
    stk::GpuVolume& cost_y, // float4
    stk::GpuVolume& cost_z  // float4
);

// Applies the specified delta to all voxels where label != 0
void gpu_apply_displacement_delta(
    stk::GpuVolume& df, 
    const stk::GpuVolume& labels, 
    const float3& delta
);

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
        int3 p {x,y,z};

        double f0 = unary_cost(x,y,z).x;
        double f1 = unary_cost(x,y,z).y;

        graph.add_term1(x, y, z, f0, f1);

        current_energy += f0;

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

void run_registration_gpu(
    const stk::VolumeFloat& fixed, 
    const stk::VolumeFloat& moving,
    stk::VolumeFloat3 initial_df
)
{
    dim3 dims = initial_df.size();

    stk::GpuVolume gpu_fixed(fixed);
    stk::GpuVolume gpu_moving(moving);

    stk::VolumeFloat4 df = df_float3_to_float4(initial_df);
    stk::GpuVolume gpu_df = stk::GpuVolume(df);

    stk::VolumeFloat2 unary_cost(dims, 0);
    stk::GpuVolume gpu_unary_cost(unary_cost);

    stk::VolumeFloat4 binary_cost_x(dims);
    stk::VolumeFloat4 binary_cost_y(dims);
    stk::VolumeFloat4 binary_cost_z(dims);
    
    stk::GpuVolume gpu_binary_cost_x(binary_cost_x);
    stk::GpuVolume gpu_binary_cost_y(binary_cost_y);
    stk::GpuVolume gpu_binary_cost_z(binary_cost_z);
    
    // Movement labels, updated per step
    stk::VolumeUChar labels(dims, (uint8_t)0);
    stk::GpuVolume gpu_labels(labels);

    float3 step_size {0.5f, 0.5f, 0.5f};

    int num_iterations = 0;

    bool done = false;
    while (!done) {
        done = true;

        size_t num_blocks_changed = 0;

        bool block_changed = false;
        for (int n = 0; n < 6; ++n) {
            // delta in [mm]
            float3 delta {
                step_size.x * _neighbors[n].x,
                step_size.y * _neighbors[n].y,
                step_size.z * _neighbors[n].z
            };

            gpu_compute_unary_cost(
                gpu_fixed,
                gpu_moving,
                gpu_df,
                delta,
                gpu_unary_cost
            );
            gpu_unary_cost.download(unary_cost);

            gpu_compute_binary_cost(
                gpu_df,
                delta,
                _regularization_weight,
                gpu_binary_cost_x,
                gpu_binary_cost_y,
                gpu_binary_cost_z
            );
            gpu_binary_cost_x.download(binary_cost_x);
            gpu_binary_cost_y.download(binary_cost_y);
            gpu_binary_cost_z.download(binary_cost_z);


            GraphCut<double> graph({(int)dims.x, (int)dims.y, (int)dims.z});
            double current_energy = build_graph(
                graph, 
                unary_cost,
                binary_cost_x,
                binary_cost_y,
                binary_cost_z
            );
            double current_emin = 0;
            {
                PROFILER_SCOPE("minimize", 0xFF985423);
                current_emin = graph.minimize();
            }

            bool changed_flag = false;

            if (1.0 - current_emin / current_energy > 0.000001f) // Accept solution
            {
                PROFILER_SCOPE("apply", 0xFF767323);
                for (int z = 0; z < (int)dims.z; ++z) {
                for (int y = 0; y < (int)dims.y; ++y) {
                for (int x = 0; x < (int)dims.x; ++x) {
                    labels(x,y,z) = (uint8_t)graph.get_var(x, y, z);
                }}}
            }
            gpu_labels.upload(labels);

            gpu_apply_displacement_delta(gpu_df, gpu_labels, delta);
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