#include "profiler/profiler.h"
#include "registration.h"

#include <stk/common/assert.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

void run_registration_cpu(
    const stk::VolumeFloat& fixed, 
    const stk::VolumeFloat& moving,
    stk::VolumeFloat3 df
);
void run_registration_gpu(
    const stk::VolumeFloat& fixed, 
    const stk::VolumeFloat& moving,
    stk::VolumeFloat3 df
);

std::mt19937 _gen(52434);

stk::VolumeFloat make_volume(const dim3& dims)
{
    stk::VolumeFloat vol(dims);

    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        vol(x,y,z) = dis(_gen);
    }}}

    vol.set_origin({0,0,0});
    vol.set_spacing({1,1,1});

    return vol;
}
stk::VolumeFloat3 make_df(const dim3& dims)
{
    stk::VolumeFloat3 vol(dims);

    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        vol(x,y,z) = float3{0};// dis(_gen), dis(_gen), dis(_gen) };
    }}}

    vol.set_origin({0,0,0});
    vol.set_spacing({1,1,1});

    return vol;
}



int main(int argc, char* argv[])
{
    using namespace std::chrono;

    PROFILER_INIT();
    
    dim3 dims { 4, 4, 4 };

    stk::VolumeFloat fixed = make_volume(dims);
    stk::VolumeFloat moving = make_volume(dims);

    stk::VolumeFloat3 cpu_df = make_df(dims); //(fixed.size(), {0,0,0});
    stk::VolumeFloat3 gpu_df = cpu_df.clone(); //(fixed.size(), {0,0,0});

    std::cout << std::fixed;
    std::cout << std::setprecision(8);
    std::cout << "Initial Energy: " << calculate_energy(fixed, moving, cpu_df) << std::endl;

    auto cpu_start = high_resolution_clock::now();
    run_registration_cpu(fixed, moving, cpu_df);
    auto cpu_stop = high_resolution_clock::now();
    
    std::cout << "CPU Energy: " << calculate_energy(fixed, moving, cpu_df) << std::endl;

    auto gpu_start = high_resolution_clock::now();
    run_registration_gpu(fixed, moving, gpu_df);
    auto gpu_stop = high_resolution_clock::now();
    
    std::cout << "GPU Energy: " << calculate_energy(fixed, moving, gpu_df) << std::endl;

    auto cpu_elapsed = duration_cast<milliseconds>(cpu_stop - cpu_start);
    auto gpu_elapsed = duration_cast<milliseconds>(gpu_stop - gpu_start);

    std::cout << "CPU: " << cpu_elapsed.count() << " ms" << std::endl;
    std::cout << "GPU: " << gpu_elapsed.count() << " ms" << std::endl;

    PROFILER_SHUTDOWN();
}