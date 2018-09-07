#include "profiler/profiler.h"
#include "registration.h"

#include <stk/common/assert.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include <chrono>
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

stk::VolumeFloat make_volume(const dim3& dims)
{
    stk::VolumeFloat vol(dims);

    std::mt19937 gen(52434);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        vol(x,y,z) = dis(gen);
    }}}

    vol.set_origin({100.0f*dis(gen), 100.0f*dis(gen), 100.0f*dis(gen)});
    vol.set_spacing({1.9f, 1.9f, 2.6f});

    return vol;
}


int main(int argc, char* argv[])
{
    using namespace std::chrono;

    PROFILER_INIT();
    
    dim3 dims { 256, 256, 256 };

    stk::VolumeFloat fixed = make_volume(dims);
    stk::VolumeFloat moving = make_volume(dims);

    stk::VolumeFloat3 cpu_df(fixed.size(), {0,0,0});
    stk::VolumeFloat3 gpu_df(fixed.size(), {0,0,0});

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