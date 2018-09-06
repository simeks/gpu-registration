#include "profiler/profiler.h"

#include <stk/common/assert.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include <random>

stk::VolumeFloat3 run_registration_cpu(
    const VolumeFloat& fixed, 
    const VolumeFloat& moving
);
stk::VolumeFloat3 run_registration_gpu(
    const VolumeFloat& fixed, 
    const VolumeFloat& moving
);

stk::VolumeFloat make_volume(const dim3& dims)
{
    stk::VolumeFloat vol(dims);

    std::mt19937 gen(52434);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int z = 0; z < dims.z; ++z) {
    for (int y = 0; y < dims.y; ++y) {
    for (int x = 0; x < dims.x; ++x) {
        vol(x,y,z) = dis(gen);
    }}}

    vol.set_origin({100.0f*dis(gen), 100.0f*dis(gen), 100.0f*dis(gen)});
    vol.set_spacing({1.9f, 1.9f, 2.6f});

    return vol;
}


int main(int argc, char* argv[])
{
    ASSERT(argc > 1);

    PROFILER_INIT();
    
    dim3 dims { 256, 256, 256 };

    stk::VolumeFloat fixed = make_volume(dims);
    stk::VolumeFloat moving = make_volume(dims);

    stk::VolumeFloat3 df;
    if (strcmp(argv[1], "cpu") == 0) {
        df = run_registration_cpu(fixed, moving);
    }
    else if (strcmp(argv[1], "gpu") == 0) {
        df = run_registration_gpu(fixed, moving);
    }
    else {
        ASSERT(false);
    }

    PROFILER_SHUTDOWN();
}