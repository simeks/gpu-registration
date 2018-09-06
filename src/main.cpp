#include "profiler/profiler.h"

#include <stk/common/assert.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

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
    ASSERT(argc > 1);

    PROFILER_INIT();
    
    dim3 dims { 256, 256, 256 };

    stk::VolumeFloat fixed = make_volume(dims);
    stk::VolumeFloat moving = make_volume(dims);

    stk::VolumeFloat3 df(fixed.size(), {0,0,0});
    if (strcmp(argv[1], "cpu") == 0) {
        run_registration_cpu(fixed, moving, df);
    }
    else if (strcmp(argv[1], "gpu") == 0) {
        run_registration_gpu(fixed, moving, df);
    }
    else {
        ASSERT(false);
    }

    PROFILER_SHUTDOWN();
}