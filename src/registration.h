#pragma once

#include <stk/image/volume.h>
#include <stk/math/types.h>

inline const int _max_iteration_count = 1;
inline const float _regularization_weight = 0.1f;
inline const int3 _block_size {16,16,16}; // TODO: Crashes for non pow2 sizes
inline const int3 _neighbors[] = {
    {1, 0, 0},
    {-1, 0, 0},
    {0, 1, 0},
    {0, -1, 0},
    {0, 0, 1},
    {0, 0, -1}
};

double calculate_energy(
    const stk::VolumeFloat& fixed, 
    const stk::VolumeFloat& moving, 
    const stk::VolumeFloat3& df
);
