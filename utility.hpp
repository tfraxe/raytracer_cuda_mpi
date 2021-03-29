#ifndef UTILITY_H
#define UTILITY_H

#include <limits>

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

float clamp(const float s, const float s_min, const float s_max) {
    if (s < s_min) return s_min;
    if (s > s_max) return s_max;
    return s;
}

#endif
