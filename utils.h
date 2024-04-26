#pragma once

#include <cstdlib>
#include <vector>

#include "half.hpp"

using float16 = half_float::half;

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                                                \
  if (status != hipSuccess) {                                                  \
    fprintf(stderr, "hip error: '%s'(%d) at %s:%d\n",                          \
            hipGetErrorString(status), status, __FILE__, __LINE__);            \
    exit(EXIT_FAILURE);                                                        \
  }
#endif

std::vector<float16> generateFloatVector(int vecSize) {
  std::vector<float16> vector_container;
  float16 r;
  for (int i = 0; i < vecSize; i++) {
    r = static_cast<float16>(rand()) / static_cast<float16>(RAND_MAX);
    vector_container.push_back(r);
  }

  return vector_container;
}