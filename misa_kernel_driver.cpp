#include "half.hpp"
#include "utils.h"
#include <chrono>
#include <fstream>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using float16 = half_float::half;

// Datatype defined as in MISA for convolution kernel parameters
typedef struct {
  void *p_in;
  void *p_wei;
  void *p_out;
  int hi;
  int wi;
  int n;
  int k; // this is indeed k_per_group
  int c; // this is indeed c_per_group
  int ho;
  int wo;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int pad_h;
  int pad_w;
  int y;
  int x;
  int group;
  uint32_t magic_0; // denom: n*b / n_per_block
  uint32_t magic_1; // denom: ((n / nb_n0) * b) / nb_n1b
  uint32_t magic_2; // denom: y*x, if nxe==0 not used
  uint32_t magic_3; // denom: x, if nxe==0 not used
  uint32_t magic_4; // denom: b
  uint32_t magic_5; // denom: wo
  uint32_t magic_6; // denom: n*b*k / (m_per_block*n_per_block)
  uint32_t shift_pack_0;
  uint32_t shift_pack_1;
  uint32_t ks;
} __attribute__((packed)) igemm_fwd_gtc_karg_t;

std::vector<char> readFileIntoVector(const char *filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return std::vector<char>();
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  file.close();
  return buffer;
}

hipFunction_t loadKernel(const char *filePath, const char *kernelName) {
  hipModule_t module;
  hipFunction_t kernelFunc;
  std::vector<char> hsacoVec = readFileIntoVector(filePath);
  if (hipModuleLoadDataEx(&module, hsacoVec.data(), 0, NULL, NULL) !=
      hipSuccess) {
    std::cout << "Failed to load module!\n";
    return NULL;
  }
  if (hipModuleGetFunction(&kernelFunc, module, kernelName) != hipSuccess) {
    std::cout << "Failed to get function!\n";
    return NULL;
  }
  return kernelFunc;
}

igemm_fwd_gtc_karg_t initializeKernelArgs() {

  igemm_fwd_gtc_karg_t karg;
  karg.hi = 34;
  karg.wi = 34;
  karg.n = 2;
  karg.k = 1280;
  karg.c = 1280;
  karg.ho = 32;
  karg.wo = 32;
  karg.stride_h = 1;
  karg.stride_w = 1;
  karg.dilation_h = 1;
  karg.dilation_w = 1;
  karg.pad_h = 0;
  karg.pad_w = 0;
  karg.y = 3;
  karg.x = 3;
  karg.group = 1;
  karg.magic_0 = 2576980378;
  karg.magic_1 = 1;
  karg.magic_2 = 1;
  karg.magic_3 = 2576980378;
  karg.magic_4 = 0;
  karg.magic_5 = 0;
  karg.shift_pack_0 = 117770756;
  karg.shift_pack_1 = 828006248;
  karg.ks = 1919907636;

  std::vector<float16> h_in, h_out, h_wei;
  h_in = generateFloatVector(karg.n * karg.hi * karg.wi * karg.k);
  h_wei = generateFloatVector(karg.x * karg.y * karg.k * karg.c);
  h_out = generateFloatVector(karg.n * karg.ho * karg.wo * karg.c);

  float16 *d_in, *d_wei, *d_out;
  const size_t bytesInput = h_in.size() * sizeof(float16);
  const size_t bytesWeight = h_wei.size() * sizeof(float16);
  const size_t bytesOutput = h_out.size() * sizeof(float16);

  CHECK_HIP_ERROR(hipMalloc(&d_in, bytesInput));
  CHECK_HIP_ERROR(hipMalloc(&d_out, bytesOutput));
  CHECK_HIP_ERROR(hipMalloc(&d_wei, bytesWeight));

  CHECK_HIP_ERROR(
      hipMemcpy(d_in, h_in.data(), bytesInput, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(
      hipMemcpy(d_wei, h_wei.data(), bytesWeight, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(
      hipMemcpy(d_out, h_out.data(), bytesOutput, hipMemcpyHostToDevice));

  karg.p_in = (void *)d_in;
  karg.p_wei = (void *)d_wei;
  karg.p_out = (void *)d_out;

  return karg;
}

void runKernel(hipFunction_t kernelFunc, void *args, size_t arg_size,
               std::vector<size_t> grid_size, std::vector<size_t> block_size) {

  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                    HIP_LAUNCH_PARAM_END};
  hipEvent_t start;
  hipEvent_t stop;
  CHECK_HIP_ERROR(hipEventCreate(&start));
  CHECK_HIP_ERROR(hipEventCreate(&stop));
  float ms = .0;

  CHECK_HIP_ERROR(hipExtModuleLaunchKernel(
      kernelFunc, grid_size[0], grid_size[1], grid_size[2], block_size[0],
      block_size[1], block_size[2], 0, 0, NULL, (void **)&config, start, stop));

  CHECK_HIP_ERROR(hipEventSynchronize(stop));
  CHECK_HIP_ERROR(hipEventElapsedTime(&ms, start, stop));
  CHECK_HIP_ERROR(hipEventDestroy(start));
  CHECK_HIP_ERROR(hipEventDestroy(stop));
}

int main() {
  // MISA convolution kernel object code compiled for gfx940 
  const char *hsaco_file = "igemm_fwd_gtc_gfx940_nhwc_fp16.hsaco";
  const char *kernel_name =
      "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_"
      "ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64";
  hipFunction_t kernel = loadKernel(hsaco_file, kernel_name);
  if (!kernel)
    exit(EXIT_FAILURE);
  igemm_fwd_gtc_karg_t karg = initializeKernelArgs();
  size_t arg_size = 128;
  runKernel(kernel, (void *)&karg, arg_size, {20480, 1, 1}, {256, 1, 1});
  return 0;
}