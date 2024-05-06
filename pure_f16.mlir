#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module attributes {torch.debug_module_name = "Conv2d"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<2x1280x34x34xf16>) -> tensor<2x1280x32x32xf16> {
    %cst = util.unfoldable_constant dense<0.0> : tensor<1x1280xf16>
    %cst_0 = util.unfoldable_constant dense<1.0> : tensor<1280x1280x3x3xf16>
    // %cst = arith.constant dense_resource<__elided__> : tensor<1x1280xf16>
    // %cst_0 = arith.constant dense_resource<__elided__> : tensor<1280x1280x3x3xf16>
    %0 = tensor.empty() : tensor<1x1280xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<1x1280xf16>) outs(%0 : tensor<1x1280xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x1280xf16>
    %2 = tensor.empty() : tensor<2x1x32x32x1280xf16>
    %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x1280xf16>) outs(%2 : tensor<2x1x32x32x1280xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<2x1x32x32x1280xf16>
    %4 = tensor.empty() : tensor<2x34x34x1280xf16>
    %transposed = linalg.transpose ins(%arg0 : tensor<2x1280x34x34xf16>) outs(%4 : tensor<2x34x34x1280xf16>) permutation = [0, 2, 3, 1] 
    %5 = tensor.empty() : tensor<3x3x1280x1280xf16>
    %transposed_1 = linalg.transpose ins(%cst_0 : tensor<1280x1280x3x3xf16>) outs(%5 : tensor<3x3x1280x1280xf16>) permutation = [2, 3, 1, 0] 
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2], [3], [4]] : tensor<2x1x32x32x1280xf16> into tensor<2x32x32x1280xf16>
    %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%transposed, %transposed_1 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%collapsed : tensor<2x32x32x1280xf16>) -> tensor<2x32x32x1280xf16>
    %9 = tensor.empty() : tensor<2x1280x32x32xf16>
    %transposed_3 = linalg.transpose ins(%6 : tensor<2x32x32x1280xf16>) outs(%9 : tensor<2x1280x32x32xf16>) permutation = [0, 3, 1, 2] 
    return %transposed_3 : tensor<2x1280x32x32xf16>
  }
}