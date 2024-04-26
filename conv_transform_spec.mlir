// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx942", ukernels = "none"}>

module attributes {transform.with_named_sequence} {

    util.func private @argmax_1d_f32_entry_point(%arg0: tensor<1x?xf32>) -> tensor<2x1x32x32x1280xf16> {
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<1x?xf32>
    // Note: This is not safe if the dim size exceeds INT32_MAX. To pass a 64
    // bit value it must be broken down into two 32-bit values for the high and
    // low bits.
    %dim_i32 = arith.index_cast %dim : index to i32
    // Inline external dispatch that conforms to the ABI that the kernel
    // requires. This is the primary reason for the surrounding function as
    // details like tensor shape and push constants need to line up after
    // splicing in the custom dispatch. This allows the kernel author to manage
    // such details by hand without needing the rewrite patterns to worry about
    // things like order of push constants.
    %4 = hal.dispatch.extern "argmax_F32I64"[%dim](%dim_i32, %arg0) : (i32, tensor<1x?xf32>{%dim}) -> tensor<1xi64>
      count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        hal.return %c1_0, %c1_0, %c1_0 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 1, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/home/nmeganat/MISA-Kernel-Integration/igemm_fwd_gtc_gfx940_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 32, workgroup_size = [32 : index, 1 : index, 1 : index]}
    util.return %4 : tensor<2x1x32x32x1280xf16>
  }

  transform.named_sequence @match_conv(
      %root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    // Fail fast on non-linalg generics.
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    transform.yield %root : !transform.any_op
  }
  
  transform.named_sequence @cast_and_call_conv(%conv: !transform.any_op {transform.readonly}) {
    %module = transform.util.get_nearest_symbol_table %conv : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point into %module if undefined : (!transform.any_op) -> !transform.any_op
    %ins = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    %outs = transform.get_result %argmax[1] : (!transform.any_op) -> !transform.any_value
    transform.util.cast_and_call %func(%ins) -> %outs before %argmax {
          // This specifies how to resolve type mismatches between the arguments
          // of the function and the inputs to the argmax. In this example, the
          // only casts this will generate are same-rank tensor casts that drop
          // static information.
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {

    // Gather the set of functions within the module.
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op   
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_conv -> @cast_and_call_conv
          : (!transform.any_op) -> (!transform.any_op)
    }
    // Cleanup now dead instances of argmax.
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}