// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx942", ukernels = "none"}>

module attributes {transform.with_named_sequence} {

    util.func private @conv_entry_point(%arg0: tensor<2x34x34x1280xf16>, %arg1: tensor<3x3x1280x1280xf16>) 
                                          -> tensor<2x32x32x1280xf16> {
    %hi = arith.constant 34 : i32
    %wi = arith.constant 34 : i32
    %n = arith.constant 2 : i32
    %k = arith.constant 1280 : i32
    %c = arith.constant 1280 : i32
    %ho = arith.constant 32 : i32
    %wo = arith.constant 32 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    // TODO: figure out how to use ui32
    // %magic_0 = arith.constant 2576980378 : i32
    %magic_0 = arith.constant 2576980378 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 2576980378 : i32
    %magic_4 = arith.constant 0 : i32
    %magic_5 = arith.constant 0 : i32
    %shift_pack_0 = arith.constant 117770756 : i32
    %shift_pack_1 = arith.constant 828006248 : i32
    %ks = arith.constant 1919907636 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64"(%hi, %wi, %n,
        %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group, 
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks,
        %arg0, %arg1) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        i32, i32, i32, i32, i32, i32, i32, i32, tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) -> tensor<2x32x32x1280xf16>
      count(%device: !hal.device) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        %c80_0 = arith.constant 80 : index
        hal.return %c80_0, %c1_0, %c1_0 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 25, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/home/nmeganat/MISA-Kernel-Integration/igemm_fwd_gtc_gfx940_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    // %4 = tensor.empty() : tensor<2x32x32x1280xf16>
    util.return %5 : tensor<2x32x32x1280xf16>
  }

  transform.named_sequence @match_conv(
      %root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    // transform.print %root: !transform.any_op
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %matched = transform.match.structured failures(propagate) %root : (!transform.any_op) -> (!transform.any_op) {
    ^bb1(%conv_nhwc: !transform.any_op):
      // TODO: Verify rank of the operands
      // Verify two inputs (input and weight tensor) and single output
      %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      %n_inputs = transform.match.structured.num_inputs %conv_nhwc : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %n_inputs, %c2 : !transform.param<i64>
      %n_outputs = transform.match.structured.num_inits %conv_nhwc : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %n_outputs, %c1 : !transform.param<i64>
  
      transform.match.structured.yield %conv_nhwc : !transform.any_op 
    }
    // Verify the operand shapes of the conv op.
    // Define input and filter tensor shapes for verification.
    %c_in = transform.param.constant 2 : i64 -> !transform.param<i64>
    %c_ic = transform.param.constant 1280 : i64 -> !transform.param<i64>
    %c_ih = transform.param.constant 34 : i64 -> !transform.param<i64>
    %c_iw = transform.param.constant 34 : i64 -> !transform.param<i64>
    %c_fh = transform.param.constant 3 : i64 -> !transform.param<i64>
    %c_fw = transform.param.constant 3 : i64 -> !transform.param<i64>
    %c_fc = transform.param.constant 1280 : i64 -> !transform.param<i64>
    %c_ff = transform.param.constant 1280 : i64 -> !transform.param<i64>
    %c_ow = transform.param.constant 32 : i64 -> !transform.param<i64>
    %c_oh = transform.param.constant 32 : i64 -> !transform.param<i64>

    // %in0 = transform.get_operand %matched[0] : (!transform.any_op) -> !transform.any_value
    // transform.match.param.cmpi eq %in0[0], %c_in : !transform.param<i64>
    // transform.match.param.cmpi eq %in0[1], %c_ih : !transform.param<i64>
    // transform.match.param.cmpi eq %in0[2], %c_iw : !transform.param<i64>
    // transform.match.param.cmpi eq %in0[3], %c_ic : !transform.param<i64>
    // // TODO: what's transform.iree.match.cast_compatible_type
    // // transform.iree.match.cast_compatible_type %in0 = tensor<1x?xf32> : !transform.any_value

    // %in1 = transform.get_operand %matched[1] : (!transform.any_op) -> !transform.any_value
    // transform.match.param.cmpi eq %in1[0], %c_fh : !transform.param<i64>
    // transform.match.param.cmpi eq %in1[1], %c_fw : !transform.param<i64>
    // transform.match.param.cmpi eq %in1[2], %c_fc : !transform.param<i64>
    // transform.match.param.cmpi eq %in1[3], %c_ff : !transform.param<i64>
    // // transform.iree.match.cast_compatible_type %out0 = tensor<1xf32> : !transform.any_value

    // %out0 = transform.get_operand %matched[2] : (!transform.any_op) -> !transform.any_value
    // transform.match.param.cmpi eq %out0[0], %c_in : !transform.param<i64>
    // transform.match.param.cmpi eq %out0[1], %c_oh : !transform.param<i64>
    // transform.match.param.cmpi eq %out0[2], %c_ow : !transform.param<i64>
    // transform.match.param.cmpi eq %out0[3], %c_ff : !transform.param<i64>

    // transform.iree.match.cast_compatible_type %out1 = tensor<1xi64> : !transform.any_value

    // transform.iree.match.regions %matched : !transform.any_op {
    //   ^bb0(%target: tensor<1x?xf32>, %empty_max: tensor<1xf32>, %empty_idx: tensor<1xi64>):
    //     %expanded = tensor.expand_shape %6 [[0], [1, 2], [3], [4]] : tensor<2x32x32x1280xf32> into tensor<2x1x32x32x1280xf32>
    //     %7 = tensor.empty() : tensor<2x1x32x32x1280xf16>
    //     %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<2x1x32x32x1280xf32>) outs(%7 : tensor<2x1x32x32x1280xf16>) {
    //     ^bb0(%in: f32, %out: f16):
    //       %10 = arith.truncf %in : f32 to f16
    //       linalg.yield %10 : f16
    //     } -> tensor<2x1x32x32x1280xf16>
    //     %9 = tensor.empty() : tensor<2x1280x32x32xf16>
    //     %collapsed_2 = tensor.collapse_shape %8 [[0], [1, 2], [3], [4]] : tensor<2x1x32x32x1280xf16> into tensor<2x32x32x1280xf16>
    //     %transposed_3 = linalg.transpose ins(%collapsed_2 : tensor<2x32x32x1280xf16>) outs(%9 : tensor<2x1280x32x32xf16>) permutation = [0, 3, 1, 2] 
    // }
    transform.yield %root : !transform.any_op
  }
  
  transform.named_sequence @cast_and_call_conv(%conv: !transform.any_op {transform.readonly}) {
    // transform.print {name = "hi"}
    %module = transform.util.get_nearest_symbol_table %conv : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point into %module if undefined : (!transform.any_op) -> !transform.any_op
    %ins = transform.get_operand %conv[0,1] : (!transform.any_op) -> !transform.any_value
    %outs = transform.get_result %conv[0] : (!transform.any_op) -> !transform.any_value
    transform.util.cast_and_call %func(%ins) -> %outs before %conv {
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op   
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_conv -> @cast_and_call_conv
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.yield
  }
}