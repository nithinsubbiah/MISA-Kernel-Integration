export HSA_OVERRIDE_GFX_VERSION=9.4.2

./../iree/build_rocm/tools/iree-compile --iree-hal-target-backends=rocm     --iree-rocm-target-chip=gfx942     --iree-rocm-link-bc=true --iree-rocm-bc-dir=./../2024-q1-sdxl-sprint/bitcode-2024-03-07  conv_linalg.mlir -o conv.vmfb

./../iree/build_rocm/tools/iree-run-module --device=rocm --input=2x1280x34x34xf16 --module=conv.vmfb --function=forward

#  ./../iree/build_rocm/tools/iree-compile --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=rocm --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu-features=host --iree-llvmcpu-target-triple=x86_64-linux-gnu --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 --iree-hal-target-backends=rocm     --iree-rocm-target-chip=gfx940 --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode/ --iree-preprocessing-transform-spec-filename=conv_transform_spec.mlir --iree-stream-resource-max-allocation-size=4294967296  --iree-util-zero-fill-elided-attrs --mlir-disable-threading pure_f16.mlir -o conv.vmfb