# 1. Convert model
```bash
/usr/src/tensorrt/bin/trtexec --onnx=dinov2_11value.onnx \
                                --saveEngine=dinov2_11value_batch128.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x224x224 \
                                --optShapes=input:128x3x224x224 \
                                --maxShapes=input:128x3x224x224 \
                                --verbose \
                                --device=1
```