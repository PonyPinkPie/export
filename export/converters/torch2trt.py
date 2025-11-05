import io

from .torch2onnx import torch2onnx
from .onnx2trt import onnx2trt


def torch2trt(
        model=None,
        dummy_input=None,
        log_level='ERROR',
        max_batch_size=1,
        min_input_shapes=None,
        max_input_shapes=None,
        max_workspace_size=4,
        fp16_mode=True,
        strict_type_constraints=False,
        int8_mode=False,
        int8_calibrator=None,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        fp8_mode=False,
):

    assert not (bool(min_input_shapes) ^ bool(max_input_shapes))

    f = io.BytesIO()
    dynamic_shape = bool(min_input_shapes) and bool(max_input_shapes)
    torch2onnx(model, dummy_input, f, dynamic_shape, opset_version,
               do_constant_folding, verbose)

    f.seek(0)

    trt_model = onnx2trt(f, log_level, max_batch_size, min_input_shapes,
                         max_input_shapes, max_workspace_size, fp16_mode,
                         strict_type_constraints, int8_mode, int8_calibrator, fp8_mode)

    return trt_model
