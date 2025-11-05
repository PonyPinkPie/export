import tensorrt as trt
from .utils import *
from .calibrators import EntropyCalibrator2, CustomDataset

trt_version = [n for n in trt.__version__.split('.')]


def onnx2trt(
        model,
        log_level='ERROR',
        max_batch_size=1,
        min_input_shapes=None,
        max_input_shapes=None,
        max_workspace_size=1,
        fp16_mode=True,
        strict_type_constraints=False,
        int8_mode=False,
        int8_calibrator=None,
        fp8_mode=True,
):
    logger = trt.Logger(getattr(trt.Logger, log_level))
    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if isinstance(model, str):
        with open(model, 'rb') as f:
            flag = parser.parse(f.read())
    else:
        flag = parser.parse(model.read())
    if not flag:
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    # re-order output tensor
    output_tensors = [network.get_output(i)
                      for i in range(network.num_outputs)]

    [network.unmark_output(tensor) for tensor in output_tensors]
    for tensor in output_tensors:
        identity_out_tensor = network.add_identity(tensor).get_output(0)
        identity_out_tensor.name = 'identity_{}'.format(tensor.name)
        network.mark_output(tensor=identity_out_tensor)

    if int(trt_version[0]) < 10:
        builder.max_batch_size = max_batch_size

    config = builder.create_builder_config()
    if int(trt_version[0]) < 10:
        config.max_workspace_size = max_workspace_size * (1 << 32)
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        if int8_calibrator is None:
            shapes = [(1,) + network.get_input(i).shape[1:]
                      for i in range(network.num_inputs)]
            dummy_data = gen_ones(shapes)
            int8_calibrator = EntropyCalibrator2(CustomDataset(dummy_data))
        config.int8_calibrator = int8_calibrator
    if fp8_mode:
        config.set_flag(trt.BuilderFlag.FP8)
    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)


    # set dynamic shape profile
    assert not (bool(min_input_shapes) ^ bool(max_input_shapes))

    profile = builder.create_optimization_profile()

    input_shapes = [network.get_input(i).shape[1:]
                    for i in range(network.num_inputs)]
    if not min_input_shapes:
        min_input_shapes = input_shapes
    if not max_input_shapes:
        max_input_shapes = input_shapes

    assert len(min_input_shapes) == len(max_input_shapes) == len(input_shapes)

    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        min_shape = (1,) + min_input_shapes[i]
        max_shape = (max_batch_size,) + max_input_shapes[i]
        opt_shape = [(min_ + max_) // 2
                     for min_, max_ in zip(min_shape, max_shape)]
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    if int(trt_version[0]) < 10:
        engine = builder.build_engine(network, config)
    else:
        engine = builder.build_serialized_network(network, config)
    return engine
