import warnings
warnings.filterwarnings("ignore")

import torch

from .utils import flatten, get_names, to


def torch2onnx(
        model=None,
        dummy_input=None,
        onnx_model_name=None,
        dynamic_shape=False,
        opset_version=17,
        # do_constant_folding=False,
        do_constant_folding=True,
        verbose=False):


    if isinstance(dummy_input, tuple):
        dummy_input = list(dummy_input)
    dummy_input = to(dummy_input, 'cuda')
    model.eval().cuda()
    with torch.no_grad():
        output = model(dummy_input)

    assert not isinstance(dummy_input, dict), 'input should not be dict.'
    assert not isinstance(output, dict), 'output should not be dict'

    input_names = get_names(dummy_input, 'input')
    # print(input_names)
    output_names = get_names(output, 'output')

    dynamic_axes = dict()
    for name, tensor in zip(input_names+output_names,
                            flatten(dummy_input)+flatten(output)):
        dynamic_axes[name] = list(range(tensor.dim())) if dynamic_shape else [0]
    # input(f"dynamic_axes = {dynamic_axes}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
        dynamic_axes=dynamic_axes)

    torch.cuda.empty_cache()
