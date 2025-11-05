import os
import re
import torch
import torch.nn as nn
import tensorrt as trt
from .utils import flatten, get_names, to
from os.path import join as opj
from os.path import exists as ope
from .onnx2trt import onnx2trt

trt_version = [n for n in trt.__version__.split('.')]
__all__ = ['TRTModel', 'load', 'save']


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.uint8:
        return torch.uint8
    else:
        raise TypeError('{} is not supported by torch'.format(dtype))


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        raise TypeError('{} is not supported by torch'.format(device))


class TRTModel(nn.Module):
    def __init__(self, engine: str, **kwargs):
        super(TRTModel, self).__init__()
        assert engine.split('.')[-1] in ['onnx', 'engine'], f"path must end with '.onnx' or '.engine', got {engine.split('.')[-1]}"
        engine_file = engine.replace(".onnx", ".engine")
        onnx_file = engine.replace('.engine', '.onnx')

        self.pre_trt_10_version = int(trt_version[0]) < 10

        if ope(engine_file):
            self.engine = load(engine_file)
        elif ope(onnx_file):
            self.engine = onnx2trt(onnx_file, **kwargs)
            if self.pre_trt_10_version:
                save(self.engine, engine_file)
            else:
                save_post_trt_10(self.engine, engine_file)
                with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(self.engine)
        else:
            raise RuntimeError(f"{engine_file} and {onnx_file} are not exist")
        self.context = self.engine.create_execution_context()

        self.version_pre_10 = int(trt_version[0]) < 10

        if self.version_pre_10:
            self.update_name_binding_maps_pre_trt_10()
        else:
            self.update_name_binding_maps_post_trt_10()

        # default profile index is 0
        self.profile_index = 0

    def update_name_binding_maps_pre_trt_10(self):
        # get engine input tensor names and output tensor names
        self.input_names, self.output_names = [], []
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            if not re.match(r'.* \[profile \d+\]', name):
                if self.engine.binding_is_input(idx):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)

    def update_name_binding_maps_post_trt_10(self):
        # get engine input tensor names and output tensor names
        self.input_names, self.output_names = [], []
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if not re.match(r'.* \[profile \d+\]', name):
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)



    @staticmethod
    def _rename(idx, name):
        if idx > 0:
            name += ' [profile {}]'.format(idx)

        return name

    def _activate_profile_pre_trt_10(self, inputs):
        for idx in range(self.engine.num_optimization_profiles):
            is_matched = True
            for name, inp in zip(self.input_names, inputs):
                name = self._rename(idx, name)
                min_shape, _, max_shape = self.engine.get_profile_shape(idx, name)
                for s, min_s, max_s in zip(inp.shape, min_shape, max_shape):
                    is_matched = min_s <= s <= max_s

            if is_matched:
                if self.profile_index != idx:
                    self.profile_index = idx
                    self.context.active_optimization_profile = idx

                return True

        return False

    def _activate_profile_post_trt_10(self, inputs):
        for idx in range(self.engine.num_optimization_profiles):
            is_matched = True
            for name, inp in zip(self.input_names, inputs):
                name = self._rename(idx, name)
                min_shape, _, max_shape = self.engine.get_tensor_profile_shape(name, 0)
                for s, min_s, max_s in zip(inp.shape, min_shape, max_shape):
                    is_matched = min_s <= s <= max_s

            if is_matched:
                if self.profile_index != idx:
                    self.profile_index = idx
                    self.context.active_optimization_profile = idx

                return True

        return False


    def _set_binding_shape_pre_trt_10(self, inputs):
        for name, inp in zip(self.input_names, inputs):
            name = self._rename(self.profile_index, name)
            idx = self.engine.get_binding_index(name)
            binding_shape = tuple(self.context.get_binding_shape(idx))
            shape = tuple(inp.shape)
            if shape != binding_shape:
                self.context.set_binding_shape(idx, shape)

    def _set_binding_shape_post_trt_10(self, inputs):
        for name, inp in zip(self.input_names, inputs):
            name = self._rename(self.profile_index, name)
            # idx = self.engine.get_binding_index(name)
            # binding_shape = tuple(self.context.get_binding_shape(idx))
            # shape = tuple(inp.shape)
            # if shape != binding_shape:
            #     self.context.set_binding_shape(idx, shape)
            binding_shape = self.context.get_tensor_shape(name)
            # tensor_shape = self.engine.get_tensor_shape(name)
            shape = tuple(inp.shape)
            if binding_shape != shape:
                self.context.set_input_shape(name, shape)





    def _get_bindings_pre_trt_10(self, inputs):
        bindings = [None] * self.total_length
        outputs = [None] * self.output_length

        for i, name in enumerate(self.input_names):
            name = self._rename(self.profile_index, name)
            idx = self.engine.get_binding_index(name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            bindings[idx % self.total_length] = (
                inputs[i].to(dtype).contiguous().data_ptr())

        for i, name in enumerate(self.output_names):
            name = self._rename(self.profile_index, name)
            idx = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(idx))
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(
                size=shape, dtype=dtype, device=device).contiguous()
            outputs[i] = output
            bindings[idx % self.total_length] = output.data_ptr()

        return outputs, bindings

    def _get_bindings_post_trt_10(self, inputs):
        outputs = [None] * self.output_length

        for i, input_name in enumerate(self.input_names):
            shape = tuple(inputs[i].shape)
            data_ptr = inputs[i].contiguous().data_ptr()
            self.context.set_tensor_address(input_name, data_ptr)
            self.context.set_input_shape(input_name, shape)

        for i, output_name in enumerate(self.output_names):
            dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            shape = tuple(self.context.get_tensor_shape(output_name))
            device = torch_device_from_trt(self.engine.get_tensor_location(output_name))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            self.context.set_tensor_address(output_name, output.data_ptr())

        return outputs



    @property
    def input_length(self):
        return len(self.input_names)

    @property
    def output_length(self):
        return len(self.output_names)

    @property
    def total_length(self):
        return self.input_length + self.output_length

    def forward_pre_trt_10(self, inputs):
        inputs = flatten(inputs)
        # support dynamic shape when engine has explicit batch dimension.
        if not self.engine.has_implicit_batch_dimension:
            status = self._activate_profile_pre_trt_10(inputs)
            assert status, (
                f'input shapes {[inp.shape for inp in inputs]} out of range')
            self._set_binding_shape_pre_trt_10(inputs)

        outputs, bindings = self._get_bindings_pre_trt_10(inputs)

        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)

        return outputs

    def forward_post_trt_10(self, inputs):
        inputs = flatten(inputs)
        # support dynamic shape when engine has explicit batch dimension.
        # if not self.engine.has_implicit_batch_dimension:
        status = self._activate_profile_post_trt_10(inputs)
        assert status, (f'input shapes {[inp.shape for inp in inputs]} out of range')
        self._set_binding_shape_post_trt_10(inputs)

        outputs = self._get_bindings_post_trt_10(inputs)

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        # self.context.execute_async_v3(torch.cuda.Stream().cuda_stream)
        return outputs


    def __call__(self, inputs):
        if self.version_pre_10:
            return self.forward_pre_trt_10(inputs)
        else:
            return self.forward_post_trt_10(inputs)




def load(engine, log_level='ERROR'):

    logger = trt.Logger(getattr(trt.Logger, log_level))

    with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    return engine


def save(engine, name):

    with open(name, 'wb') as f:
        f.write(engine.serialize())

def save_post_trt_10(engine, name):

    with open(name, 'wb') as f:
        f.write(engine)
