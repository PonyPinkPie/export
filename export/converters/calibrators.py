import os
import torch
import numpy as np
import tensorrt as trt
from abc import ABCMeta, abstractmethod
from .utils import *


class BaseCalibrator(object):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,
    ):

        super(BaseCalibrator, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.cache_file = cache_file

        # create buffers that will hold data batches
        self.buffers = []
        for data in flatten(self.dataset[0]):
            size = (self.batch_size,) + tuple(data.shape)
            buf = torch.zeros(
                size=size, dtype=data.dtype, device='cuda').contiguous()
            self.buffers.append(buf)

        self.num_batch = len(dataset) // self.batch_size
        self.batch_idx = 0

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self, *args, **kwargs):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache, *args, **kwargs):
        if self.cache_file is not None:
            with open(self.cache_file, "wb") as f:
                f.write(cache)

    def get_batch(self, *args, **kwargs):
        if self.batch_idx < self.num_batch:
            for i in range(self.batch_size):
                inputs = flatten(
                    self.dataset[self.batch_idx*self.batch_size+i])
                for buffer, inp in zip(self.buffers, inputs):
                    buffer[i].copy_(inp)
            self.batch_idx += 1

            return [int(buf.data_ptr()) for buf in self.buffers]
        else:
            return []

    def __str__(self):
        raise NotImplemented


class LegacyCalibrator(BaseCalibrator, trt.IInt8LegacyCalibrator):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,
            quantile=None,
            regression_cutoff=None,
    ):

        BaseCalibrator.__init__(self, dataset, batch_size, cache_file)
        trt.IInt8LegacyCalibrator.__init__(self)

        if quantile:
            self.quantile = quantile

            def get_quantile():
                return self.quantile
            setattr(self, 'get_quantile', get_quantile)

        if regression_cutoff:
            self.regression_cutoff = regression_cutoff

            def get_regression_cutoff():
                return self.regression_cutoff
            setattr(self, 'get_regression_cutoff', get_regression_cutoff)

    def read_histogram_cache(self, length):
        pass

    def write_histogram_cache(self, data, length):
        pass

    def __str__(self):
        return 'legacy'


class EntropyCalibrator(BaseCalibrator, trt.IInt8EntropyCalibrator):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,

    ):

        BaseCalibrator.__init__(self, dataset, batch_size, cache_file)
        trt.IInt8EntropyCalibrator.__init__(self)

    def __str__(self):
        return 'entropy'


class EntropyCalibrator2(BaseCalibrator, trt.IInt8EntropyCalibrator2):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,

    ):

        BaseCalibrator.__init__(self, dataset, batch_size, cache_file)
        trt.IInt8EntropyCalibrator2.__init__(self)

    def __str__(self):
        return 'entropy_2'


class MinMaxCalibrator(BaseCalibrator, trt.IInt8MinMaxCalibrator):
    def __init__(
            self,
            dataset,
            batch_size=1,
            cache_file=None,

    ):

        BaseCalibrator.__init__(self, dataset, batch_size, cache_file)
        trt.IInt8MinMaxCalibrator.__init__(self)

    def __str__(self):
        return 'minmax'

class Dataset(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        """base dataset
        """

        super(Dataset, self).__init__()

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

class CustomDataset(Dataset):
    def __init__(self, inputs, targets=None):

        super(CustomDataset, self).__init__()

        self.inputs_forms = get_forms(inputs)
        self.inputs = flatten(inputs)

        if targets is not None:
            self.targets_forms = get_forms(targets)
            self.targets = flatten(targets)

    def __getitem__(self, idx):
        inputs = [inp[idx] for inp in self.inputs]
        inputs = reconstruct(inputs, self.inputs_forms)

        if hasattr(self, 'targets'):
            targets = [target[idx] for target in self.targets]
            targets = reconstruct(targets, self.targets_forms)

            return inputs, targets

        return inputs  # [3, H, W]

    def __len__(self):
        return self.inputs[0].shape[0]