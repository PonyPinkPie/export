from export.converters import TRTModel, onnx2trt
from export.converters.model import save_post_trt_10
import torch
from tqdm import tqdm
import shutil


onnx_path = 'ckpt/resnet18.onnx'
engine_path = onnx_path.replace('.onnx', '.engine')

trt_cfg = dict(
    max_batch_size=1,
    fp16_mode=True,
    fp8_mode=False,
    int8_mode=False,
)
print('no engine do onnx2trt')
model = TRTModel(onnx_path, **trt_cfg)
print('no engine do onnx2trt done')

print('move to fp16')
shutil.move(engine_path, engine_path.replace('.engine', '_fp16.engine'))


dummy_input = torch.ones(1, 3, 416, 384).cuda()
_ = model(dummy_input)



for _ in tqdm(range(10000)):
    _ = model(dummy_input)