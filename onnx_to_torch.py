import onnx2torch
import onnx

onnx_model_path = "/home/yash/Downloads/resnet_small.onnx"
onnx_model = onnx.load(onnx_model_path)

model = onnx2torch.convert(onnx_model)

print(model)