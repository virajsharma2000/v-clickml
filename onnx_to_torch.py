import onnx2torch
import onnx

onnx_model_path = "/home/yash/Downloads/logistic_regression.onnx"

onnx_model = onnx.load(onnx_model_path)

model = onnx2torch.convert(onnx_model)

print(model)