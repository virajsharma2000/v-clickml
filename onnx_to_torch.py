import onnx2torch
import onnx

onnx_model_path = "welcome_to_click_ml.onnx"
onnx_model = onnx.load(onnx_model_path)

model = onnx2torch.convert(onnx_model)

print(model)