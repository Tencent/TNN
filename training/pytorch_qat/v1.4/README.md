# Quantization-aware training for TRT/TNN

pytorch v1.4 quantize clone from

https://github.com/pytorch/pytorch/tree/9e7dc37f902af78b26739410766e79e63f53e27f/torch/quantization (commit:9e7dc37)



**自动量化工具使用示例:**

目前支持resnet，mobilenetv2，shufflenet_v2，googlenet，后续模型支持持续更新。(googlenet仅支持自动量化，暂不支持ONNX转换)

**一键配置**

```python
import torch
import torchvision
from quantization_wrap.quant_model import QuantizableModel

x_shape = [2, 3, 224, 224]

x = torch.ones(x_shape)

model = torchvision.models.resnet18(pretrained=True)

qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

model_auto = QuantizableModel(model, x_shape, qconfig, black_list=[])

result = model_auto(x)
```

**手动配置**

```python
import torch
import torchvision
from quantization_wrap.quant_model import QuantizableModel

x_shape = [2, 3, 224, 224]

x = torch.ones(x_shape)

model = torchvision.models.resnet18(pretrained=True)

model_auto = QuantizableModel(model, x_shape, auto=False)

# Fuses modules
model_auto.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights and activation

qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
model_auto.qconfig = qconfig

# black_list to add layers without quantization
black_list = []
model_auto.prepare_qat(black_list=black_list)

result = model_auto(x)
```

**模型转换**

```python
from quantization_wrap.quant_model import convert

# model convert
model_auto = convert(model_auto, inplace=True)

outputs = model_auto(x)
input_names = ["x"]

traced = torch.jit.trace(model_auto, x)

model = traced

torch.onnx.export(model, x, 'resnet_qat.onnx', input_names=input_names, example_outputs=outputs,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
```

