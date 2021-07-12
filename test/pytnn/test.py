import pytnn
import numpy as np

#module api
input=np.ones((1,3,224,224), np.float32, 'F')
module=pytnn.load("/home/neiltian/resnet50.ts")
output=module.forward(input)
print(output)

#tnn api
tnn=pytnn.TNN()

model_config=pytnn.ModelConfig()
model_config.model_type=pytnn.MODEL_TYPE_TORCHSCRIPT
model_config.params=["/home/neiltian/resnet50.ts"]
tnn.Init(model_config)

network_config=pytnn.NetworkConfig()
network_config.device_type=pytnn.DEVICE_CUDA;
network_config.network_type=pytnn.NETWORK_TYPE_TNNTORCH

ret=pytnn.Status()
instance=tnn.CreateInst(network_config, ret, {"input_0":[1,3,224,224]})

#input_mat=pytnn.Mat(pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT, [1,3,224,224], input.tobytes())
input_mat=pytnn.convert_numpy_to_mat(input)
instance.SetInputMat(input_mat, pytnn.MatConvertParam())
instance.Forward()

#output_mat=pytnn.Mat(pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT)
#instance.GetOutputMat(output_mat, pytnn.MatConvertParam(), "", pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT)
output_mat=instance.GetOutputMat(pytnn.MatConvertParam(), "", pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT)

output_mat_numpy=pytnn.convert_mat_to_numpy(output_mat)
print(output_mat_numpy)
