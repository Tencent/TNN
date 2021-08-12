import pytnn
import numpy as np

#tnn api
tnn=pytnn.TNN()

model_config=pytnn.ModelConfig()
model_config.model_type=pytnn.MODEL_TYPE_TORCHSCRIPT
model_config.params=["../../model/SqueezeNet/squeezenet_v1.1.ts"]
tnn.Init(model_config)

network_config=pytnn.NetworkConfig()
network_config.device_type=pytnn.DEVICE_CUDA;
network_config.network_type=pytnn.NETWORK_TYPE_TNNTORCH

ret=pytnn.Status()
instance=tnn.CreateInst(network_config, ret, {"input_0":[1,3,224,224]})

input=np.ones((1,3,224,224), np.float32, 'F')
#input_mat=pytnn.Mat(pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT, [1,3,224,224], input.tobytes())
input_mat=pytnn.convert_numpy_to_mat(input)
instance.SetInputMat(input_mat, pytnn.MatConvertParam(), "input_0")
instance.Forward()
command_queue=instance.GetCommandQueue()

#output_mat=pytnn.Mat(pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT)
#instance.GetOutputMat(output_mat, pytnn.MatConvertParam(), "", pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT)
output_mat=instance.GetOutputMat(pytnn.MatConvertParam(), "output_0", pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT)
output_mat_numpy=pytnn.convert_mat_to_numpy(output_mat)
print(output_mat_numpy)

output_mat=instance.GetOutputMat(pytnn.MatConvertParam(), "output_0", pytnn.DEVICE_CUDA, pytnn.NCHW_FLOAT)
dims=output_mat.GetDims()
cpu_output_mat = pytnn.Mat(pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT, dims)
pytnn.MatUtils.Copy(output_mat, cpu_output_mat, command_queue)
output_mat_numpy=pytnn.convert_mat_to_numpy(cpu_output_mat)
print(output_mat_numpy)

output_blobs=instance.GetAllOutputBlobs()
for value in output_blobs.values():
    desc = value.GetBlobDesc()
    print (desc.name, desc.dims, desc.device_type, desc.data_type, desc.data_format)

