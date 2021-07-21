import pytnn
import numpy as np

input=np.ones((1,3,224,224), np.float32, 'F')

# torchscript 
#module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[ {"min": [1,3,224,224], "max": [1,3,224,224]} ], "network_type": pytnn.NETWORK_TYPE_TNNTORCH, "device_type": pytnn.DEVICE_CUDA})
#module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[ {"min": [1,3,224,224], "max": [1,3,224,224]} ]})
#module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[[1,3,224,224]]})
module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[(1,3,224,224)]})
output=module.forward(input)
print(output[0])

# tnnproto
module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.tnnproto")
#output=module.forward({"data": input}, rtype="dict")
#for key, value in output.items():
#    print (key, value)
output=module.forward({"data" : input})
print(output[0])

