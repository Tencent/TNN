import module
import numpy as np

module=module.Load("../../model/SqueezeNet/squeezenet_v1.1.ts", input_shapes={"input_0": [1,3,224,224]})
input=np.ones((1,3,224,224), np.float32, 'F')
output=module.forward(input)
print(output)
