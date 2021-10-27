from typing import *
import torch
import pytnn

def optimize(
    input_model: Any,
    optimization_level: int,
) -> Tuple[Any, str]:
    network_config=pytnn.NetworkConfig()
    network_config.device_type=pytnn.DEVICE_CUDA;
    network_config.network_type=pytnn.NETWORK_TYPE_TNNTORCH 
    opt_model=pytnn.CompileTorch(input_model._c, {"input_0":[1,3,224,224]}, network_config) 
    py_opt_model=torch.jit._recursive.wrap_cpp_module(opt_model)
    py_opt_model(torch.rand(1, 3, 224, 224).cuda())
    return (py_opt_model, "optimize")
