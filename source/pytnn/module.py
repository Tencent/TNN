from typing import List, Dict, Any
import pytnn

def _supported_input_size_type(input_size) -> bool:
    if isinstance(input_size, tuple):
        return True
    elif isinstance(input_size, list):
        return True
    else:
        raise TypeError(
            "Input sizes for inputs are required to be a List, tuple or a Dict of two sizes (min, max), found type: "
            + str(type(input_size)))


def _parse_input_ranges(input_sizes: List):
    print(input_sizes)
    if any(not isinstance(i, dict) and not _supported_input_size_type(i) for i in input_sizes):
        raise KeyError("An input size must either be a static size or a range of two sizes (min, max) as Dict")
    min_input_shapes = {}
    max_input_shapes = {}
    for index, value in enumerate(input_sizes):
        if isinstance(value, dict):
            if all(k in value for k in ["min", "max"]):
                min_input_shapes["input_" + str(index)] = value["min"];
                max_input_shapes["input_" + str(index)] = value["max"];
            else:
                raise KeyError(
                    "An input size must either be a static size or a range of three sizes (min, opt, max) as Dict")
        elif isinstance(value, list):
            min_input_shapes["input_" + str(index)] = value;
            max_input_shapes["input_" + str(index)] = value;
        elif isinstance(value, tuple):
            min_input_shapes["input_" + str(index)] = value;
            max_input_shapes["input_" + str(index)] = value;
    return (min_input_shapes, max_input_shapes)


def load(model_path, network_config=None, min_input_shapes={}, max_input_shapes={}):
    return Module(model_path, network_config, min_input_shapes, max_input_shapes)

def load(model_path, network_config=None, input_shapes={}):
    return Module(model_path, network_config, input_shapes, input_shapes)

def load(model_path, config_dict):
    min_input_shapes, max_input_shapes = _parse_input_ranges(config_dict["input_shapes"])
    return Module(model_path, None, min_input_shapes, max_input_shapes)

class Module:
    def __init__(self, model_path, network_config, min_input_shapes, max_input_shapes):
        self.model_path = model_path
        self.tnn=pytnn.TNN()
        model_config=pytnn.ModelConfig()
        model_config.model_type=pytnn.MODEL_TYPE_TORCHSCRIPT
        model_config.params=[model_path]
        self.tnn.Init(model_config)
        ret=pytnn.Status()
        if network_config is None:
            network_config=pytnn.NetworkConfig()
            network_config.device_type=pytnn.DEVICE_CUDA;
            network_config.network_type=pytnn.NETWORK_TYPE_TNNTORCH
        self.instance=self.tnn.CreateInst(network_config, ret, min_input_shapes, max_input_shapes)

    def forward(self, *inputs):
        for index, value in enumerate(inputs):
            input_mat=pytnn.convert_numpy_to_mat(value)
            self.instance.SetInputMat(input_mat, pytnn.MatConvertParam(), "input_" + str(index))
        self.instance.Forward()
        
        output_mat=self.instance.GetOutputMat(pytnn.MatConvertParam(), "output_0", pytnn.DEVICE_NAIVE, pytnn.NCHW_FLOAT)
        output_mat_numpy=pytnn.convert_mat_to_numpy(output_mat)
        return output_mat_numpy
