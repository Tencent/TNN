from pytnn._pytnn import *
from typing import List, Dict, Any
import numpy

def _supported_input_size_type(input_size) -> bool:
    if isinstance(input_size, tuple):
        return True
    elif isinstance(input_size, list):
        return True
    elif isinstance(input_size, dict):
        return True
    else:
        raise TypeError(
            "input size is required to be a List, tuple or a Dict of two sizes (min, max), found type: "
            + str(type(input_size)))


def _parse_input_ranges(input_sizes, input_names):
    if not isinstance(input_sizes, list) and not isinstance(input_sizes, dict):
        raise KeyError("input sizes required to be a List or Dict, found type: " + str(type(input_sizes)))
    if isinstance(input_sizes, list) and any(not _supported_input_size_type(i) for i in input_sizes):
        raise KeyError("An input size must either be a static size or a range of two sizes (min, max) as Dict")
    if isinstance(input_sizes, dict) and any(not _supported_input_size_type(i) for i in input_sizes.values()):
        raise KeyError("An input size must either be a static size or a range of two sizes (min, max) as Dict")
    min_input_shapes = {}
    max_input_shapes = {}
    if isinstance(input_sizes, list):
        for index, value in enumerate(input_sizes):
            input_name = "input_" + str(index)
            if len(input_names) > index:
                input_name = input_names[index]
            min_value, max_value = _parse_min_max_value(value)
            min_input_shapes[input_name] = min_value
            max_input_shapes[input_name] = max_value
    if isinstance(input_sizes, dict):
        for key, value in input_sizes.items():
            min_value, max_value = _parse_min_max_value(value)
            min_input_shapes[key] = min_value
            max_input_shapes[key] = max_value
    return (min_input_shapes, max_input_shapes)

def _parse_min_max_value(value):
    if isinstance(value, dict):
        if all(k in value for k in ["min", "max"]):
            return (value["min"], value["max"])
        else:
            raise KeyError(
                "An input size must either be a static size or a range of two sizes (min, max) as Dict")
    elif isinstance(value, list):
        return (value, value)
    elif isinstance(value, tuple):
        return (value, value)

def _parse_device_type(device_type):
    if isinstance(device_type, DeviceType):
        return device_type
    elif isinstance(device_type, str):
        if device_type == "cuda" or device_type == "CUDA":
            return DEVICE_CUDA
        elif device_type == "x86" or device_type == "X86":
            return DEVICE_X86
        elif device_type == "arm" or device_type == "ARM":
            return DEVICE_ARM
        elif device_type == "naive" or device_type == "NAIVE":
            return DEVICE_NAIVE
        elif device_type == "metal" or device_type == "METAL":
            return DEVICE_METAL
        elif device_type == "opencl" or device_type == "OPENCL":
            return DEVICE_OPENCL
        else:
            raise ValueError("Got a device_type unsupported (type: " + device_type + ")")
    else:
        raise TypeError("device_type must be of type string or DeviceType, but got: " +
                        str(type(device_type)))         

def _parse_network_type(network_type):
    if isinstance(network_type, NetworkType):
        return network_type
    elif isinstance(network_type, str):
        if network_type == "auto" or network_type == "AUTO":
            return NETWORK_TYPE_AUTO
        elif network_type == "default" or network_type == "DEFAULT":
            return NETWORK_TYPE_DEFAULT
        elif network_type == "openvino" or network_type == "OPENVINO":
            return NETWORK_TYPE_OPENVINO
        elif network_type == "coreml" or network_type == "COREML":
            return NETWORK_TYPE_COREML
        elif network_type == "tensorrt" or network_type == "TENSORRT":
            return NETWORK_TYPE_TENSORRT
        elif network_type == "tnntorch" or network_type == "ATLAS":
            return NETWORK_TYPE_TNNTORCH
        elif network_type == "atlas" or network_type == "ATLAS":
            return NETWORK_TYPE_ATLAS
        else:
            raise ValueError("Got a network_type unsupported (type: " + network_type + ")")
    else:
        raise TypeError("network_type must be of type string or NetworkType, but got: " +
                        str(type(network_type)))

def _parse_precision(precision):
    if isinstance(precision, Precision):
        return precision
    elif isinstance(precision, str):
        if precision == "auto" or precision == "AUTO":
            return PRECISION_AUTO
        if precision == "normal" or precision == "NORMAL":
            return PRECISION_NORMAL
        elif precision == "high" or precision == "HIGH" or precision == "fp32" or precision == "FP32" \
            or precision == "float32" or precision == "FLOAT32":
            return PRECISION_HIGH
        elif precision == "low" or precision == "LOW" or precision == "fp16" or precision == "FP16" \
            or precision == "float16" or precision == "FLOAT16" or precision == "bfp16" or precision == "BFP16":
            return PRECISION_LOW
        else:
            raise ValueError("Got a precision unsupported (type: " + precision + ")")
    else:
        raise TypeError("precision must be of type string or Precision, but got: " +
                        str(type(precision)))

def _parse_share_memory_mode(share_memory_mode):
    if isinstance(share_memory_mode, ShareMemoryMode):
        return share_memory_mode
    elif isinstance(share_memory_mode, str):
        if share_memory_mode == "default" or share_memory_mode == "DEFAULT":
            return SHARE_MEMORY_MODE_DEFAULT
        elif share_memory_mode == "share_one_thread" or share_memory_mode == "SHARE_ONE_THREAD":
            return SHARE_MEMORY_MODE_SHARE_ONE_THREAD
        elif share_memory_mode == "set_from_external" or share_memory_mode == "SET_FROM_EXTERNAL":
            return SHARE_MEMORY_MODE_SET_FROM_EXTERNAL
        else:
            raise ValueError("Got a share_memory_mode unsupported (type: " + share_memory_mode + ")")
    else:
        raise TypeError("share_memory_mode must be of type string or ShareMemoryMode, but got: " +
                        str(type(share_memory_mode)))

def _parse_data_format(data_format):
    if isinstance(data_format, DataFormat):
        return data_format
    elif isinstance(data_format, str):
        if data_format == "NCHW" or data_format == "nchw":
            return DATA_FORMAT_NCHW
        elif data_format == "NC4HW4" or data_format == "nc4hw4":
            return DATA_FORMAT_NC4HW4
        elif data_format == "NHC4W4" or data_format == "nhc4w4":
            return DATA_FORMAT_NHC4W4
        else:
            raise ValueError("Got a data_format unsupported (type: " + data_format + ")")
    else:
        raise TypeError("data_format must be of type string or DataFormat, but got: " +
                        str(type(data_format)))

def _parse_network_config(config_dict):
    network_config = NetworkConfig()
    if "device_type" in config_dict:
        network_config.device_type = _parse_device_type(config_dict["device_type"])
    else:
        network_config.device_type = DEVICE_CUDA
    if "device_id" in config_dict:
        assert isinstance(config_dict["device_id"], int)
        network_config.device_id = config_dict["device_id"]
    if "data_format" in config_dict:
        network_config.data_format = _parse_data_format(config_dict["data_format"])
    if "network_type" in config_dict:
        network_config.network_type = _parse_network_type(config_dict["network_type"])
    if "share_memory_mode" in config_dict:
        network_config.share_memory_mode = _parse_share_memory_mode(config_dict["share_memory_mode"])
    if "library_path" in config_dict:
        assert isinstance(config_dict["library_path"], str)
        network_config.library_path = config_dict["library_path"]
    if "precision" in config_dict:
        network_config.precision = _parse_precision(config_dict["precision"])
    if "cache_path" in config_dict:
        assert isinstance(config_dict["cache_path"], str)
        network_config.cache_path = config_dict["cache_path"]
    if "enable_tune_kernel" in config_dict:
        assert isinstance(config_dict["enable_tune_kernel"], bool)
        network_config.enable_tune_kernel = config_dict["enable_tune_kernel"]
    return network_config

def _replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail

def load(model_path, config_dict = {}):
    module = Module(model_path)
    input_names = module.parsed_input_names()
    min_input_shapes = None
    max_input_shapes = None
    if "input_names" in config_dict:
        input_names = config_dict["input_names"]
    if "input_shapes" in config_dict:
        min_input_shapes, max_input_shapes = _parse_input_ranges(config_dict["input_shapes"], input_names)
    network_config = _parse_network_config(config_dict)
    module.create_inst(network_config, min_input_shapes, max_input_shapes)
    return module

def load_raw(model_path, network_config, input_shapes=None):
    module = Module(model_path)
    module.create_inst(network_config, input_shapes, input_shapes)
    return module
def load_raw_range(model_path, network_config, min_input_shapes, max_input_shapes):
    module = Module(model_path)
    module.create_inst(network_config, min_input_shapes, max_input_shapes)
    return module

class Module:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tnn=TNN()
        self.model_config=ModelConfig()
        if model_path.endswith("tnnproto"):
            weights_path=_replace_last(model_path, "tnnproto", "tnnmodel")
            self.model_config.model_type=MODEL_TYPE_TNN
            params = []
            with open(model_path, "r") as f:
                params.append(f.read())
            with open(weights_path, "rb") as f:
                params.append(f.read())
            self.model_config.params=params
        else:
            self.model_config.model_type=MODEL_TYPE_TORCHSCRIPT
            self.model_config.params=[model_path]
        self.tnn.Init(self.model_config)

    def parsed_input_names(self):
        return self.tnn.GetModelInputNames()

    def parsed_output_names(self):
        return self.tnn.GetModelOutputNames()

    def create_inst(self, network_config, min_input_shapes, max_input_shapes):
        ret=Status()
        if network_config is None:
            network_config=NetworkConfig()
            network_config.device_type=DEVICE_CUDA
        if not isinstance(network_config, NetworkConfig):
            raise TypeError("network_config can be None or of type NetworkConfig, but got: " +
                          str(type(network_config)))
        if self.model_config.model_type == MODEL_TYPE_TORCHSCRIPT:
            network_config.network_type=NETWORK_TYPE_TNNTORCH
        if min_input_shapes is None:
            self.instance=self.tnn.CreateInst(network_config, ret)
        elif max_input_shapes is None:
            self.instance=self.tnn.CreateInst(network_config, ret, min_input_shapes)
        else:
            self.instance=self.tnn.CreateInst(network_config, ret, min_input_shapes, max_input_shapes)

        self.input_names = self.parsed_input_names()
        self.output_names = self.parsed_output_names()

        if len(self.input_names) == 0:
            self.input_names = list(self.instance.GetAllInputBlobs().keys())

    def forward(self, *inputs, rtype="list"):
        input_mats = {}
        if len(inputs) > 1:
            for index, value in enumerate(inputs):
                input_mats[self.input_names[index]] = Mat(value)
        else:
            if isinstance(inputs[0], tuple) or isinstance(inputs[0], list):
                for index, value in enumerate(inputs[0]):
                    input_mats[self.input_names[index]] = Mat(value)
            elif isinstance(inputs[0], dict):
                for key, value in inputs[0].items():
                    input_mats[key] = Mat(value)
            else:
                input_mats[self.input_names[0]] = Mat(inputs[0])
                
        input_shapes = {}
        for key, value in input_mats.items():
            input_shapes[key] = value.GetDims()
        self.instance.Reshape(input_shapes) 
        
        for key, value in input_mats.items():
            self.instance.SetInputMat(value, MatConvertParam(), key) 
      
        self.instance.Forward()
        output_blobs = self.instance.GetAllOutputBlobs()
        output = []
        is_dict = False
        if rtype == "dict":
            output = {}
            is_dict = True
        if len(self.output_names) == 0:
            self.output_names = list(self.instance.GetAllOutputBlobs().keys())
        for output_name in self.output_names:
            output_mat=self.instance.GetOutputMat(MatConvertParam(), output_name, DEVICE_NAIVE, NCHW_FLOAT)
            output_mat_numpy=numpy.array(output_mat, copy=False)
            if is_dict:
                output[key] = output_mat_numpy
            else:
                output.append(output_mat_numpy)
        return output
