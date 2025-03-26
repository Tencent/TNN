from tiacc_inference._pytnn import *
#from pytnn._pytnn import *
from typing import List, Dict, Any
import numpy
import pickle
import genericpath
from subprocess import getoutput
#from tkinter.messagebox import NO
from typing import *
import torch
import sys
import GPUtil
import json
import re
import copy
import os
from threading import Lock
import hashlib
import onnxruntime as ort
import time
import shutil
import torch.nn as nn
import torch.nn.functional as F

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

infer_framework_map = dict()

try:
    import torch
except ModuleNotFoundError:
    infer_framework_map['torch'] = False
else:
    infer_framework_map['torch'] = True

def convert_data_to_shape(obj, pre):
    status = Status(StatusCode.TIACC_OK, '')
    shape = {}
    types = {}
    if isinstance(obj, dict):
        # print('dict')
        for key,value in obj.items():
            shape_tmp, types_tmp, rtn = convert_data_to_shape(value, pre + '[' + key + ']')
            if rtn.code != StatusCode.TIACC_OK:
                return None, None, rtn
            shape = {**shape, **shape_tmp}
            types = {**types, **types_tmp}
        return shape, types, status
    elif isinstance(obj, list):
        # print('list')
        for i in range(len(obj)):
            shape_tmp, types_tmp, rtn = convert_data_to_shape(obj[i], pre + '[' + str(i) + ']')
            if rtn.code != StatusCode.TIACC_OK:
                return None, None, rtn
            shape = {**shape, **shape_tmp}
            types = {**types, **types_tmp}
        return shape, types, status
    elif isinstance(obj, tuple):
        # print('tuple')
        for i in range(len(obj)):
            shape_tmp, types_tmp, rtn = convert_data_to_shape(obj[i], pre + '(' + str(i) + ')')
            if rtn.code != StatusCode.TIACC_OK:
                return None, None, rtn
            shape = {**shape, **shape_tmp}
            types = {**types, **types_tmp}
        return shape, types, status
    elif infer_framework_map['torch'] and torch.is_tensor(obj):
        # print('tensor')
        shape[pre] = list(obj.shape)
        types[pre] = convert_tensor_type(obj.dtype)
        return shape, types, status
    else:
        print("unsupport data type")
        status = Status(StatusCode.TIACC_ERR, "unsupport data type")
        return None, None, status

def gen_shape_from_data(data):
    min_input_shapes, max_input_shapes, status = {}, {}, Status(StatusCode.TIACC_OK, '')
    types = {}
    for ii in range(len(data)):
        name = "input_" + str(ii)
        shape, type, rtn = convert_data_to_shape(data[ii], name)
        return None, None, None, rtn
        # if rtn.code != StatusCode.TIACC_OK: #？？?
            # return None, None, None, rtn

        min_input_shapes = {**min_input_shapes, **shape}
        max_input_shapes = {**max_input_shapes, **shape}
        types = {**types, **type}
    return min_input_shapes, max_input_shapes, types, status

onnx_framework_map = dict()
try:
    import onnxruntime
except ModuleNotFoundError:
    onnx_framework_map['onnx'] = False
else:
    onnx_framework_map['onnx'] = True

def gen_report(test_shape):
    # generate report info
    software_env = []
    if infer_framework_map['torch']:
        torch_version = ""
        try:
            torch_version = torch.__version__.split('+')[0]
        except:
            torch_version = ""
        software_env.append({"software_environment": "pytorch", "version": torch_version})
    if onnx_framework_map['onnx']:
        onnx_version = ""
        try:
            onnx_version = onnxruntime.__version__.split('+')[0]
        except:
            onnx_version = ""
        software_env.append({"software_environment": "onnx", "version": onnx_version})
    # software_env.append({"software_environment": "cuda", "version": GetCUDAVersion()})
    # software_env.append({"software_envrionment": "cudnn", "version": GetCudnnVersion()})
    # software_env.append({"software_environment": "TensorRT", "version": GetTensorRTVersion()})
    software_env.append({"software_environment": "cuda", "version": "11.1"})
    software_env.append({"software_envrionment": "cudnn", "version": "8.3.2"})
    software_env.append({"software_environment": "TensorRT", "version": "8.4.0.6"})
    
    hardware_env = {}
    hardware_env['device_type'] = "gpu"
    gpu_name = ""
    try:
        gpu_name = GPUtil.getGPUs()[0].name
    except:
        gpu_name = ""
    hardware_env['microarchitecture'] = gpu_name

    test_data_info = {}
    test_data_info['test_data_source'] = 'user provided'
    test_data_info['test_data_shape'] = str(test_shape)
    test_data_info['test_data_type'] = ''

    optimization_result = {}
    optimization_result['baseline_time'] = ""
    optimization_result['optimized_time'] = ""
    optimization_result['baseline_qps'] = ""
    optimization_result['optimized_qps'] = ""
    optimization_result['speed_up'] = ""

    status = {}
    status['code'] = StatusCode.TIACC_OK
    status['message'] = ""

    result = {}
    result['software_environment'] = software_env
    result['hardware_environment'] = hardware_env
    result['test_data_info'] = test_data_info
    result['optimization_result'] = optimization_result
    result['status'] = status

    return result

from enum import Enum, unique

@unique
class StatusCode(int, Enum):
    TIACC_OK    = 0

    TIACC_ERR      = 1000

class Status:
    def __init__(self, code=StatusCode.TIACC_OK, message=''):
        self.code = code
        self.message = message
    def get_dict(self):
        return {'code': self.code, 'message': self.message}

shape_type = "^[0-9]+(\*[0-9]+)*$"

def seperate_shapes(shapes : str):
    status = Status(StatusCode.TIACC_OK, '')
    dim = shapes[0].count('*') + 1
    min_shape = [100000] * dim
    max_shape = [-1] * dim
    for shape in shapes:
        if re.match(shape_type, shape) == None:
            print("Error shape format! Shape format should be positive numbers splited by '*', \n\
                   e.g. 'n*c*h*w'.")

            status = Status(StatusCode.TIACC_ERR, "Error shape format! Shape format should be positive numbers splited by '*', \n\
                            e.g. 'n*c*h*w'.")
            return None, None, status
        if shape.count('*') != dim - 1:
            print("Range shape should keep the dim size consistent!")

            status = Status(StatusCode.TIACC_ERR, "Range shape should keep the dim size consistent!")
            return None, None, status
        shape = shape.replace('min:', '')
        shape = shape.replace('max:', '')
        shape = shape.split('*')
        shape = list(map(int, shape))
        for i in range(dim):
            if shape[i] < 1:
                print("Shapes should be positive!")

                status = Status(StatusCode.TIACC_ERR, "Shapes should be positive!")
                return None, None, status
            if shape[i] < min_shape[i]:
                min_shape[i] = shape[i]
            if shape[i] > max_shape[i]:
                max_shape[i] = shape[i]
    return min_shape, max_shape, status

type_pattern = '(int)|(float)|(fp16)|(int8)'
range_pattern = '(range)|(seperate)'
def seperate_key(key: str):
    if re.search('long', key):
        return 'long'
    if re.search('int', key):
        return 'int32'
    if re.search('int8', key):
        return 'int8'
    if re.search('fp16', key):
        return 'fp16'
    return 'float'

def convert_to_tnn_name(obj, pre) -> dict:
    '''
    Returns:
        {
            'min_shapes' : dict of prefix and shape,
            'max_shapes' : dict of prefix and shape,
        },
        Status(TIACC_OK/TIACC_ERR, msg)
    '''
    status = Status(StatusCode.TIACC_OK, '')
    min_shapes, max_shapes, types = {}, {}, {}
    if isinstance(obj, dict):
        # filter keyword 'range'
        key = list(obj.keys())[0]
        if re.search(type_pattern, key) or re.search(range_pattern, key):
            if isinstance(obj[key], list):
                min_shape, max_shape, rtn = seperate_shapes(obj[key])
                if rtn.code != StatusCode.TIACC_OK:
                    return None, rtn
            else:
                min_shape, max_shape, rtn = seperate_shapes([obj[key]])
                if rtn.code != StatusCode.TIACC_OK:
                    return None, rtn
            min_shapes[pre] = min_shape
            max_shapes[pre] = max_shape
            types[pre] = seperate_key(key)
            return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status
        # if 'range' in obj:
        #     min_shape, max_shape = seperate_shapes(obj['range'])
        #     min_shapes[pre] = min_shape
        #     max_shapes[pre] = max_shape
        #     return {'min_shapes': min_shapes, 'max_shapes': max_shapes}
        # if 'seperate' in obj:
        #     min_shape, max_shape = seperate_shapes(obj['seperate'])
        #     min_shapes[pre] = min_shape
        #     max_shapes[pre] = max_shape
        #     return {'min_shapes': min_shapes, 'max_shapes': max_shapes}
        for key,value in obj.items():
            shapes, status = convert_to_tnn_name(value, pre + '[' + key + ']')
            min_shapes = {**min_shapes, **shapes['min_shapes']}
            max_shapes = {**max_shapes, **shapes['max_shapes']}
            types      = {**types, **shapes['types']}
        return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status

    elif isinstance(obj, list):
        # print("list")
        for i in range(len(obj)):
            shapes, status = convert_to_tnn_name(obj[i], pre + '[' + str(i) + ']')
            min_shapes = {**min_shapes, **shapes['min_shapes']}
            max_shapes = {**max_shapes, **shapes['max_shapes']}
            types      = {**types, **shapes['types']}
        return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status

    elif isinstance(obj, tuple):
        # print("tuple")
        for i in range(len(obj)):
            shapes, status = convert_to_tnn_name(obj[i], pre + '(' + str(i) + ')')
            min_shapes = {**min_shapes, **shapes['min_shapes']}
            max_shapes = {**max_shapes, **shapes['max_shapes']}
            types      = {**types, **shapes['types']}
        return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status

    elif isinstance(obj, str):
        # print("string")
        if re.match(shape_type, obj) != None:
            min_shape, max_shape, rtn = seperate_shapes([obj])
            if rtn.code != StatusCode.TIACC_OK:
                return None, rtn
            min_shapes[pre] = min_shape
            max_shapes[pre] = max_shape
            types[pre] = 'float'
            return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status
        else:
            print("Error shape format! Shape format should be positive numbers splited by '*', \n\
                           e.g. 'n*c*h*w'.")

            status = Status(StatusCode.TIACC_ERR, "Error shape format! Shape format should be positive numbers splited by '*', \n\
                            e.g. 'n*c*h*w'.")
            return None, status
    else:
        print('Error type for tnn input name convert!')

        status = Status(StatusCode.TIACC_ERR, 'Error type for tnn input name convert!')
        return None, status

def seperate_shape(input):
    status = Status(StatusCode.TIACC_OK, '')
    min_input, max_input, types = {}, {}, {}
    for ii in range(len(input)):
        name = "input_" + str(ii)
        shapes, rtn = convert_to_tnn_name(input[ii], name)
        if rtn.code != StatusCode.TIACC_OK:
            return None, None, None, rtn

        min_input = {**min_input, **shapes['min_shapes']}
        max_input = {**max_input, **shapes['max_shapes']}
        types     = {**types,     **shapes['types']}
    return min_input, max_input, types, status

def seperate_type_v2(key: str):
    type = 'float'
    if (re.search('int32', key)):
        type = 'int32'
    if (re.search('int64', key)):
        type = 'long'
    if (re.search('half', key)):
        type = 'fp16'
    if (re.search('bool', key)):
        type = 'bool'
    
    format = 'tensor'
    if (re.search('scalar', key)):
        format = 'scalar'
    if (re.search('array', key)):
        format = 'array'
    
    return type, format

def sepearte_info_v2(info: str):
    status = Status(StatusCode.TIACC_OK, '')
    type, shape = info.split('(')
    data_type, format = seperate_type_v2(type)
    shape = shape.replace(')', '')
    shape = shape.replace(' ', '')
    shape = shape.replace('[', '')
    shape = shape.replace(']', '')
    shapes = shape.split(',')
    min_shape, max_shape, status = seperate_shapes(shapes)
    return data_type, format, min_shape, max_shape, status

def seperate_shape_v2(inputs):
    status = Status(StatusCode.TIACC_OK, '')
    min_input, max_input, types, formats = {}, {}, {}, {}
    input: str
    for input in inputs:
        try:
            name, info = input.split(':')
            data_type, format, min_shape, max_shape, status = sepearte_info_v2(info)
            min_input[name] = min_shape
            max_input[name] = max_shape
            types[name]     = data_type
            formats[name]   = format
        except Exception as ex:
            print('shape format error: {}'.format(ex))
            status = Status(StatusCode.TIACC_ERR, 'shape format error: {}'.format(ex))
            return (None, None, None, None, status)
    
    return min_input, max_input, types, formats, status

def build_val(shape, type, format, device):
    if (format == 'tensor'):
        if type == 'float':
            res = torch.rand(*shape)
        if type == 'long':
            res = (torch.rand(*shape)*256.0).long()
        if type == 'int32':
            res = (torch.rand(*shape) * 256.0).int()
        if type == 'fp16':
            res = torch.rand(*shape, dtype = torch.float16)
        if (device == 0):
            res = res.cuda()
        return res

    if (format == 'scalar'):
        if type == 'float':
            res = torch.rand(1)
        if type == 'long':
            res = (torch.rand(*shape)*256.0).long()
        if type == 'int32':
            res = (torch.rand(*shape)*256.0).int()
        if type == 'fp16':
            res = torch.rand(*shape, dtype = torch.float16)
        if (device == 0):
            res = res.cuda()
        return res.item()
    
    if (format == 'array'):
        import numpy as np
        if type == 'float':
            res = np.random.random(shape).astype(np.float32)
        if type == 'long':
            res = np.random.randint(shape).astype(np.int64)
        if type == 'int32':
            res = np.random.randint(shape)
        if type == 'fp16':
            res = np.random.random(shape).astype(np.float16)
        return res

def iterate_name_v2(name: str, inputs, val):
    status = Status(StatusCode.TIACC_OK, '')

    if (len(name) == 0): 
        return val, status
    
    if (re.match('^\[', name) != None):
        res = re.match('^\[\d+\]', name)
        if (res != None):
            # list
            index = int(name[res.start()+1:res.end()-1])
            if (inputs == None):
                inputs = []
            # todo: incorrect order of index may get error
            if (index < len(inputs)):
                inputs[index], status = iterate_name_v2(name[res.end():], inputs[index], val)
            elif (index == len(inputs)):
                rtp = iterate_name_v2(name[res.end():], None, val)
                inputs.append(rtp[0])
                status = rtp[1]
            else:
                status = Status(StatusCode.TIACC_ERR, 'invalid data type')
                return None, status
            return inputs, status
        else:
            res = re.match('^\[.*?\]', name)
            print(res)
            print(name)
            print(inputs)
            if (res != None):
                # dict
                if (inputs == None):
                    inputs = {}
                # inputs = {}
                key = name[res.start()+1:res.end()-1]
                if (inputs.get(key) != None):
                    inputs[key], status = iterate_name_v2(name[res.end():], inputs[key], val)
                else:
                    inputs[key], status = iterate_name_v2(name[res.end():], None, val)
                return inputs, status
            else:
                print("Input name format error.")
                status = Status(StatusCode.TIACC_ERR, 'Input name format error.')
                return None, status
    else:
        print("Input name format error.")
        status = Status(StatusCode.TIACC_ERR, 'Input name format error.')
        return None, status
                
    # todo: tuple
        
def convert_shape_to_data_v2(input_shapes:dict, types:dict, format:dict, device:dict):
    status = Status(StatusCode.TIACC_OK, '')
    # for input_name, format_val in format.items():#???
    #     if format_val = tensor: 
    #         inputs = []
    #         for name, val in input_shapes.items():
    #             import copy
    #             tmp_name = copy.deepcopy(name)
    #             if (re.match('^input_', tmp_name)):
    #                 tmp_name = tmp_name.lstrip('input_')
    #                 mat = re.match('^\d+', tmp_name)
    #                 name_list = list(tmp_name)
    #                 name_list.insert(mat.end(), ']')
    #                 name_list.insert(mat.start(), '[')
    #                 tmp_name = ''.join(name_list)
    #             else:
    #                 print("Input name format error.")
    #                 status = Status(StatusCode.TIACC_ERR, 'Input name format error.')
    #                 return None, status
    #             value = build_val(val, types[name], format[name], device)
    #             inputs, status = iterate_name_v2(tmp_name, inputs, value)
    #     elif format_val = array:
    #         inputs = {}
    #         for name, val in input_shapes.items():
    #             inputs[name] = value
    #             value = build_val(val, types[name], format[name], device)
    # return inputs, status
    inputs = {}
    for name, val in input_shapes.items():
        import copy
        value = build_val(val, types[name], format[name], device)
        inputs[name] = value
    return inputs, status


def gen_torch_tensor(shape, type = 'float'):
    if type == 'long':
        return (torch.rand(*shape)*256.0).long()
    if type == 'int32':
        return (torch.rand(*shape) * 256.0).int()
    if type == 'int8':
        return (torch.rand(*shape) * 256.0).type(torch.int8)
    if type == 'fp16':
        return torch.rand(*shape, dtype = torch.float16)
    return torch.rand(*shape)

def convert_shape_to_data(obj, device_type):
    status = Status(StatusCode.TIACC_OK, '')
    if isinstance(obj, list):
        test_data = []
        for i in range(len(obj)):
            data, rtn = convert_shape_to_data(obj[i], device_type)
            if rtn.code != StatusCode.TIACC_OK:
                return None, rtn
            test_data.append(data)
        return test_data, status

    elif isinstance(obj, dict):
        key = list(obj.keys())[0]
        if re.search(type_pattern, key) or re.search(range_pattern, key):
            if isinstance(obj[key], list):
                min_shape, max_shape, rtn = seperate_shapes(obj[key])
                if rtn.code != StatusCode.TIACC_OK:
                    return None, rtn
            else:
                min_shape, max_shape, rtn = seperate_shapes([obj[key]])
                if rtn.code != StatusCode.TIACC_OK:
                    return None, rtn
            type = seperate_key(key)
            data_tmp = gen_torch_tensor(max_shape, type)
            if device_type == 0:
                data_tmp = data_tmp.cuda()
            return data_tmp, status

        test_data = {}
        for key,value in obj.items():
            data, rtn = convert_shape_to_data(obj[key], device_type)
            if rtn.code != StatusCode.TIACC_OK:
                return None, rtn
            test_data[key] = data
        return test_data, status

    elif isinstance(obj, tuple):
        test_data = []
        for i in range(len(obj)):
            data, rtn = convert_shape_to_data(obj[i], device_type)
            if rtn.code != Status.TIACC_OK:
                return None, rtn
            test_data.append(data)
        return list(test_data), status

    elif isinstance(obj, str):
        min_shape, max_shape, rtn = seperate_shapes([obj])
        if rtn.code != StatusCode.TIACC_OK:
            return None, rtn

        data_tmp = torch.rand(*max_shape)
        if device_type == 0:
            data_tmp = data_tmp.cuda()
        return data_tmp, status
    
    else:
        print('Error input shape format!')
        status = Status(StatusCode.TIACC_ERR, 'Error input shape format!')
        return None, status

def optimize(
    input_model: Any,
    # input_onnx: Any,
    optimization_level: int,
    device_type: int,
    input_shapes = {},
    input_nodes_names = [],
    output_nodes_names = [],
    test_data = [],
    save_path = "",
    device_id = 0,
):
    dirs = os.listdir(input_model)
    flag1 = False
    flag2 = False
    flag3 = False
    for file in dirs:
        if "tnnproto" in file:
            tnnproto_name = file
            flag1 = True
        elif "tnnmodel" in file:
            tnnmodel_name = file
            flag2 = True
        elif "onnx" in file:
            onnx_name = file
            flag3 = True
    if flag1 == False or flag2 == False or flag3 == False:
        print("There is no tnnmodel or onnx in your path.")
        return
    module = Module(input_model + '/' + tnnproto_name)
    # set min&max input shapes
    types = {}
    if len(input_shapes) > 0:
        min_input_shapes, max_input_shapes, types, formats, status = seperate_shape_v2(input_shapes)
        if status.code != StatusCode.TIACC_OK:
            report = gen_report('')
            report['status'] = status.get_dict()
            report = json.dumps(report, indent=4, separators=(',', ': '))
            return (None, report)
    elif len(test_data) > 0:
        min_input_shapes, max_input_shapes, types, status = gen_shape_from_data(test_data,list(min_input_shapes.keys()))
        if status.code != StatusCode.TIACC_OK:
            report = gen_report('')
            report['status'] = status.get_dict()
            report = json.dumps(report, indent=4, separators=(',', ': '))
            return (None, report)
    else:
        report = gen_report('')
        report['status'] = Status(StatusCode.TIACC_ERR,
                                  'Error: At least one between input_shapes and test_data should be provieded!').get_dict()
        report = json.dumps(report, indent=4, separators=(',', ': '))
        return (None, report)
    report = {}

    # set test_data
    if len(test_data) == 0:
        # test_data, status = convert_shape_to_data(input_shapes, device_type)
        test_data, status = convert_shape_to_data_v2(max_input_shapes, types, formats, device_type)
        report = gen_report(max_input_shapes)
        report['test_data_info']['test_data_source'] = 'tiacc provided'
        if status.code != StatusCode.TIACC_OK:
            report['status'] = status.get_dict()
            report = json.dumps(report, indent=4, separators=(',', ': '))
            return (None, report)
    else:
        real_input_shape_res  = gen_shape_from_data(test_data)
        status = real_input_shape_res[3]
        if status.code != StatusCode.TIACC_OK:
            report = gen_report('')
            report['status'] = status.get_dict()
            report = json.dumps(report, indent=4, separators=(',', ': '))
            return (None, report)
        report = gen_report(real_input_shape_res[0])
    # print(min_input_shapes)
    # print(max_input_shapes)
    # print(types)
    # print(formats)
    # print(test_data)
    report['test_data_info']['test_data_type'] = str(types)

    if device_type == 0:
        report['hardware_environment']['device_type'] = 'GPU'
    else:
        report['hardware_environment']['device_type'] = 'CPU'
        try:
            cpu_name = get_cpu_name()
        except:
            cpu_name = ''
        report['hardware_environment']['microarchitecture'] = cpu_name

    config_dict={}
    config_dict["precision"] = "fp32" if optimization_level == 0 else "fp16"
    config_dict["device_type"] = "cuda" if device_type == 0 else "x86"
    config_dict["cache_path"] = save_path
    network_config = _parse_network_config(config_dict)
    module.create_inst(network_config, min_input_shapes, max_input_shapes)
    # save input info as pickle file in save_dir
    a_dict = {'min_input_shapes':min_input_shapes, 'max_input_shapes':max_input_shapes, 'types':types, 'formats':formats, 'precision':config_dict["precision"], 'device_type':device_type}
    file = open(save_path + '/' + 'input_info.pickle','wb')
    pickle.dump(a_dict, file)
    file.close()
    # save tnnproto, tnnmodel and cache in save_dir
    shutil.copyfile(input_model + '/' + tnnproto_name, save_path + '/' + tnnproto_name[:-8] + 'optimize.tnnproto' )
    shutil.copyfile(input_model + '/' + tnnmodel_name, save_path + '/' + tnnproto_name[:-8] + 'optimize.tnnmodel' )

    try:
        # tnn runtime
        N = 50
        output=[]
        output=module.forward(test_data)
        # warm up
        for i in range(10):
            module.forward(test_data)
        time_0=time.time()
        for i in range(N):
            output=module.forward(test_data)
        time_1=time.time()
        time_tnn = (time_1-time_0)/N*1000.0
        # onnxruntime
        test_data_onnx, status = convert_shape_to_data_v2(max_input_shapes, types, formats, device_type)
        # test_data_onnx=test_data
        for name, val in test_data_onnx.items():
            if type(val) == torch.Tensor:
                value = val.cpu().numpy()
                test_data_onnx[name]=value
        ort_sess = ort.InferenceSession(input_model + '/' + onnx_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        outputs_onnx = ort_sess.run(None, test_data_onnx)
        # warm up
        for i in range(10):
            ort_sess.run(None, test_data_onnx)
        time_2 = time.time()
        for i in range(N):
            outputs_onnx = ort_sess.run(None, test_data_onnx)
        time_3 = time.time()
        time_onnx = (time_3-time_2)/N*1000.0
        baseline_time = time_onnx
        optimized_time = time_tnn
    except Exception as e:
        print("Error: " + repr(e))
        report['status'] = Status(StatusCode.TIACC_ERR,
                                  'Error: input model incompatible with test data!').get_dict()
        report['optimization_result']['baseline_time']  = None
        report['optimization_result']['baseline_qps']   = None
        report['optimization_result']['optimized_time'] = None
        report['optimization_result']['optimized_qps']  = None
        report['optimization_result']['speed_up']       = None 
    else:
        report['optimization_result']['baseline_time'] = "%.2f" % baseline_time + "ms"
        report['optimization_result']['baseline_qps']  = "%.2f" % (1000 / baseline_time)
        report['optimization_result']['optimized_time'] = "%.2f" % optimized_time + "ms"
        report['optimization_result']['optimized_qps']  = "%.2f" % (1000 / optimized_time)
        report['optimization_result']['speed_up'] = "%.2f" % (baseline_time / optimized_time) 
        
    report = json.dumps(report, indent=4, separators=(',', ': '))
    print(report)
    return (module, report)

def load(model_path):
    config_dict = {}
    dirs = os.listdir(model_path)
    flag1 = False
    flag2 = False
    flag3 = False
    cache_file = ""
    for file in dirs:
        if "optimize.tnnproto" in file:
            tnnproto_name = file
            flag1 = True
        elif "optimize.tnnmodel" in file:
            tnnmodel_name = file
            flag2 = True
        elif "pickle" in file:
            input_info = file
            flag3 = True
        elif file.endswith(".cache"):
            cache_file = file
    if flag1 == False or flag2 == False or flag3 == False:
        print("There is no optimized tnnmodel or input info in your path.")
        return
    module = Module(model_path + '/' + tnnproto_name)
    # input_names = module.parsed_input_names()
    min_input_shapes = None
    max_input_shapes = None
    config_dict["cache_path"] = model_path + "/" + cache_file
    # if "input_names" in config_dict:
    #     input_names = config_dict["input_names"]
    # if "input_shapes" in config_dict:
    #     min_input_shapes, max_input_shapes = _parse_input_ranges(config_dict["input_shapes"], input_names)
    f = open(model_path + '/' + input_info,'rb')
    c = pickle.load(f)
    min_input_shapes = c["min_input_shapes"]
    max_input_shapes = c["max_input_shapes"]
    config_dict["precision"] = c["precision"]
    config_dict["device_type"] = "cuda" if c["device_type"] == 0 else "x86"
    f.close()
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


class Module(nn.Module):
    def __init__(self, model_path):
        super(Module,self).__init__()
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
        import tiacc_inference
        ret=tiacc_inference._pytnn.Status()
        #import pytnn
        #ret=pytnn._pytnn.Status()
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
        tensor_flag = False
        tensor_gpu_flag = False
        if len(inputs) > 1:
            for index, value in enumerate(inputs):
                if type(value) == torch.Tensor:
                    if (value.is_cuda) == True:
                        tensor_gpu_flag = True
                    tensor_flag = True
                    value = value.cpu().numpy()
                input_mats[self.input_names[index]] = Mat(value)
        else:
            if isinstance(inputs[0], tuple) or isinstance(inputs[0], list):
                for index, value in enumerate(inputs[0]):
                    if type(value) == torch.Tensor:
                        if (value.is_cuda) == True:
                            tensor_gpu_flag = True
                        tensor_flag = True
                        value = value.cpu().numpy()
                    input_mats[self.input_names[index]] = Mat(value)
            elif isinstance(inputs[0], dict):
                for key, value in inputs[0].items():
                    if type(value) == torch.Tensor:
                        if (value.is_cuda) == True:
                            tensor_gpu_flag = True
                        tensor_flag = True
                        value = value.cpu().numpy()
                    input_mats[key] = Mat(value)
            else:
                data = inputs[0]
                if type(data) == torch.Tensor:
                    if (data.is_cuda) == True:
                        tensor_gpu_flag = True
                    tensor_flag = True
                    data = data.cpu().numpy()
                input_mats[self.input_names[0]] = Mat(data)
                
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
            if tensor_flag == True:
                output_mat_final = torch.from_numpy(output_mat_numpy)
                if tensor_gpu_flag == True:
                    output_mat_final = output_mat_final.cuda()
            else:
                output_mat_final = output_mat_numpy
            if is_dict:
                output[key] = output_mat_final
            else:
                output.append(output_mat_final)
        
        return output
