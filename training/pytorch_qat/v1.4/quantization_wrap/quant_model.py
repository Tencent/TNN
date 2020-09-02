import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.intrinsic as nni
import torchvision
from torchvision.models.quantization.utils import _replace_relu
from torch.quantization.quantize import propagate_qconfig_, DEFAULT_QAT_MODULE_MAPPING
from torch.quantization.fuse_modules import _get_module
from torch.quantization.default_mappings import (DEFAULT_DYNAMIC_MODULE_MAPPING,
                               DEFAULT_MODULE_MAPPING,
                               DEFAULT_QAT_MODULE_MAPPING,
                               DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST)

from quantization_wrap.graph.graph import Graph
from quantization_wrap.quantization.fake_onnx_qconfig import default_fake_onnx_qconfig

def replace_relu(module):
    _replace_relu(module)


def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output
    """
    return self.activation_post_process(output)


def add_observer_(module):
    r"""Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules that we want to quantize

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
    """
    for child in module.children():
        if type(child) == nnq.FloatFunctional:
            if hasattr(child, 'qconfig') and child.qconfig is not None:
                child.activation_post_process = child.qconfig.activation()
        else:
            add_observer_(child)

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if hasattr(module, 'qconfig') and module.qconfig is not None and \
       len(module._modules) == 0 and not isinstance(module, torch.nn.Sequential):
        # observer and hook will be gone after we swap the module
        if hasattr(module, 'onnx') and module.onnx is True:
            module.qconfig=default_fake_onnx_qconfig
            module.add_module('activation_post_process', module.qconfig.activation())
        else:
            module.add_module('activation_post_process', module.qconfig.activation())
        module.register_forward_hook(_observer_forward_hook)


def prepare(model, inplace=False, black_list = []):
    r"""Prepares a copy of the model for quantization calibration or quantization-aware training.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    The model will be attached with observer or fake quant modules, and qconfig
    will be propagated.

    Args:
        model: input model to be modified in-place
        inplace: carry out model transformations in-place, the original module is mutated
    """
    if not inplace:
        model = copy.deepcopy(model)
    propagate_qconfig_(model)
    # sanity check common API misusage
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in model.modules()):
        warnings.warn("None of the submodule got qconfig applied. Make sure you "
                      "passed correct configuration through `qconfig_dict` or "
                      "by assigning the `.qconfig` attribute directly on submodules")

    # black list
    for module in black_list:
        # delattr(_get_module(model, module), 'qconfig')
        setattr(_get_module(model, module), 'qconfig', None)

    add_observer_(model)
    return model


def convert(module, mapping=None, inplace=False):
    r"""Converts the float module with observers (where we can get quantization
    parameters) to a quantized module.

    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized
                 module type, can be overwrritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated

    """
    if mapping is None:
        mapping = DEFAULT_MODULE_MAPPING
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    # TODO(jerryzh): remove after deciding on the impl of intrinsic modules
    # This is required because intrinsic modules right now are implemented as
    # nn.Sequential and we don't want to swap their constituents
    SWAPPABLE_MODULES = (nni.ConvBn2d,
                         nni.ConvBnReLU2d,
                         nni.LinearReLU,
                         nni.ConvReLU2d,
                         nni.ConvReLU3d)

    for name, mod in module.named_children():
        if type(mod) not in SWAPPABLE_MODULES:
            convert(mod, mapping, inplace=True)
        reassign[name] = swap_module(mod, mapping)

    for key, value in reassign.items():
        module._modules[key] = value

    return module


def swap_module(mod, mapping):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    # Always replace dequantstub with dequantize
    if hasattr(mod, 'qconfig') and mod.qconfig is not None or type(mod) == torch.quantization.DeQuantStub:
        if type(mod) in mapping:
            # if hasattr(mod, 'onnx') and mod.onnx is True:
            #     mod.activation_post_process.set_qparams(scale, zero)
            new_mod = mapping[type(mod)].from_float(mod)
        if hasattr(mod, 'module_pre_process'):
            def pre_hook(self, input):
                return self.module_pre_process(input[0])
            new_mod.add_module('module_pre_process', mod.module_pre_process)
            new_mod.register_forward_pre_hook(pre_hook)
        if hasattr(mod, 'layer_process_2'):
            def pre_hook(self, input):
                return self.layer_process_2(input[0])
            new_mod.add_module('layer_process_2', mod.layer_process_2)
            new_mod.register_forward_pre_hook(lambda self, input: self.layer_process_2(input[0]))
        if hasattr(mod, 'layer_process_3'):
            new_mod.add_module('layer_process_3', mod.layer_process_3)
            new_mod.register_forward_pre_hook(lambda self, input: self.layer_process_3(input[0]))
    return new_mod


class QuantizableModel(nn.Module):

    def __init__(self, model, input_shape=None, qconfig=None, black_list=[], auto=True):
        super(QuantizableModel, self).__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        if input_shape == None:
            warnings.warn("None of input_shape. Make sure you input_shape")
        self.image = torch.randn(input_shape)

        # Use _replace_relu() provided by torchvision to replace torch.nn.relu6() with
        # torch.nn.quantized.functional.relu() that supports activation function for pytorch quantization
        _replace_relu(self.model)

        self.black_list = black_list

        if auto:
            self.fuse_model()
            self.qconfig = qconfig
            self.prepare_qat(black_list=self.black_list)

    def forward(self, x):
        if hasattr(self.model, '_transform_input'):
            x = self._forward(x)
        else:
            x = self.quant(x)
            x = self.model.forward(x)
            x = self.dequant(x)
        return x

    def _forward(self, x):
        x = self.model._transform_input(x)
        x = self.quant(x)
        x = self.model._forward(x)
        x = self.dequant(x[0])
        return x

    def fuse_model(self):
        g = Graph(self.model, self.image)
        g.optimizer()

    def prepare_qat(self, mapping=None, black_list=[]):

        if mapping is None:
            mapping = DEFAULT_QAT_MODULE_MAPPING
        prepare(self, inplace=True, black_list=black_list)
        convert(self, mapping, inplace=True)

    def prepare_qat_simple_for_tnn(self, mapping=None, black_list=[]):

        import torch.nn.quantized as nnq

        def hook(self, input):
            return self.input_pre_process(input[0])

        def add_observer_simple(module):
            r"""Add observer for the leaf child of the module.
            """
            for child in module.children():
                if type(child) == nnq.FloatFunctional:
                    if hasattr(child, 'qconfig') and child.qconfig is not None:
                        child.activation_post_process = child.qconfig.activation()
                else:
                    add_observer_simple(child)

            if hasattr(module, 'qconfig') and module.qconfig is not None and \
                    len(module._modules) == 0 and not isinstance(module, torch.nn.Sequential):
                module.add_module('input_pre_process', module.qconfig.activation())
                module.register_forward_pre_hook(hook)

        if mapping is None:
            mapping = DEFAULT_QAT_MODULE_MAPPING

        propagate_qconfig_(self)
        if not any(hasattr(m, 'qconfig') and m.qconfig for m in self.modules()):
            warnings.warn("None of the submodule got qconfig applied. Make sure you "
                          "passed correct configuration through `qconfig_dict` or "
                          "by assigning the `.qconfig` attribute directly on submodules")
        # black list
        for module in black_list:
            setattr(_get_module(self, module), 'qconfig', None)

        add_observer_simple(self)

        convert(self, mapping, inplace=True)


def test_quant_model(model1, model2, x_shape, layername=None, layername_auto=None, onnx=False, intermediate_results=True):

    temp_result = []
    temp_result_auto = []

    def forward_hook(self, input, output):
        temp_result.append(input)
        temp_result.append(output)

    def forward_hook_auto(self, input, output):
        temp_result_auto.append(input)
        temp_result_auto.append(output)

    x = torch.ones(x_shape)

    model = model1
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(model, inplace=True)

    # print(model)

    if intermediate_results:
        _get_module(model, layername).register_forward_hook(forward_hook)

    # pytorch
    result = model(x)

    model_auto = QuantizableModel(model2, x_shape, auto=False)

    # Optional_black_list = [name for name, module in model_auto.named_modules() if len([x for x in module.named_modules()]) == 1]
    # print(f'Optional shielding quantization network layer:{Optional_black_list}')
    black_list = []

    model_auto.fuse_model()
    model_auto.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    model_auto.prepare_qat(black_list=black_list)

    # print(model_auto)

    if intermediate_results:
        _get_module(model_auto, 'model.' + layername_auto).register_forward_hook(forward_hook_auto)

    # our
    result_auto = model_auto(x)

    # export onnx model
    if onnx:
        torch.onnx.export(model, x, "output_fake.onnx", input_names=['input'], example_outputs=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          verbose=True, training=False, opset_version=9)
        torch.onnx.export(model_auto, x, "output_fake_auto.onnx", input_names=['input'], example_outputs=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          verbose=True, training=False, opset_version=9)

    if intermediate_results:
        print('input')
        try:
            print(temp_result[0].equal(temp_result_auto[0]))
        except:
            print(temp_result[0][0].equal(temp_result_auto[0][0]))

        print('output')
        try:
            print(temp_result[1].equal(temp_result_auto[1]))
        except:
            print(temp_result[1][0].equal(temp_result_auto[1][0]))

    print('result output')
    print(result.equal(result_auto))

    # model convert
    model_auto = convert(model_auto, inplace=True)

    outputs = model_auto(x)
    input_names = ["x"]

    traced = torch.jit.trace(model_auto, x)

    model = traced

    torch.onnx.export(model, x, 'qat.onnx', input_names=input_names, example_outputs=outputs,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

def main():

    print(f'resnet50')
    test_quant_model(
        torchvision.models.quantization.resnet18(pretrained=True),
        torchvision.models.resnet18(pretrained=True),
        x_shape=[2, 3, 224, 224],  layername='layer1', layername_auto='layer1',
        onnx=False, intermediate_results=True)

    print('\n')
    print(f'mobilenet_v2')
    test_quant_model(
        torchvision.models.quantization.mobilenet_v2(pretrained=True),
        torchvision.models.mobilenet_v2(pretrained=True),
        x_shape=[2, 3, 224, 224], layername='features', layername_auto='features',
        onnx=False, intermediate_results=True)

    # print('\n')
    # print(f'shufflenet_v2_x0_5')
    # test_quant_model(
    #     torchvision.models.quantization.shufflenet_v2_x0_5(pretrained=True),
    #     torchvision.models.shufflenet_v2_x0_5(pretrained=True),
    #     x_shape=[2, 3, 224, 224], onnx=False, intermediate_results=False)

    # print('\n')
    # print(f'googlenet')
    # test_quant_model(
    #     torchvision.models.quantization.googlenet(pretrained=True),
    #     torchvision.models.googlenet(pretrained=True, aux_logits=True),
    #     x_shape=[2, 3, 299, 299], layername='inception5b', layername_auto='inception5b',
    #     onnx=False, intermediate_results=True)

if __name__ == '__main__':
    main()

