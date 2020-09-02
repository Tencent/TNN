# danerli add 2020.8.14
import torch
import numpy as np

from torch.quantization.fuse_modules import _set_module, _get_module, fuse_known_modules

from quantization_wrap.utils.subgraph_search import conv_bn_match, conv_bn_relu_match, aten_op_match
from quantization_wrap.utils.input_to_ops import InputToOps
from collections import Counter

#init
input_to_ops = None

WRAPPER_LIST = [
    'aten::flatten',
    'aten::max_pool2d',
    'aten::reshape',
    'aten::mean',
    'aten::dropout',
]

INVISIBLE_WRAPPER_LIST = [
    'aten::view',
]

INVISIBLE_LIST = [
    'aten::add_',
    'aten::add',
    'aten::cat',
    'aten::_cat',
]


def _fuse_modules(model, modules_to_fuse, fuser_func=fuse_known_modules):

    mod_list = []
    for item in modules_to_fuse:
        mod_list.append(_get_module(model, item))

    # mod_list.append(torch.nn.ReLU(inplace=True))
    mod_list.append(torch.nn.ReLU(inplace=False))

    # Fuse list of modules
    new_mod_list = fuser_func(mod_list)

    # Replace original module list with fused module list
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])


def _get_layer_name(node):
    return str(node.scopeName()).split('/')[-1].replace('__module.', '')


def get_pre_layer_node(node, result=[]):
    for input_node in node.inputs():
        layer = _get_layer_name(input_node.node())

        if layer is not '' and 'aten' in input_node.node().kind():
            result.append(input_node.node())
        elif list(input_node.node().inputs()) == []:
            continue
        else:
            get_pre_layer_node(input_node.node(), result)
    return result


def get_post_layer_node(node, result=[]):
    # input_to_ops.consumer_operations(node)
    for next_node in input_to_ops.consumer_operations(node):
        layer = _get_layer_name(next_node)
        if layer is not '' and 'aten' in next_node.kind() and not next_node.kind() in ['aten::chunk']:
            result.append(next_node)
        elif list(next_node.outputs()) == None:
            continue
        else:
            get_post_layer_node(next_node, result)
    return result


class GraphOptimizer(object):

    def __init__(self, model, graph, fuse_mode):
        self.model = model
        self.graph = graph
        self.fuse_mode = fuse_mode
        self.input_to_ops = InputToOps(self.graph)
        self.init()

    def init(self):
        global input_to_ops
        input_to_ops = self.input_to_ops

    def graph_optimizer(self):

        self.nodes = [node for node in self.graph.nodes()]

        self.fuse_op()

        self.unrecognized_op_optimizer()

        self.invisible_op_optimizer()


    # fix bug for unrecognized quantized operator for flatten, reshape, maxpool and etc.
    def unrecognized_op_optimizer(self):

        def DeQuantWrapper(layer1, layer2):

            def hook(self, input, output):
                return self.module_post_process(output)

            def pre_hook(self, input):
                return self.module_pre_process(input[0])

            if layer1 is layer2:

                _get_module(self.model, layer1).add_module('module_pre_process', torch.quantization.DeQuantStub())
                _get_module(self.model, layer1).module_pre_process.onnx = True
                _get_module(self.model, layer1).register_forward_pre_hook(pre_hook)

                _get_module(self.model, layer2).add_module('module_post_process', torch.quantization.QuantStub())
                _get_module(self.model, layer2).module_post_process.onnx = True
                _get_module(self.model, layer2).register_forward_hook(hook)

            else:

                _get_module(self.model, layer1).add_module('module_post_process', torch.quantization.DeQuantStub())
                _get_module(self.model, layer1).module_post_process.onnx = True
                _get_module(self.model, layer1).register_forward_hook(hook)

                _get_module(self.model, layer2).add_module('module_pre_process', torch.quantization.QuantStub())
                _get_module(self.model, layer2).module_pre_process.onnx = True
                _get_module(self.model, layer2).register_forward_pre_hook(pre_hook)

        processed_node = set()

        for i, node in enumerate(self.nodes):

            if node.kind() in WRAPPER_LIST and node not in processed_node:
                if node.scopeName() is not '':
                    layer = _get_layer_name(node)
                    DeQuantWrapper(layer, layer)
                else:
                    layer1 = _get_layer_name(get_pre_layer_node(node, result=[])[0]).replace('.conv', '')

                    # Combine multiple unrecognized ops
                    next_node = get_post_layer_node(node, result=[])[0]
                    while next_node.kind() in WRAPPER_LIST:
                        processed_node.add(next_node)
                        next_node = get_post_layer_node(next_node, result=[])[0]
                    layer2 = _get_layer_name(next_node).replace('.conv', '')

                    DeQuantWrapper(layer1, layer2)

            if node.kind() in INVISIBLE_WRAPPER_LIST:

                layer1 = _get_layer_name(get_pre_layer_node(node, result=[])[0]).replace('.conv', '')

                # Combine multiple unrecognized ops
                next_node = get_post_layer_node(node, result=[])[0]
                while next_node.kind() in WRAPPER_LIST:
                    processed_node.add(next_node)
                    next_node = get_post_layer_node(next_node, result=[])[0]
                layer2 = _get_layer_name(next_node).replace('.conv', '')

                DeQuantWrapper(layer1, layer2)

    # fix bug for invisible operator for add, cat and etc.
    def invisible_op_optimizer(self):

        def hook(self, input, output):
            # output = torch.nn.ReLU(inplace=True).forward(output)
            output = self.layer_process_process(output)
            return output

        def pre_hook(self, input):
            return self.layer_process_process(input[0])

        for i, node in enumerate(self.nodes):
            if node.kind() in INVISIBLE_LIST:
                layer = _get_layer_name(node)

                # In this case, the invisible op does not need to add quantization operations
                if layer is '':
                    continue

                pre_layers = get_pre_layer_node(node, result=[])
                post_layers = get_post_layer_node(node, result=[])

                post_layer_name = _get_layer_name(post_layers[0])#.replace('.relu', '')

                # for mobilenet
                if not layer.split('.')[:2] == post_layer_name.split('.')[:2]:
                    post_layer_name = layer

                for pre_layer in pre_layers:

                    # for shufflenetv2
                    if _get_layer_name(pre_layer) == _get_layer_name(node):
                        pre_layer = get_pre_layer_node(pre_layer, result=[])[0]

                    layer_name = _get_layer_name(pre_layer).replace('.relu', '')
                    _get_module(self.model, layer_name).add_module('layer_process_1', torch.quantization.DeQuantStub())
                    _get_module(self.model, layer_name).layer_process_1.onnx = True
                    _get_module(self.model, layer_name).register_forward_hook(lambda self, input, output: self.layer_process_1(output))

                    if not layer_name.split('.')[:3] == post_layer_name.split('.')[:3]:

                        for layer_next_node in get_post_layer_node(pre_layer, result=[]):
                            if not layer_next_node is node:
                                _get_module(self.model, _get_layer_name(layer_next_node)).add_module('layer_process_2', torch.quantization.QuantStub())
                                _get_module(self.model, _get_layer_name(layer_next_node)).layer_process_2.onnx = True
                                _get_module(self.model, _get_layer_name(layer_next_node)).register_forward_pre_hook(lambda self, input: self.layer_process_2(input[0]))

                relus = set(k for k, v in self.aten_op_matches['relu'].items() if v > 1)
                if not post_layer_name in relus and 'relu' in post_layer_name:
                    _get_module(self.model, post_layer_name).add_module('layer_process_3', torch.quantization.QuantStub())
                    _get_module(self.model, post_layer_name).layer_process_3.onnx = True
                    _get_module(self.model, post_layer_name).register_forward_pre_hook(lambda self, input: self.layer_process_3(input[0]))
                    continue

                post_layer_name = post_layer_name.replace('.relu', '')

                _get_module(self.model, post_layer_name).add_module('layer_process_3', torch.quantization.QuantStub())
                _get_module(self.model, post_layer_name).layer_process_3.onnx = True
                _get_module(self.model, post_layer_name).register_forward_hook(lambda self, input, output: self.layer_process_3(output))

                _get_module(self.model, _get_layer_name(node)).add_module('layer_process_process',
                    torch.nn.Sequential(torch.quantization.DeQuantStub(), torch.quantization.QuantStub()))
                _get_module(self.model, _get_layer_name(node)).register_forward_hook(hook)



    # fuse op
    def fuse_op(self):

        if self.fuse_mode == 1:
            layer_matches = conv_bn_match(self.graph)
        elif self.fuse_mode == 2:
            layer_matches = conv_bn_relu_match(self.graph)

        # Mark reused modules
        self.aten_op_matches = aten_op_match(self.graph)
        for i in self.aten_op_matches:
            self.aten_op_matches[i] = Counter(list(np.array(self.aten_op_matches[i]).flatten().tolist()))

        for layer_matche in layer_matches:

            # fix relu for reuse
            if len(layer_matche) == 3 and \
                    isinstance(_get_module(self.model, layer_matche[2]), torch.nn.Identity) and \
                    self.aten_op_matches.get('relu').get(layer_matche[2]) > 1:
                _set_module(self.model, layer_matche[2], torch.nn.ReLU(inplace=True))

            # fix relu does not display the definition directly use torch.nn.functional.relu()
            # InceptionV3 and Googlenet
            if len(layer_matche) == 3 and \
                    not isinstance(_get_module(self.model, layer_matche[2]), torch.nn.Identity) and \
                    not isinstance(_get_module(self.model, layer_matche[2]), torch.nn.ReLU):
                _fuse_modules(self.model, layer_matche[0:2])
                continue

            torch.quantization.fuse_modules(self.model, layer_matche, inplace=True)

        def hook(self, input, output):
            r"""post layer hook that calls observer on the output
            """
            return torch.nn.ReLU(inplace=True).forward(output)

        # resnet add relu
        # layers = set([k.split('conv')[0][:-1] for k, v in self.model.named_parameters() if 'conv' in k and 'layer' in k])
        layers = set(k.replace('.relu', '') for k, v in self.aten_op_matches['relu'].items() if v > 1)
        for layer in layers:
            _get_module(self.model, layer).register_forward_hook(hook)
