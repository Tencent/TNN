import src.c2oObject as Node
from typing import List, Dict
import onnx


def get_attributes(layer) -> Dict:
    axis = layer.flatten_param.axis
    end_axis = layer.flatten_param.end_axis
    if end_axis != -1:
        print("not support end_axis param!")
        exit(-1)
    attributes = {
        "axis": axis
    }
    return attributes


def get_flatten_output_shape(input_shape: List,
                             attributes: Dict) -> List:
    shape = input_shape[0]
    input_prod = 1
    axis = attributes.get("axis")
    for i in range(axis, len(shape)):
        input_prod = input_prod * shape[i]

    output_shape = [shape[0:axis]+ [input_prod]]
    return output_shape

def create_flatten_node(layer, node_name : str,
                        input_names: List,
                        output_name: List,
                        input_shape: List) -> onnx.NodeProto:
    attributes = get_attributes(layer)

    output_shape = get_flatten_output_shape(input_shape, attributes)

    node = Node.c2oNode(layer, node_name, "Flatten", input_names,
                        output_name, input_shape, output_shape, attributes)
    return node
