import src.c2oObject as Node


def get_MVN_output_shape(input_shape):
    output_shape = input_shape
    return output_shape


def create_MVN_op(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_MVN_output_shape(input_shape)
    node = Node.c2oNode(layer, node_name, "MeanVarianceNormalization", input_name, output_name, input_shape, output_shape)
    return node
