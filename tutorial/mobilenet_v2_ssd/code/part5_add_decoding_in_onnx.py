import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper


# ------------------------------ create decode model ------------------------------
def create_slice_node(name: str, start: list, end: list, axes: list, step: list, input_name: str, output_shape: list):
    param_dict = {"start": start, "end": end, "axes": axes, "step": step}
    inputs_name = [input_name]
    initializer_list = []
    for suffix, param in param_dict.items():
        initializer_name = "{}_{}".format(name, suffix)
        value = np.array(param, dtype=np.int64)
        initializer = helper.make_tensor(initializer_name, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype],
                                         value.shape, value)
        inputs_name.append(initializer_name)
        initializer_list.append(initializer)

    output = helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
    slice_def = helper.make_node(op_type="Slice", inputs=inputs_name, outputs=[name])

    return slice_def, initializer_list, [output, ]


def create_mul_node(name: str, input_name: str, output_shape: list, weights=None):
    initializer_name = "{}_{}".format(name, "weights")
    data_type = weights.dtype if type(weights) is np.ndarray else np.dtype(np.float32)
    shape = weights.shape if type(weights) is np.ndarray else (len(weights),)
    value = list(weights.reshape(-1, )) if type(weights) is np.ndarray else weights
    initializer = helper.make_tensor(initializer_name, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data_type], shape, value)
    output = helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
    mul_def = helper.make_node(op_type="Mul", inputs=[input_name, initializer_name], outputs=[name])

    return mul_def, [initializer, ], [output, ]


def create_add_node(name: str, input_name: str, output_shape: list, weights=None):
    initializer_name = "{}_{}".format(name, "weights")
    data_type = weights.dtype if type(weights) is np.ndarray else np.dtype(np.float32)
    shape = weights.shape if type(weights) is np.ndarray else (len(weights),)
    value = list(weights.reshape(-1, )) if type(weights) is np.ndarray else weights
    initializer = helper.make_tensor(initializer_name, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data_type], shape, value)
    output = helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
    add_def = helper.make_node(op_type="Add", inputs=[input_name, initializer_name], outputs=[name])

    return add_def, [initializer, ], [output, ]


def create_exp_node(name: str, input_name: str, output_shape: list):
    output = helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
    exp_def = helper.make_node(op_type="Exp", inputs=[input_name, ], outputs=[name])

    return exp_def, [output, ]


def create_concat_node(name: str, axis: int, inputs_name, output_shape: list):
    output = helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
    concat_def = helper.make_node(op_type="Concat", inputs=inputs_name, outputs=[name], axis=axis)

    return concat_def, [output, ]


box = helper.make_tensor_value_info('box', TensorProto.FLOAT, [1, 1917, 1, 4])

slice_yx, slice_yx_ini, slice_yx_output = create_slice_node("slice_yx", [0], [2], [3], [1], box.name, [1, 1917, 1, 2])
slice_hw, slice_hw_ini, slice_hw_output = create_slice_node("slice_hw", [2], [4], [3], [1], box.name, [1, 1917, 1, 2])

yx_mul_const, yx_mul_const_ini, yx_mul_const_output = create_mul_node("yx_mul_const", "slice_yx", [1, 1917, 1, 2],
                                                                      [0.1])

anchor_yx = np.load("anchor_data/anchor_yx.npy")
anchor_hw = np.load("anchor_data/anchor_hw.npy")

anchor_yx = np.expand_dims(anchor_yx.transpose((1, 2, 0)), axis=0)
anchor_hw = np.expand_dims(anchor_hw.transpose((1, 2, 0)), axis=0)

yx_mul_anchor_hw, yx_mul_anchor_hw_ini, yx_mul_anchor_hw_output = create_mul_node("yx_mul_anchor_hw", "yx_mul_const",
                                                                                  [1, 1917, 1, 2], anchor_hw)
yx_add_anchor_yx, yx_add_anchor_yx_ini, yx_add_anchor_yx_output = create_add_node("yx_add_anchor_yx",
                                                                                  "yx_mul_anchor_hw", [1, 1917, 1, 2],
                                                                                  anchor_yx)

hw_mul_const, hw_mul_const_ini, hw_mul_const_output = create_mul_node("hw_mul_const", "slice_hw", [1, 1917, 1, 2],
                                                                      [0.2])

hw_exp, hw_exp_output = create_exp_node("hw_exp", "hw_mul_const", [1, 1917, 1, 2])

hw_mul_anchor_hw, hw_mul_anchor_hw_ini, hw_mul_anchor_hw_output = create_mul_node("hw_mul_anchor_hw", "hw_exp",
                                                                                  [1, 1917, 1, 2], anchor_hw)

concat, concat_output = create_concat_node("output", -1, ["yx_add_anchor_yx", "hw_mul_anchor_hw"], [1, 1917, 1, 2])

graph_def = helper.make_graph(
    [slice_yx, yx_mul_const, yx_mul_anchor_hw, yx_add_anchor_yx, slice_hw, hw_mul_const, hw_exp, hw_mul_anchor_hw,
     concat],  # nodes
    'test-model',  # name
    [box],  # inputs
    concat_output,  # outputs
    initializer=slice_yx_ini + yx_mul_const_ini + yx_mul_anchor_hw_ini + yx_add_anchor_yx_ini + slice_hw_ini + hw_mul_const_ini + hw_mul_anchor_hw_ini
)

model_def = helper.make_model(graph_def, producer_name='onnx-example')
model_def.ir_version = 6
imp = model_def.opset_import[0]
imp.version = 11

onnx.checker.check_model(model_def)
print("The model is checked!")

onnx.save(model_def, "decoder.onnx")

# ------------------------------ merge model ------------------------------
ssd_model = onnx.load("saved_model_modify.onnx")
decode_model = onnx.load("decoder.onnx")

combined_model = onnx.compose.merge_models(ssd_model, decode_model, io_map=[("box", "box"), ])

onnx.save(combined_model, './new_model.onnx')
print("The model is merged!")
