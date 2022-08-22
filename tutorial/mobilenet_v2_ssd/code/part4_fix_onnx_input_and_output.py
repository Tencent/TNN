import onnx

onnx_model = onnx.load("saved_model.onnx")


def modify_input_name(model, src_name, dst_name):
    for i in range(len(onnx_model.graph.input)):
        if onnx_model.graph.input[i].name == src_name:
            onnx_model.graph.input[i].name = dst_name
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].input)):
            if onnx_model.graph.node[i].input[j] == src_name:
                onnx_model.graph.node[i].input[j] = dst_name

    return model


def modify_output_name(model, src_name, dst_name):
    for i in range(len(onnx_model.graph.output)):
        if onnx_model.graph.output[i].name == src_name:
            onnx_model.graph.output[i].name = dst_name
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].output)):
            if onnx_model.graph.node[i].output[j] == src_name:
                onnx_model.graph.node[i].output[j] = dst_name

    return model


modify_input_name(onnx_model, "Preprocessor/sub:0", "input")
modify_output_name(onnx_model, "Postprocessor/convert_scores:0", "score")
modify_output_name(onnx_model, "concat:0", "box")

onnx.save(onnx_model, "saved_model_modify.onnx")
