import pytnn

def Load(model_path, network_config=None, min_input_shapes={}, max_input_shapes={}):
    return Module(model_path, network_config, min_input_shapes, max_input_shapes)

def Load(model_path, network_config=None, input_shapes={}):
    return Module(model_path, network_config, input_shapes, input_shapes)

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
