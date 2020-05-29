from google.protobuf import text_format
from proto import caffe_upsample_pb2
import onnx
from onnx import utils


def LoadCaffeModel(net_path, model_path):
    # read prototxt
    net = caffe_upsample_pb2.NetParameter()
    text_format.Merge(open(net_path).read(), net)
    # read caffemodel
    model = caffe_upsample_pb2.NetParameter()
    f = open(model_path, 'rb')
    model.ParseFromString(f.read())
    f.close()
    return net, model


def LoadOnnxModel(onnx_path):
    onnxmodel = onnx.load(onnx_path)
    return onnxmodel


def SaveOnnxModel(onnx_model, onnx_save_path, need_polish=True):
    try:

        if need_polish:
            polished_model = onnx.utils.polish_model(onnx_model)
            onnx.save_model(polished_model, onnx_save_path)
        else:
            onnx.save_model(onnx_model, onnx_save_path)
        print("模型保存成功,已保存至:" + onnx_save_path)
    except Exception as e:
        print("模型存在问题,未保存成功:", e)
