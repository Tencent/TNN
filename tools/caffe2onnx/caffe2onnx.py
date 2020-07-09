from src.load_save_model import LoadCaffeModel, SaveOnnxModel
from src.caffe2onnx import Caffe2Onnx
from src.args_parser import parse_args
from src.utils import is_ssd_model

def main(args):
    caffe_graph_path = args.proto_file
    caffe_params_path = args.caffe_model_file

    pos_s = caffe_graph_path.rfind("/")
    if  pos_s == -1:
        pos_s = 0

    pos_dot = caffe_graph_path.rfind(".")
    onnx_name = caffe_graph_path[pos_s+1:pos_dot]
    save_path = caffe_graph_path[0:pos_dot] + '.onnx'
    if args.onnx_file is not None:
        save_path = args.onnx_file

    graph, params = LoadCaffeModel(caffe_graph_path,caffe_params_path)
    print('2. 开始进行模型转换')
    c2o = Caffe2Onnx(graph, params, onnx_name)
    print('3. 创建 onnx 模型')
    onnx_model = c2o.createOnnxModel()
    print('4. 保存 onnx 模型')
    is_ssd = is_ssd_model(caffe_graph_path)
    if is_ssd:
        SaveOnnxModel(onnx_model, save_path, need_polish=False)
    else:
        SaveOnnxModel(onnx_model, save_path, need_polish=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
