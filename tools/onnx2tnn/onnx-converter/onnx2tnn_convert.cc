#include <float.h>
#include <stdio.h>
#include <limits.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <limits>
#include <algorithm>

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/xattr.h>
#include <sys/types.h>

#include "onnx2tnn.h"
#include "tnn/core/const_folder.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/tnn/model_packer.h"
#include "tnn/core/blob.h"
using namespace TNN_NS;

#include <pybind11/pybind11.h>

static string version_key = "user.version";
static string time_key = "user.time";

int onnx2tnn_time(std::string file_path)
{
    //获取版本信息到文件属性
    if (file_path.length() > 0) {
        char file_time[1024] = {'\0'};
#if __APPLE__
        getxattr(file_path.c_str(), time_key.c_str(), file_time, 1024, 0, 0);
#elif __linux__
        getxattr(file_path.c_str(), time_key.c_str(), file_time, 1024);
#endif
        printf("%s\n", file_time);
    }
    return 0;
}

int onnx2tnn_set_time(std::string file_path, std::string file_time)
{
    //设置版本信息到文件属性
    if (file_time.length() > 0) {
        if (file_time.length() > 1024) {
            DLog("time length must <= 1024\n");
            assert(0);
        }
#if __APPLE__
        if (setxattr(file_path.c_str(), time_key.c_str(), file_time.c_str(), file_time.length(), 0, 0) != 0) {
            DLog("setxattr error\n");
            assert(0);
        }
#elif __linux__
        if (setxattr(file_path.c_str(), time_key.c_str(), file_time.c_str(), file_time.length(), 0) != 0) {
            DLog("setxattr error\n");
            // assert(0);
        }
#endif
    }
    return 0;
}

int onnx2tnn_version(std::string file_path)
{
    //获取版本信息到文件属性
    if (file_path.length() > 0) {
        char file_version[1024] = {'\0'};
#if __APPLE__
        getxattr(file_path.c_str(), version_key.c_str(), file_version, 1024, 0, 0);
#elif __linux__
        // getxattr(file_path.c_str(), version_key.c_str(), file_version, 1024);
#endif
        printf("%s\n", file_version);
    }
    return 0;
}

int onnx2tnn_set_version(std::string file_path, std::string file_version)
{
    //设置版本信息到文件属性
    if (file_version.length() > 0) {
        if (file_version.length() > 1024) {
            DLog("version length must <= 1024\n");
            // assert(0);
        }
#if __APPLE__
        if (setxattr(file_path.c_str(), version_key.c_str(), file_version.c_str(), file_version.length(), 0, 0) != 0) {
            DLog("setxattr error\n");
            assert(0);
        }
#elif __linux__
        if (setxattr(file_path.c_str(), version_key.c_str(), file_version.c_str(), file_version.length(), 0) != 0) {
            DLog("setxattr error\n");
            // assert(0);
        }
#endif
    }
    return 0;
}


Status parse_input_info(std::string input_info, TNN_NS::InputShapesMap & input_shape_map) {
    if (input_info.empty()) {
        return TNN_NS::TNN_OK;
    }
    int size = input_info.size();
    std::vector<std::string> split_input_info;
    Status status = SplitUtils::SplitStr(input_info.c_str(), split_input_info, " ", true, false);
    if (status != TNN_NS::TNN_OK) {
        return Status(TNNERR_INVALID_NETCFG, "split input info error\n");
    }
    for (auto& item: split_input_info) {
        str_arr input_cfg_vec;
        status = SplitUtils::SplitStr(item.c_str(), input_cfg_vec, ":", true, false);
        if (status != TNN_NS::TNN_OK || input_cfg_vec.size() < 2) {
            return Status(TNNERR_INVALID_NETCFG, "split input line error\n");
        }
        auto input_name = input_cfg_vec[0];
        DimsVector& input_shape = input_shape_map[input_name];
        str_arr input_shape_vec;
        status = SplitUtils::SplitStr(input_cfg_vec[1].c_str(), input_shape_vec, ",", true, false);
        if (status != TNN_NS::TNN_OK || input_shape_vec.size() < 1) {
            return Status(TNNERR_INVALID_NETCFG, "split input line error\n");
        }
        for (int i = 0; i < input_shape_vec.size() ; ++i) {
            input_shape.push_back(atoi(input_shape_vec[i].c_str()));
        }
    }
    return TNN_NS::TNN_OK;
}


//data_type: 0:float 1:half 2:int8 not support now
int onnx2tnn_convert(std::string onnx_model_path, std::string output_dir, std::string algo_version,
                     std::string file_time, int data_type, int fixed_input_shape, std::string input_info)
{
    std::string onnx_model_name;
    std::string onnx_suffix  = ".onnx";
    std::size_t sep_position = onnx_model_path.rfind('/');
    if (sep_position != std::string::npos) {
        onnx_model_name =
            onnx_model_path.substr(sep_position + 1, onnx_model_path.size() - 1 - onnx_suffix.size() - sep_position);
    }
    std::string tnn_proto = output_dir + "/" + onnx_model_name + ".tnnproto";
    std::string tnn_model = output_dir + "/" + onnx_model_name + ".tnnmodel";
    TNN_NS::InputShapesMap input_shape_map = {};
    LOGE("The input_info %s\n", input_info.c_str());
    Status status = parse_input_info(input_info, input_shape_map);
    Onnx2TNN converter(onnx_model_path, tnn_proto, tnn_model, input_shape_map);
    int ret = converter.Convert((DataType)data_type);
    if(ret != 0) {
        DLog("tnn converter error:(%d)\n", ret);
        return -1;
    }

    //do net const folding
    {
        //网络初始化
        NetworkConfig network_config;
        {
            network_config.network_type = NETWORK_TYPE_DEFAULT;
            network_config.device_type =  DEVICE_NAIVE;
        }

        ModelConfig model_config;
        {
            model_config.model_type = MODEL_TYPE_TNN;
            {
                std::ifstream proto_stream(tnn_proto);
                if (!proto_stream.is_open() || !proto_stream.good()) {
                    DLog("read proto_file failed!\n");
                    return -1;
                }
                auto buffer = std::string((std::istreambuf_iterator<char>(proto_stream)),
                                          std::istreambuf_iterator<char>());
                model_config.params.push_back(buffer);
            }

            {
                std::ifstream model_stream(tnn_model);
                if (!model_stream.is_open() || !model_stream.good()) {
                    DLog("read model_file failed!\n");
                    return -1;
                }
                auto buffer = std::string((std::istreambuf_iterator<char>(model_stream)),
                                          std::istreambuf_iterator<char>());
                model_config.params.push_back(buffer);
            }
        }
        
        auto interpreter = dynamic_cast<DefaultModelInterpreter *>(CreateModelInterpreter(model_config.model_type));
        if (!interpreter) {
            return Status(TNNERR_NET_ERR, "interpreter is nil");
        }
        auto status = interpreter->Interpret(model_config.params);
        if (status != TNN_OK) {
            DLog("Interpret Error: %s\n", status.description().c_str());
            return status;
        }

        auto const_folder = std::make_shared<ConstFolder>();
        status = const_folder->Init(network_config, model_config, interpreter, {}, {});
        if (status != TNN_OK) {
            DLog("ConstFolder Init Error: %s\n", status.description().c_str());
            return status;
        }

        std::shared_ptr<NetStructure> opt_structure = nullptr;
        std::shared_ptr<NetResource> opt_resource = nullptr;
        status = const_folder->GetOptimizedNet(opt_structure, opt_resource,
                                        fixed_input_shape ? DATA_FLAG_CHANGE_IF_SHAPE_DIFFER : DATA_FLAG_CHANGE_NEVER);
        if (status != TNN_OK) {
            DLog("GetOptimizedNet Error: %s\n", status.description().c_str());
            return status;
        }
        
        auto packer = std::make_shared<ModelPacker>(opt_structure.get(), opt_resource.get());
        if (packer->Pack(tnn_proto, tnn_model) != 0) {
            DLog("ModelPacker Pack failed!\n");
            return -1;
        }
    }
    
    //添加版本信息到文件属性
    onnx2tnn_set_version(tnn_proto, algo_version);
    onnx2tnn_set_version(tnn_model, algo_version);

    //添加时间信息到文件属性
    if (file_time.length() > 0) {
      onnx2tnn_set_time(tnn_proto, file_time);
      onnx2tnn_set_time(tnn_model, file_time);
    }

    return ret;
}

PYBIND11_MODULE(onnx2tnn, m) {
    m.doc() = "pybind11 onnx2tnn plugin"; // optional module docstring
    m.def("convert", &onnx2tnn_convert, "A function to convert onnx to tnn");
    m.def("version", &onnx2tnn_version, "A function to get file version");
    m.def("set_version", &onnx2tnn_set_version, "A function to set version to tnn");
    m.def("time", &onnx2tnn_time, "A function to get file time");
    m.def("set_time", &onnx2tnn_set_time, "A function to set time to tnn");
}
