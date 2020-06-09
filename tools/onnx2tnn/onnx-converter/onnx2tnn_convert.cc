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

//data_type: 0:float 1:half 2:int8 not support now
int onnx2tnn_convert(std::string onnx_model_path, std::string output_dir, std::string algo_version, std::string file_time, int data_type)
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
    Onnx2TNN converter(onnx_model_path, tnn_proto, tnn_model);
    int ret = converter.Convert((DataType)data_type);
    if(ret != 0) {
        DLog("tnn converter error:(%d)\n", ret);
        assert(0);
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
