// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License.

#ifndef onnx_op_converter_hpp
#define onnx_op_converter_hpp

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "tnn/interpreter/tnn/objseri.h"
#include "onnx.pb.h"
#include "onnx2tnn_prefix.h"

using namespace std;
using namespace onnx;
using namespace TNN_NS;

#ifdef __fp16
typedef __fp16 float16;
#else
typedef uint16_t float16;
#endif

typedef std::map<std::string, onnx::TensorProto> TensorProtoMap;
typedef std::map<std::string, onnx::TensorShapeProto> TensorShapeMap;
struct OnnxNetInfo {
    DataType data_type = DATA_TYPE_FLOAT;
    // onnx weight node and weight reshape node
    TensorProtoMap weights_map;
    TensorShapeMap weights_shape_map;
    std::set<std::string > used_const_node;
    std::map<std::string, onnx::NodeProto> proxy_node_map;
    bool is_3D_model = false;
    int opset = 0;
};

class OnnxOpConverter {
public:
    OnnxOpConverter(string onnx_op_type) {
        onnx_op_type_ = onnx_op_type;
    };
    virtual ~OnnxOpConverter(){};
    string OnnxOpType() {
        return onnx_op_type_;
    };
    virtual string TNNOpType(NodeProto &node, OnnxNetInfo &net_info) = 0;
    std::vector<std::string> GetAllInputNames(NodeProto &node, OnnxNetInfo &net_info);
    virtual std::vector<std::string> GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info);
    virtual std::vector<std::string> GetValidOutputNames(NodeProto &node, OnnxNetInfo &net_info);
    string TNNLayerProto(NodeProto &node, OnnxNetInfo &net_info);
    virtual string TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
        return "";
    };
    virtual void ProcessConstantNode(NodeProto &node ,OnnxNetInfo &net_info) {
        // do nothing
        return;
    };
    
    //TNN是否有固定的layerresource去解析，有的话输入为const的就无需写到constant_map里面，如conv；
    //而concat之前没有，所以其输入日过为const就写到constant_map里面。
    virtual bool HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
        return false;
    };
    
    //有权值写入的返回1， 没有的返回0
    virtual int WriteTNNModel(Serializer *writer, NodeProto &node,
                                   OnnxNetInfo &net_info) {
        return 0;
    };
    
    //write will write the shape and data of tensor
    static int WriteTensorData(const onnx::TensorProto &tensor, Serializer *writer,
                        DataType dst_data_type);
    
    static int WriteRawData(const void *raw_data, int data_count, int src_data_type, Serializer *writer,
                     DataType dst_data_type, std::vector<int32_t> dims);
    //depreceted
    static int WriteRawData(const float *raw_data, int data_count, Serializer *writer,
                     DataType dst_data_type, std::vector<int32_t> dims);

    static int WriteIntTensorData(const onnx::TensorProto& tensor, Serializer* writer);

protected:
    string onnx_op_type_;
};

class OnnxOpConverterManager {
public:
    static std::shared_ptr<OnnxOpConverterManager> &Shared();
    OnnxOpConverterManager();
    ~OnnxOpConverterManager();
    std::shared_ptr<OnnxOpConverter> GetOnnxOpConverter(string onnx_type);
    int SetOnnxOpConverter(string onnx_type,
                           std::shared_ptr<OnnxOpConverter> converter);

private:
    std::map<string, std::shared_ptr<OnnxOpConverter>> converter_map_;
};

template <typename T>
class OnnxOpConverterRegister {
public:
    OnnxOpConverterRegister(string onnx_op_type) {
        auto converter = std::make_shared<T>(onnx_op_type);
        auto &manager  = OnnxOpConverterManager::Shared();
        manager->SetOnnxOpConverter(onnx_op_type, converter);
    };
    OnnxOpConverterRegister(string raidnet_op_type, string onnx_op_type) {
        auto converter = std::make_shared<T>(raidnet_op_type, onnx_op_type);
        auto &manager  = OnnxOpConverterManager::Shared();
        manager->SetOnnxOpConverter(onnx_op_type, converter);
    };
    ~OnnxOpConverterRegister(){};

private:
    OnnxOpConverterRegister();
};

#define DECLARE_OP_CONVERTER(onnx_type)                                        \
    class OnnxOpConverter##onnx_type : public OnnxOpConverter {                \
    public:                                                                    \
        OnnxOpConverter##onnx_type(string ignore) : OnnxOpConverter(ignore){}; \
        virtual ~OnnxOpConverter##onnx_type(){};                               \
        virtual string TNNOpType(NodeProto &, OnnxNetInfo &net_info);     \
        virtual string TNNLayerParam(NodeProto &, OnnxNetInfo &);         \
        virtual bool HasLayerResource(NodeProto &node, OnnxNetInfo &net_info);  \
        virtual int WriteTNNModel(Serializer *, NodeProto &,              \
                                       OnnxNetInfo &);                         \
    }

#define DECLARE_OP_CONVERTER_WITH_FUNC(onnx_type, extra_func)                                        \
    class OnnxOpConverter##onnx_type : public OnnxOpConverter {                \
    public:                                                                    \
        OnnxOpConverter##onnx_type(string ignore) : OnnxOpConverter(ignore){}; \
        virtual ~OnnxOpConverter##onnx_type(){};                               \
        virtual string TNNOpType(NodeProto &, OnnxNetInfo &net_info);     \
        virtual string TNNLayerParam(NodeProto &, OnnxNetInfo &);         \
        virtual bool HasLayerResource(NodeProto &node, OnnxNetInfo &net_info);  \
        virtual int WriteTNNModel(Serializer *, NodeProto &,              \
                                       OnnxNetInfo &);                         \
        extra_func \
    }

#define DECLARE_OP_CONVERTER_WITH_PROCESS(onnx_type)                                                                   \
    class OnnxOpConverter##onnx_type : public OnnxOpConverter {                                                        \
    public:                                                                                                            \
        OnnxOpConverter##onnx_type(string ignore) : OnnxOpConverter(ignore){};                                         \
        virtual ~OnnxOpConverter##onnx_type(){};                                                                       \
        virtual string TNNOpType(NodeProto &, OnnxNetInfo &net_info);                                                  \
        virtual string TNNLayerParam(NodeProto &, OnnxNetInfo &);                                                      \
        virtual int WriteTNNModel(Serializer *, NodeProto &, OnnxNetInfo &);                                           \
        virtual void ProcessConstantNode(NodeProto &node, OnnxNetInfo &net_info);                                      \
    }

#define REGISTER_OP_CONVERTER(converter_suffix, onnx_type)                     \
    OnnxOpConverterRegister<OnnxOpConverter##converter_suffix>                 \
        g_converter_##onnx_type(#onnx_type)



class OnnxOpConverterNoParamNoWeight : public OnnxOpConverter {
public:
    OnnxOpConverterNoParamNoWeight(string tnn_type, string onnx_type)
        : OnnxOpConverter(onnx_type) {
        tnn_type_ = tnn_type;
    };
    virtual ~OnnxOpConverterNoParamNoWeight(){};
    virtual string TNNOpType(NodeProto &, OnnxNetInfo &) {
        return tnn_type_;
    };
    virtual string TNNLayerParam(NodeProto &, OnnxNetInfo &) {
        return "";
    };
    virtual int WriteTNNModel(Serializer *, NodeProto &, OnnxNetInfo &) {
        return 0;
    };

private:
    string tnn_type_;
};

#define REGISTER_OP_CONVERTER_NoParamNoWeight(tnn_type, onnx_type)        \
    OnnxOpConverterRegister<OnnxOpConverterNoParamNoWeight>                    \
        g_converter_##tnn_type_##onnx_type(#tnn_type, #onnx_type)

#endif /* onnx_op_converter_hpp */
