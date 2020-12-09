#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

#include <stdlib.h>

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(FusedReduce, LAYER_FUSED_REDUCE);

Status FusedReduceLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    FusedReduceLayerParam* layer_param = new FusedReduceLayerParam();
    *param                            = layer_param;
    int index                         = start_index;

    // pool_type
    layer_param->axis = atoi(layer_cfg_arr[index++].c_str());

    return TNN_OK;
}

Status FusedReduceLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status FusedReduceLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    FusedReduceLayerParam* layer_param = dynamic_cast<FusedReduceLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->axis << " ";

    return TNN_OK;
}

Status FusedReduceLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(FusedReduce, LAYER_FUSED_REDUCE);

}  // namespace TNN_NS
