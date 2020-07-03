#include "abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(SignedMul, LAYER_SIGNED_MUL);

    Status SignedMulLayerInterpreter::InterpretProto();

    Status SignedMulLayerInterpreter::InterpretResource();

    Status SignedMulLayerInterpreter::SaveProto();

    Status SignedMulLayerInterpreter::SaveResource();

REGISTER_LAYER_INTERPRETER(SignedMul, LAYER_SIGNED_MUL);

} // namespace TNN_NS