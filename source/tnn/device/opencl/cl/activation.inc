enum ActivationType {
    ActivationType_None        = 0x0000,
    ActivationType_ReLU        = 0x0001,
    ActivationType_ReLU6       = 0x0002,
    ActivationType_SIGMOID_MUL = 0x0100,
};

inline FLOAT4 ActivationProcess(FLOAT4 out0, enum ActivationType activation_type) {
    if (activation_type == ActivationType_ReLU) {
        return fmax(out0, (FLOAT4)0);
    } else if (activation_type == ActivationType_ReLU6) {
        return clamp(out0, (FLOAT4)0, (FLOAT4)6);
    } else if (activation_type == ActivationType_SIGMOID_MUL) {
        float4 convert_out0 = convert_float4(out0);
        return CONVERT_FLOAT4(native_recip((float4)1 + native_exp(-convert_out0)) * convert_out0);
    } else {
        return out0;
    }
}
