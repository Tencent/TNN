inline FLOAT4 ActivationProcess(FLOAT4 out0) {
#ifdef RELU
    return fmax(out0, (FLOAT4)0);
#elif defined RELU6
    return clamp(out0, (FLOAT4)0, (FLOAT4)6);
#elif defined SIGMOID_MUL
    float4 convert_out0 = convert_float4(out0);
    return CONVERT_FLOAT4(native_recip((float4)1 + native_exp(-convert_out0)) * convert_out0);
#else
    return out0;
#endif
}
