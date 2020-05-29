inline FLOAT4 ActivationProcess(FLOAT4 out0) {
#ifdef RELU
    return fmax(out0, (FLOAT4)0);
#elif defined RELU6
    return clamp(out0, (FLOAT4)0, (FLOAT4)6);
#else
    return out0;
#endif
}
