inline bool InRange(int2 coor, int2 dims) {
    return 0 <= coor.x && coor.x < dims.x && 0 <= coor.y && coor.y < dims.y;
}

// Conv-Add fusion helpers.
// When -DFUSE_ADD=1 is enabled the conv kernel reads an extra image2d_t add_input
// (sharing the same image layout as the output) and accumulates it into the output
// before writing back. The accumulation is done at the same (cw, bh) coordinate as
// the corresponding output write -- this matches BinaryElementWise semantics exactly,
// keeping the fused result IEEE-754 identical to a separate Conv + Add execution.
//
// FUSE_ADD_BEFORE_ACT controls the order with respect to ActivationProcess:
//   - undefined (default): add is applied AFTER activation.
//                          Corresponds to FusionType_Conv_Activation_Add.
//   - defined            : add is applied BEFORE activation.
//                          Corresponds to FusionType_Conv_Add_Activation
//                          (only emitted by the optimizer when activation_type==None,
//                           in which case ActivationProcess is identity and order
//                           does not matter).
#ifdef FUSE_ADD
#define FUSE_ADD_DECL_PARAM       , __read_only image2d_t add_input
#define FUSE_ADD_LOAD(var, x, y)  FLOAT4 var = RI_F(add_input, SAMPLER, (int2)((x), (y)))
#else
#define FUSE_ADD_DECL_PARAM
#endif

#ifdef CHECK_INPUT_COOR
#define READ_INPUT_IMAGE(i, base)                                                                                      \
    int in_width_value##i = in_width##i + base;                                                                        \
    in_width_value##i =                                                                                                \
        select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_wh.x));            \
    if (InRange((int2)(in_width_value##i, in_hb_value), input_dims))                                                   \
        in##i = RI_F(input, SAMPLER, (int2)(in_width_value##i, in_hb_value));                                          \
    else                                                                                                               \
        in##i = (FLOAT4)0;
#else
#define READ_INPUT_IMAGE(i, base)                                                                                      \
    int in_width_value##i = in_width##i + base;                                                                        \
    in_width_value##i =                                                                                                \
        select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_wh.x));            \
    in##i = RI_F(input, SAMPLER, (int2)(in_width_value##i, in_hb_value));
#endif

inline void WriteOutputAntiOutOfBounds(__write_only image2d_t output, FLOAT4 out0, FLOAT4 out1, FLOAT4 out2, FLOAT4 out3,const int output_w_idx, const int output_h_idx, const int remain)
{ 
    if (remain >= 4) {
        WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
        WI_F(output, (int2)(output_w_idx + 1, output_h_idx), out1);
        WI_F(output, (int2)(output_w_idx + 2, output_h_idx), out2);
        WI_F(output, (int2)(output_w_idx + 3, output_h_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
        WI_F(output, (int2)(output_w_idx + 1, output_h_idx), out1);
        WI_F(output, (int2)(output_w_idx + 2, output_h_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
        WI_F(output, (int2)(output_w_idx + 1, output_h_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
    }
}

inline void WriteSliceOutputAntiOutOfBounds(__write_only image2d_t output, FLOAT4 out0, FLOAT4 out1, FLOAT4 out2, FLOAT4 out3,const int output_w_idx, const int output_h_idx, const int remain)
{
    WI_F(output, (int2)(output_w_idx, output_h_idx), out0);
    if (remain >= 2) {
        WI_F(output, (int2)(output_w_idx + 1, output_h_idx), out1);
    }
    if (remain >= 3) {
        WI_F(output, (int2)(output_w_idx + 2, output_h_idx), out2);
    }
    if (remain >= 4) {
        WI_F(output, (int2)(output_w_idx + 3, output_h_idx), out3);
    }
}
