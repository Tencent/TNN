#include "base.inc"

__kernel void Inverse(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __write_only image2d_t output,
                      __private const int input_batch, __private const int input_channel,
                      __private const int input_height, __private const int input_width) {
    const int cw = get_global_id(0);
    const int hb = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, hb);

    int in_batch   = (hb / input_height);
    int in_channel = (cw / input_width);
    int h          = (hb % input_height);
    int w          = (cw % input_width);

    if (h != 0 || w != 0) {
        return;
    }

    FLOAT4 in0  = RI_F(input, SAMPLER, (int2)(in_channel * input_width + w, in_batch * input_height + h));
    FLOAT4 in1  = RI_F(input, SAMPLER, (int2)(in_channel * input_width + w + 1, in_batch * input_height + h));
    FLOAT4 in2  = RI_F(input, SAMPLER, (int2)(in_channel * input_width + w, in_batch * input_height + h + 1));
    FLOAT4 in3  = RI_F(input, SAMPLER, (int2)(in_channel * input_width + w + 1, in_batch * input_height + h + 1));
    FLOAT4 det4 = in0 * in3 - in1 * in2;
    det4        = ((FLOAT)1.0) / det4;
    FLOAT4 out0 = in3 * det4;
    FLOAT4 out1 = ((FLOAT)(-1)) * in1 * det4;
    FLOAT4 out2 = ((FLOAT)(-1)) * in2 * det4;
    FLOAT4 out3 = in0 * det4;
    WI_F(output, (int2)(cw, hb), out0);
    WI_F(output, (int2)((in_channel * input_width + w + 1), hb), out1);
    WI_F(output, (int2)(cw, in_batch * input_height + h + 1), out2);
    WI_F(output, (int2)((in_channel * input_width + w + 1), in_batch * input_height + h + 1), out3);
}