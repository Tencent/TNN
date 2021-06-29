#include "base.inc"

__kernel void Tile(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __write_only image2d_t output,
                   __private const int input_batch, __private const int input_channel, __private const int input_height,
                   __private const int input_width, __private const int output_batch,
                   __private const int output_channel, __private const int output_height,
                   __private const int output_width, __private const int out_chw, __private const int out_hw) {
    const int cw = get_global_id(0);
    const int hb = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, hb);
    int out_batch   = hb / output_height;
    int out_channel = cw / output_width;
    int out_height  = hb % output_height;
    int out_width   = cw % output_width;

    // compute four position
    // 0 index
    int index = out_batch * out_chw + out_channel * 4 * out_hw + out_height * output_width + out_width;

    int batch = index / out_chw % input_batch;

    int channel = index / out_hw % input_channel;

    int h = index / output_width % input_height;

    int w = index % input_width;

    FLOAT4 in  = RI_F(input, SAMPLER, (int2)(channel / 4 * input_width + w, batch * input_height + h));
    int idx    = channel % 4;
    FLOAT out0 = idx == 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));

    // 1 index
    index = index + output_height * output_width;

    batch = index / out_chw % input_batch;

    channel = index / out_hw % input_channel;

    h = index / output_width % input_height;

    w = index % input_width;

    in         = RI_F(input, SAMPLER, (int2)(channel / 4 * input_width + w, batch * input_height + h));
    idx        = channel % 4;
    FLOAT out1 = idx == 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));
    out1       = (out_channel * 4 + 1) >= output_channel ? (FLOAT)0 : out1;

    // 2 index
    index = index + output_height * output_width;

    batch = index / out_chw % input_batch;

    channel = index / out_hw % input_channel;

    h = index / output_width % input_height;

    w = index % input_width;

    in         = RI_F(input, SAMPLER, (int2)(channel / 4 * input_width + w, batch * input_height + h));
    idx        = channel % 4;
    FLOAT out2 = idx == 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));
    out2       = (out_channel * 4 + 2) >= output_channel ? (FLOAT)0 : out2;

    // 3 index
    index = index + output_height * output_width;

    batch = index / out_chw % input_batch;

    channel = index / out_hw % input_channel;

    h = index / output_width % input_height;

    w = index % input_width;

    in         = RI_F(input, SAMPLER, (int2)(channel / 4 * input_width + w, batch * input_height + h));
    idx        = channel % 4;
    FLOAT out3 = idx == 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));
    out3       = (out_channel * 4 + 3) >= output_channel ? (FLOAT)0 : out3;

    FLOAT4 out = (FLOAT4)(out0, out1, out2, out3);
    WI_F(output, (int2)(cw, hb), out);
}

__kernel void Tile_nhw(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __write_only image2d_t output,
                       __private const int input_batch, __private const int input_channel,
                       __private const int input_height, __private const int input_width,
                       __private const int output_batch, __private const int output_channel,
                       __private const int output_height, __private const int output_width) {
    const int cw = get_global_id(0);
    const int hb = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, hb);
    int out_batch   = (hb / output_height) % input_batch;
    int out_channel = (cw / output_width);
    int h           = (hb % output_height) % input_height;
    int w           = (cw % output_width) % input_width;

    FLOAT4 in = RI_F(input, SAMPLER, (int2)(out_channel * input_width + w, out_batch * input_height + h));
    WI_F(output, (int2)(cw, hb), in);
}