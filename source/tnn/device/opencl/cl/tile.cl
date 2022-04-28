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

__kernel void Tile6D(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __write_only image2d_t output,
                     shape_6d input_shape, shape_6d output_shape) {
    const int image_width_idx  = get_global_id(0);  // ic/4 d4 d5 ic4
    const int image_height_idx = get_global_id(1);  // b d2 d3
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int in_d2xd3 = input_shape.data[2] * input_shape.data[3];
    const int in_d4xd5 = input_shape.data[4] * input_shape.data[5];

    const int out_d2xd3         = output_shape.data[2] * output_shape.data[3];
    const int out_d4xd5         = output_shape.data[4] * output_shape.data[5];
    const int out_d3xd4xd5      = output_shape.data[3] * out_d4xd5;
    const int out_d2xd3xd4xd5   = output_shape.data[2] * out_d3xd4xd5;
    const int out_cxd2xd3xd4xd5 = output_shape.data[1] * out_d2xd3xd4xd5;
    const int out_d2xd3xd4      = out_d2xd3 * output_shape.data[4];

    const int out_batch   = image_height_idx / out_d2xd3;
    const int out_channel = image_width_idx / out_d4xd5;
    const int out_d2_d3   = image_height_idx % out_d2xd3;
    const int out_d4_d5   = image_width_idx % out_d4xd5;
    const int out_d2      = out_d2_d3 / output_shape.data[3];
    const int out_d3      = out_d2_d3 % output_shape.data[3];
    const int out_d4      = out_d4_d5 / output_shape.data[5];
    const int out_d5      = out_d4_d5 % output_shape.data[5];

    int index = out_batch * out_cxd2xd3xd4xd5 + out_channel * 4 * out_d2xd3xd4xd5 + out_d2 * out_d3xd4xd5 +
                out_d3 * out_d4xd5 + out_d4 * output_shape.data[5] + out_d5;

    const int index_div_out_d2xd3xd4xd5 = index / out_d2xd3xd4xd5;
    const int index_div_out_d3xd4xd5    = index / out_d3xd4xd5;
    const int index_div_out_d4xd5       = index / out_d4xd5;

    // compute four position
    // 0 index
    int in_batch          = index / out_cxd2xd3xd4xd5 % input_shape.data[0];
    int in_channel        = index_div_out_d2xd3xd4xd5 % input_shape.data[1];
    int in_d2             = index_div_out_d3xd4xd5 % input_shape.data[2];
    int in_d3             = index_div_out_d4xd5 % input_shape.data[3];
    int in_d4             = index / input_shape.data[5] % input_shape.data[4];
    int in_d5             = index % input_shape.data[5];
    int ix                = (in_channel >> 2) * in_d4xd5 + in_d4 * input_shape.data[5] + in_d5;
    int iy                = in_batch * in_d2xd3 + in_d2 * input_shape.data[3] + in_d3;
    FLOAT4 in             = RI_F(input, SAMPLER, (int2)(ix, iy));
    int in_channel_remain = in_channel % 4;
    FLOAT out0 =
        in_channel_remain == 0 ? in.x : (in_channel_remain == 1 ? in.y : (in_channel_remain == 2 ? in.z : in.w));

    // 1 index
    FLOAT out1 = 0.0f;
    if ((out_channel * 4 + 1) < output_shape.data[1]) {
        index             = index + out_d2xd3xd4xd5;
        in_batch          = index / out_cxd2xd3xd4xd5 % input_shape.data[0];
        in_channel        = (index_div_out_d2xd3xd4xd5 + 1) % input_shape.data[1];
        in_d2             = (index_div_out_d3xd4xd5 + output_shape.data[2]) % input_shape.data[2];
        in_d3             = (index_div_out_d4xd5 + out_d2xd3) % input_shape.data[3];
        in_d4             = index / input_shape.data[5] % input_shape.data[4];
        in_d5             = index % input_shape.data[5];
        ix                = (in_channel >> 2) * in_d4xd5 + in_d4 * input_shape.data[5] + in_d5;
        iy                = in_batch * in_d2xd3 + in_d2 * input_shape.data[3] + in_d3;
        in                = RI_F(input, SAMPLER, (int2)(ix, iy));
        in_channel_remain = in_channel % 4;
        out1 = in_channel_remain == 0 ? in.x : (in_channel_remain == 1 ? in.y : (in_channel_remain == 2 ? in.z : in.w));
    }

    // 2 index
    FLOAT out2 = 0.0f;
    if ((out_channel * 4 + 2) < output_shape.data[1]) {
        index             = index + out_d2xd3xd4xd5;
        in_batch          = index / out_cxd2xd3xd4xd5 % input_shape.data[0];
        in_channel        = (index_div_out_d2xd3xd4xd5 + 2) % input_shape.data[1];
        in_d2             = (index_div_out_d3xd4xd5 + output_shape.data[2] * 2) % input_shape.data[2];
        in_d3             = (index_div_out_d4xd5 + out_d2xd3 * 2) % input_shape.data[3];
        in_d4             = index / input_shape.data[5] % input_shape.data[4];
        in_d5             = index % input_shape.data[5];
        ix                = (in_channel >> 2) * in_d4xd5 + in_d4 * input_shape.data[5] + in_d5;
        iy                = in_batch * in_d2xd3 + in_d2 * input_shape.data[3] + in_d3;
        in                = RI_F(input, SAMPLER, (int2)(ix, iy));
        in_channel_remain = in_channel % 4;
        out2 = in_channel_remain == 0 ? in.x : (in_channel_remain == 1 ? in.y : (in_channel_remain == 2 ? in.z : in.w));
    }

    // 3 index
    FLOAT out3 = 0.0f;
    if ((out_channel * 4 + 3) < output_shape.data[1]) {
        index             = index + out_d2xd3xd4xd5;
        in_batch          = index / out_cxd2xd3xd4xd5 % input_shape.data[0];
        in_channel        = (index_div_out_d2xd3xd4xd5 + 3) % input_shape.data[1];
        in_d2             = (index_div_out_d3xd4xd5 + output_shape.data[2] * 3) % input_shape.data[2];
        in_d3             = (index_div_out_d4xd5 + out_d2xd3 * 3) % input_shape.data[3];
        in_d4             = index / input_shape.data[5] % input_shape.data[4];
        in_d5             = index % input_shape.data[5];
        ix                = (in_channel >> 2) * in_d4xd5 + in_d4 * input_shape.data[5] + in_d5;
        iy                = in_batch * in_d2xd3 + in_d2 * input_shape.data[3] + in_d3;
        in                = RI_F(input, SAMPLER, (int2)(ix, iy));
        in_channel_remain = in_channel % 4;
        out3 = in_channel_remain == 0 ? in.x : (in_channel_remain == 1 ? in.y : (in_channel_remain == 2 ? in.z : in.w));
    }

    FLOAT4 out = (FLOAT4)(out0, out1, out2, out3);
    WI_F(output, (int2)(image_width_idx, image_height_idx), out);
}