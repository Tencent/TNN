#include "base.inc"

__kernel void CopyImage(GLOBAL_SIZE_2_DIMS  
                    __read_only image2d_t input, 
                    __write_only image2d_t output,
                    int4 input_offset,
                    int4 output_offset,
                    int2 input_wh,
                    int2 output_wh,
                    int2 wh
                    ) {
    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);

    //N, C, H, W
    int4 pos = (int4)(bh/wh.y, cw/wh.x, bh%wh.y, cw%wh.x);

    int4 pos_input = input_offset + pos;
    int4 pos_output = output_offset + pos;

    int2 output_pos = (int2)(pos_output.w + pos_output.y*output_wh.x, pos_output.x*output_wh.y + pos_output.z);
    int2 input_pos = (int2)(pos_input.w + pos_input.y*input_wh.x, pos_input.x*input_wh.y + pos_input.z);

    WI_F(output, output_pos, RI_F(input, SAMPLER, input_pos));
}

__kernel void CopyImage5D(GLOBAL_SIZE_2_DIMS  
                    __read_only image2d_t input, 
                    __write_only image2d_t output,
                    shape_5d input_offset,
                    shape_3d output_inner_size, // dims2, dims3, dims4
                    shape_3d input_inner_size // dim2, dim3, dim4
                    ) {
    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);

    int dim2 = output_inner_size.data[0];
    int dim3 = output_inner_size.data[1];
    int dim4 = output_inner_size.data[2];

    const int dim3_idx                  = bh % dim3;
    const int batch_dim2_idx            = bh / dim3;
    const int dim2_idx                  = batch_dim2_idx % dim2;
    const int batch_idx                 = batch_dim2_idx / dim2;
    const int dim4_idx                  = cw % dim4;
    const int channel_updiv_4_idx       = cw / dim4;

    const int input_batch_idx           = batch_idx + input_offset.data[0];
    const int input_channel_updiv_4_idx = channel_updiv_4_idx + input_offset.data[1];
    const int input_dim2_idx            = dim2_idx + input_offset.data[2];
    const int input_dim3_idx            = dim3_idx + input_offset.data[3];
    const int input_dim4_idx            = dim4_idx + input_offset.data[4];

    int2 input_pos = (int2)(mad24(input_channel_updiv_4_idx, input_inner_size.data[2], input_dim4_idx), 
                      mad24(mad24(input_batch_idx, input_inner_size.data[0], input_dim2_idx), input_inner_size.data[1], input_dim3_idx));

    WI_F(output, (int2)(cw, bh), RI_F(input, SAMPLER, input_pos));
}

__kernel void Crop(GLOBAL_SIZE_2_DIMS  
                    __read_only image2d_t input, 
                    __write_only image2d_t output,
                    int start_x,
                    int start_y,
                    int crop_width,
                    int crop_height,
                    int src_width,
                    int src_height
                    ) {
    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);
    const int batch_idx         = bh / crop_height;
    const int height_idx        = bh % crop_height;

    int2 output_pos = (int2)(cw , height_idx + crop_height*batch_idx);
    int2 input_pos  = (int2)(cw + start_x, height_idx + src_height*batch_idx + start_y);
    WI_F(output, output_pos, RI_F(input, SAMPLER, input_pos));
}

__kernel void CopyMakeBorder(GLOBAL_SIZE_2_DIMS
                             __read_only image2d_t input,
                             __write_only image2d_t output,
                             int top,
                             int left,
                             int src_width,
                             int src_height,
                             int src_channel_blocks,
                             int dst_height,
                             __private const float border_val
                             ) {
    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);
    const int batch_idx         = bh / dst_height;
    const int height_idx        = bh % dst_height;
    const int width_idx         = cw / src_channel_blocks;
    const int channel_4_idx     = cw % src_channel_blocks;
    const int in_width_idx      = width_idx - left;
    const int in_height_idx     = height_idx - top;

    FLOAT4 out = border_val;
    int2 output_pos = (int2)(cw, bh);

    if (in_width_idx >= 0 && in_width_idx < src_width &&  in_height_idx >= 0 && in_height_idx < src_height) {
        int2 input_pos  = (int2)(channel_4_idx * src_width + in_width_idx, in_height_idx + batch_idx * src_height);
        out = RI_F(input, SAMPLER, input_pos);
    }

    WI_F(output, output_pos, out);
}

__kernel void CopyBuffer(GLOBAL_SIZE_2_DIMS  
                    const __global FLOAT *input, 
                    __global FLOAT *output,
                    int4 input_offset,
                    int4 output_offset,
                    int4 input_stride,
                    int4 output_stride,
                    int2 wh
                    ) {
    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);
    //N, C, H, W
    int4 pos = (int4)(bh/wh.y, cw/wh.x, bh%wh.y, cw%wh.x);

    int4 pos_input = input_offset + pos;
    int4 pos_output = output_offset + pos;

    int output_pos = pos_output.x * output_stride.x
        + pos_output.y * output_stride.y
        + pos_output.z * output_stride.z
        + pos_output.w * output_stride.w;
    
    int input_pos = pos_input.x * input_stride.x
        + pos_input.y * input_stride.y
        + pos_input.z * input_stride.z
        + pos_input.w * input_stride.w;
    
    output[output_pos] = input[input_pos];
}

__kernel void CopyImageToBuffer(GLOBAL_SIZE_2_DIMS  
                    __read_only image2d_t input, 
                    __global FLOAT *output,
                    int4 input_offset,
                    int4 output_offset,
                    int2 input_wh,
                    int4 output_stride,
                    int4 output_size
                    ) {
    int c = output_size.y;
    int h = output_size.z;
    int w = output_size.w;

    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);
    //N, C, H, W
    int4 pos = (int4)(bh/h, cw/w, bh%h, cw%w);
    int4 bufferPos = pos * (int4)(1, 4, 1, 1);

    int4 pos_input = input_offset + pos;
    int4 pos_output = output_offset + bufferPos;
    int2 input_pos = (int2)(pos_input.w + pos_input.y*input_wh.x, pos_input.x*input_wh.y + pos_input.z);

    FLOAT4 color = RI_F(input, SAMPLER, input_pos);

    int output_pos_basic = pos_output.x*output_stride.x
            + pos_output.y*output_stride.y
            + pos_output.z*output_stride.z
            + pos_output.w*output_stride.w;

    int output_pos_0 = output_pos_basic + 0*output_stride.y;
    output[output_pos_0] = color.x;
    if (pos_output.y + 1 < c) {
        int output_pos_1 = output_pos_basic + 1*output_stride.y;
        output[output_pos_1] = color.y;
    }
    if (pos_output.y + 2 < c) {
        int output_pos_1 = output_pos_basic + 2*output_stride.y;
        output[output_pos_1] = color.z;
    }
    if (pos_output.y + 3 < c) {
        int output_pos_1 = output_pos_basic + 3*output_stride.y;
        output[output_pos_1] = color.w;
    }
}

__kernel void CopyImage5DToBuffer(GLOBAL_SIZE_2_DIMS
                    __read_only image2d_t input,
                    __global FLOAT *output,
                    shape_5d input_offset,
                    shape_5d output_offset,
                    shape_3d input_size, // dim2 dim3 dim4
                    shape_5d output_stride,
                    shape_5d output_size
                    ) {
    int image_width_idx = get_global_id(0);
    int image_height_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    int channels = output_size.data[1];
    int dim2 = output_size.data[2];
    int dim3 = output_size.data[3];
    int dim4 = output_size.data[4];

    const int dim3_idx                  = image_height_idx % dim3;
    const int batch_dim2_idx            = image_height_idx / dim3;
    const int dim2_idx                  = batch_dim2_idx % dim2;
    const int batch_idx                 = batch_dim2_idx / dim2;
    const int dim4_idx                  = image_width_idx % dim4;
    const int channel_updiv_4_idx       = image_width_idx / dim4;

    const int input_batch_idx           = batch_idx + input_offset.data[0];
    const int input_channel_updiv_4_idx = channel_updiv_4_idx + input_offset.data[1];
    const int input_dim2_idx            = dim2_idx + input_offset.data[2];
    const int input_dim3_idx            = dim3_idx + input_offset.data[3];
    const int input_dim4_idx            = dim4_idx + input_offset.data[4];

    const int output_batch_idx          = batch_idx + output_offset.data[0];
    const int output_channel_4_idx      = (channel_updiv_4_idx << 2) + output_offset.data[1];
    const int output_dim2_idx           = dim2_idx + output_offset.data[2];
    const int output_dim3_idx           = dim3_idx + output_offset.data[3];
    const int output_dim4_idx           = dim4_idx + output_offset.data[4];

    int2 input_pos = (int2)(input_dim4_idx + input_channel_updiv_4_idx * input_size.data[2],
        (input_batch_idx * input_size.data[0] + input_dim2_idx) * input_size.data[1] + input_dim3_idx);

    FLOAT4 out = RI_F(input, SAMPLER, input_pos);

    int output_pos_basic = output_batch_idx * output_stride.data[0]
            + output_channel_4_idx * output_stride.data[1]
            + output_dim2_idx * output_stride.data[2]
            + output_dim3_idx * output_stride.data[3]
            + output_dim4_idx * output_stride.data[4];

    int remain = channels - output_channel_4_idx;
    int output_pos = output_pos_basic;
    if (remain >= 4) {
        output[output_pos] = out.x;
        output_pos += output_stride.data[1];
        output[output_pos] = out.y;
        output_pos += output_stride.data[1];
        output[output_pos] = out.z;
        output_pos += output_stride.data[1];
        output[output_pos] = out.w;
    } else if (remain == 3) {
        output[output_pos] = out.x;
        output_pos += output_stride.data[1];
        output[output_pos] = out.y;
        output_pos += output_stride.data[1];
        output[output_pos] = out.z;
    } else if (remain == 2) {
        output[output_pos] = out.x;
        output_pos += output_stride.data[1];
        output[output_pos] = out.y;
    } else {
        output[output_pos] = out.x;
    }
}

__kernel void CopyImage6DToBuffer(GLOBAL_SIZE_2_DIMS
                    __read_only image2d_t input,
                    __global FLOAT *output,
                    shape_6d input_offset,
                    shape_6d output_offset,
                    shape_4d input_size, // dim2 dim3 dim4 dim5
                    shape_6d output_stride,
                    shape_6d output_size
                    ) {
    int image_width_idx = get_global_id(0);
    int image_height_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    int channels = output_size.data[1];
    int dim2 = output_size.data[2];
    int dim3 = output_size.data[3];
    int dim4 = output_size.data[4];
    int dim5 = output_size.data[5];

    const int dim3_idx                  = image_height_idx % dim3;
    const int batch_dim2_idx            = image_height_idx / dim3;
    const int dim2_idx                  = batch_dim2_idx % dim2;
    const int batch_idx                 = batch_dim2_idx / dim2;
    const int dim5_idx                  = image_width_idx % dim5;
    const int channel_updiv_4_dim4_idx  = image_width_idx / dim5;
    const int dim4_idx                  = channel_updiv_4_dim4_idx % dim4;
    const int channel_updiv_4_idx       = channel_updiv_4_dim4_idx / dim4;

    const int input_batch_idx           = batch_idx + input_offset.data[0];
    const int input_channel_updiv_4_idx = channel_updiv_4_idx + input_offset.data[1];
    const int input_dim2_idx            = dim2_idx + input_offset.data[2];
    const int input_dim3_idx            = dim3_idx + input_offset.data[3];
    const int input_dim4_idx            = dim4_idx + input_offset.data[4];
    const int input_dim5_idx            = dim5_idx + input_offset.data[5];

    const int output_batch_idx          = batch_idx + output_offset.data[0];
    const int output_channel_4_idx      = (channel_updiv_4_idx << 2) + output_offset.data[1];
    const int output_dim2_idx           = dim2_idx + output_offset.data[2];
    const int output_dim3_idx           = dim3_idx + output_offset.data[3];
    const int output_dim4_idx           = dim4_idx + output_offset.data[4];
    const int output_dim5_idx           = dim5_idx + output_offset.data[5];

    int2 input_pos = (int2)(
        (input_dim4_idx + input_channel_updiv_4_idx * input_size.data[2]) * input_size.data[3] + input_dim5_idx,
        (input_batch_idx * input_size.data[0] + input_dim2_idx) * input_size.data[1] + input_dim3_idx);

    FLOAT4 out = RI_F(input, SAMPLER, input_pos);

    int output_pos_basic = output_batch_idx * output_stride.data[0]
            + output_channel_4_idx * output_stride.data[1]
            + output_dim2_idx * output_stride.data[2]
            + output_dim3_idx * output_stride.data[3]
            + output_dim4_idx * output_stride.data[4]
            + output_dim5_idx * output_stride.data[5];

    int remain = channels - output_channel_4_idx;
    int output_pos = output_pos_basic;
    if (remain >= 4) {
        output[output_pos] = out.x;
        output_pos += output_stride.data[1];
        output[output_pos] = out.y;
        output_pos += output_stride.data[1];
        output[output_pos] = out.z;
        output_pos += output_stride.data[1];
        output[output_pos] = out.w;
    } else if (remain == 3) {
        output[output_pos] = out.x;
        output_pos += output_stride.data[1];
        output[output_pos] = out.y;
        output_pos += output_stride.data[1];
        output[output_pos] = out.z;
    } else if (remain == 2) {
        output[output_pos] = out.x;
        output_pos += output_stride.data[1];
        output[output_pos] = out.y;
    } else {
        output[output_pos] = out.x;
    }
}

__kernel void CopyBufferToImage(GLOBAL_SIZE_2_DIMS  
                    __global FLOAT *input,
                    __write_only image2d_t output,
                    int4 input_offset,
                    int4 output_offset,
                    int4 input_stride,
                    int2 output_wh,
                    int2 wh,
                    int buffer_end
                    ) {
    int cw = get_global_id(0);
    int bh = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw, bh);
    //N, C, H, W
    int4 pos = (int4)(bh/wh.y, cw/wh.x, bh%wh.y, cw%wh.x);
    int4 bufferPos = pos * (int4)(1, 4, 1, 1);

    int4 pos_input = input_offset + bufferPos;
    int4 pos_output = output_offset + pos;
    int2 output_pos = (int2)(pos_output.w + pos_output.y*output_wh.x, pos_output.x*output_wh.y + pos_output.z);
    int input_pos_basic = pos_input.x*input_stride.x 
            +pos_input.y*input_stride.y
            +pos_input.z*input_stride.z
            +pos_input.w*input_stride.w;

    int input_pos_0 = input_pos_basic;
    int input_pos_1 = clamp(input_pos_basic + 1*input_stride.y, 0, buffer_end);
    int input_pos_2 = clamp(input_pos_basic + 2*input_stride.y, 0, buffer_end);
    int input_pos_3 = clamp(input_pos_basic + 3*input_stride.y, 0, buffer_end);

    FLOAT4 color;
    color.x = input[input_pos_0];
    color.y = input[input_pos_1];
    color.z = input[input_pos_2];
    color.w = input[input_pos_3];

    WI_F(output, output_pos, color);
}

__kernel void CopyBufferToImage5D(GLOBAL_SIZE_2_DIMS
                    __global FLOAT *input,
                    __write_only image2d_t output,
                    shape_5d input_offset,
                    shape_5d output_offset,
                    shape_5d input_stride,
                    shape_3d output_size, // dim2 dim3 dim4
                    shape_3d input_size, // dim2 dim3 dim4
                    int buffer_end
                    ) {
    int image_width_idx = get_global_id(0);
    int image_height_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    int dim2 = input_size.data[0];
    int dim3 = input_size.data[1];
    int dim4 = input_size.data[2];

    const int dim3_idx                  = image_height_idx % dim3;
    const int batch_dim2_idx            = image_height_idx / dim3;
    const int dim2_idx                  = batch_dim2_idx % dim2;
    const int batch_idx                 = batch_dim2_idx / dim2;
    const int dim4_idx                  = image_width_idx % dim4;
    const int channel_updiv_4_idx       = image_width_idx / dim4;

    const int input_batch_idx           = batch_idx + input_offset.data[0];
    const int input_channel_updiv_4_idx = (channel_updiv_4_idx << 2) + input_offset.data[1];
    const int input_dim2_idx            = dim2_idx + input_offset.data[2];
    const int input_dim3_idx            = dim3_idx + input_offset.data[3];
    const int input_dim4_idx            = dim4_idx + input_offset.data[4];

    const int output_batch_idx          = batch_idx + output_offset.data[0];
    const int output_channel_4_idx      = channel_updiv_4_idx + output_offset.data[1];
    const int output_dim2_idx           = dim2_idx + output_offset.data[2];
    const int output_dim3_idx           = dim3_idx + output_offset.data[3];
    const int output_dim4_idx           = dim4_idx + output_offset.data[4];

    int2 output_pos = (int2)(output_dim4_idx + output_channel_4_idx * output_size.data[2],
        (output_batch_idx * output_size.data[0] + output_dim2_idx) * output_size.data[1] + output_dim3_idx);

    int input_pos_basic = input_batch_idx * input_stride.data[0]
            + input_channel_updiv_4_idx * input_stride.data[1]
            + input_dim2_idx * input_stride.data[2]
            + input_dim3_idx * input_stride.data[3]
            + input_dim4_idx * input_stride.data[4];

    int input_pos_0 = input_pos_basic;
    int input_pos_1 = clamp(input_pos_basic + 1 * input_stride.data[1], 0, buffer_end);
    int input_pos_2 = clamp(input_pos_basic + 2 * input_stride.data[1], 0, buffer_end);
    int input_pos_3 = clamp(input_pos_basic + 3 * input_stride.data[1], 0, buffer_end);

    FLOAT4 color;
    color.x = input[input_pos_0];
    color.y = input[input_pos_1];
    color.z = input[input_pos_2];
    color.w = input[input_pos_3];

    WI_F(output, output_pos, color);
}


__kernel void CopyBufferToImage6D(GLOBAL_SIZE_2_DIMS
                    __global FLOAT *input,
                    __write_only image2d_t output,
                    shape_6d input_offset,
                    shape_6d output_offset,
                    shape_6d input_stride,
                    shape_4d output_size, // dim2 dim3 dim4 dim5
                    shape_4d input_size, // dim2 dim3 dim4 dim5
                    int buffer_end
                    ) {
    int image_width_idx = get_global_id(0);
    int image_height_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    int dim2 = input_size.data[0];
    int dim3 = input_size.data[1];
    int dim4 = input_size.data[2];
    int dim5 = input_size.data[3];

    const int dim3_idx                  = image_height_idx % dim3;
    const int batch_dim2_idx            = image_height_idx / dim3;
    const int dim2_idx                  = batch_dim2_idx % dim2;
    const int batch_idx                 = batch_dim2_idx / dim2;
    const int dim5_idx                  = image_width_idx % dim5;
    const int channel_updiv_4_dim4_idx  = image_width_idx / dim5;
    const int dim4_idx                  = channel_updiv_4_dim4_idx % dim4;
    const int channel_updiv_4_idx       = channel_updiv_4_dim4_idx / dim4;

    const int input_batch_idx           = batch_idx + input_offset.data[0];
    const int input_channel_updiv_4_idx = (channel_updiv_4_idx << 2) + input_offset.data[1];
    const int input_dim2_idx            = dim2_idx + input_offset.data[2];
    const int input_dim3_idx            = dim3_idx + input_offset.data[3];
    const int input_dim4_idx            = dim4_idx + input_offset.data[4];
    const int input_dim5_idx            = dim5_idx + input_offset.data[5];

    const int output_batch_idx          = batch_idx + output_offset.data[0];
    const int output_channel_4_idx      = channel_updiv_4_idx + output_offset.data[1];
    const int output_dim2_idx           = dim2_idx + output_offset.data[2];
    const int output_dim3_idx           = dim3_idx + output_offset.data[3];
    const int output_dim4_idx           = dim4_idx + output_offset.data[4];
    const int output_dim5_idx           = dim5_idx + output_offset.data[5];

    int2 output_pos = (int2)(
        (output_dim4_idx + output_channel_4_idx * output_size.data[2]) * output_size.data[3] + output_dim5_idx,
        (output_batch_idx * output_size.data[0] + output_dim2_idx) * output_size.data[1] + output_dim3_idx);

    int input_pos_basic = input_batch_idx * input_stride.data[0]
            + input_channel_updiv_4_idx * input_stride.data[1]
            + input_dim2_idx * input_stride.data[2]
            + input_dim3_idx * input_stride.data[3]
            + input_dim4_idx * input_stride.data[4]
            + input_dim5_idx * input_stride.data[5];

    int input_pos_0 = input_pos_basic;
    int input_pos_1 = clamp(input_pos_basic + 1 * input_stride.data[1], 0, buffer_end);
    int input_pos_2 = clamp(input_pos_basic + 2 * input_stride.data[1], 0, buffer_end);
    int input_pos_3 = clamp(input_pos_basic + 3 * input_stride.data[1], 0, buffer_end);

    FLOAT4 color;
    color.x = input[input_pos_0];
    color.y = input[input_pos_1];
    color.z = input[input_pos_2];
    color.w = input[input_pos_3];

    WI_F(output, output_pos, color);
}
