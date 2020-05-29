#include "base.inc"

__kernel void StrideSliceC4Unite(
                    __read_only image2d_t input, 
                    __write_only image2d_t output,
                    //[n,c,h,w]
                    int4 begins,
                    int4 strides,
                    int2 input_wh,
                    int2 output_wh
                    ) {
    int w = output_wh.x;
    int h = output_wh.y;    
    int2 xy = (int2)(get_global_id(0), get_global_id(1));
    //N, C, H, W
    int4 posOutput = (int4)(xy.y/h, xy.x/w, xy.y%h, xy.x%w);
    int4 posInput = mul24(posOutput, strides) + begins;

    int2 outputPos = (int2)(posOutput.w + posOutput.y*output_wh.x, posOutput.x*output_wh.y + posOutput.z);
    int2 inputPos = (int2)(posInput.w + posInput.y*input_wh.x, posInput.x*input_wh.y + posInput.z);

    WI_F(output, outputPos, RI_F(input, SAMPLER, inputPos));
}

__kernel void StrideSliceC4Separate(
                    GLOBAL_SIZE_2_DIMS
                    __global FLOAT *input, 
                    __write_only image2d_t output,
                    //[n,c,h,w]
                    int4 begins,
                    int4 strides,
                    int input_w_size,
                    int input_hw_size,
                    int input_chw_size,
                    int input_channel,
                    int2 output_wh,
                    int output_channel
                    ) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    int w = output_wh.x;
    int h = output_wh.y;
    //N, C, H, W
    int4 posOutput = (int4)(image_height_idx/h, (image_width_idx/w) << 2,
                              image_height_idx%h, image_width_idx%w);
    int4 posInput = mul24(posOutput, strides) + begins;

    int2 outputPos = (int2)(image_width_idx, image_height_idx);
    
    int pos = mul24(posInput.x, input_chw_size) + mul24(posInput.y, input_hw_size) + 
                  mul24(posInput.z, input_w_size) + posInput.w;

    int pos_channel_stride = mul24(strides.y , input_hw_size);

    const int remain_channel = output_channel - posOutput.y;

    FLOAT4 value = 0;
    if (remain_channel >= 4) {
        value.x = input[pos];
        value.y = input[pos + pos_channel_stride];
        value.z = input[pos + mul24(2, pos_channel_stride)];
        value.w = input[pos + mul24(3, pos_channel_stride)];
    } else if (remain_channel == 3) {
        value.x = input[pos];
        value.y = input[pos + pos_channel_stride];
        value.z = input[pos + mul24(2, pos_channel_stride)];
    } else if (remain_channel == 2) {
        value.x = input[pos];
        value.y = input[pos + pos_channel_stride];
    } else if (remain_channel == 1) {
        value.x = input[pos];
    }

    WI_F(output, outputPos, value);
}
