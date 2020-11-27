#include "base.inc"

__kernel void ConcatChannel(
                             GLOBAL_SIZE_3_DIMS
                             __read_only image2d_t input0,
                             __read_only image2d_t input1,
                             __private const int input0_channel,
                             __private const int output_channel,
                             __write_only image2d_t output) {
  const int channel_block_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

  DEAL_NON_UNIFORM_DIM3(channel_block_idx, width_idx, hb_idx);

  const int width = global_size_dim1;
  const int input1_channel = output_channel - input0_channel;

  const int input0_channel_blk = (input0_channel + 3) >> 2;

  FLOAT4 data = 0;
  if (channel_block_idx < input0_channel_blk - 1) {
    data = RI_F(input0, SAMPLER, (int2)(mad24(channel_block_idx, width, width_idx), hb_idx));
  } else if(channel_block_idx == input0_channel_blk - 1) {
    FLOAT4 data0 = RI_F(input0, SAMPLER, (int2)(mad24(channel_block_idx, width, width_idx), hb_idx));
    FLOAT4 data1 = RI_F(input1, SAMPLER, (int2)(width_idx, hb_idx));
#if CHANNEL0_MOD_4 == 1
    data = (FLOAT4)(data0.s0, data1.s0, data1.s1, data1.s2);
#elif CHANNEL0_MOD_4 == 2
    data = (FLOAT4)(data0.s0, data0.s1, data1.s0, data1.s1);
#else
    data = (FLOAT4)(data0.s0, data0.s1, data0.s2, data1.s0);
#endif
  } else {
    const int input1_channel_idx = channel_block_idx - input0_channel_blk;
    FLOAT4 data0 = RI_F(input1, SAMPLER, (int2)(mad24(input1_channel_idx, width, width_idx), hb_idx));
    FLOAT4 data1 = 0;
    if (((input1_channel_idx + 1) << 2) < input1_channel) {
      data1 = RI_F(input1, SAMPLER, (int2)(mad24((input1_channel_idx + 1), width, width_idx), hb_idx));
    }
#if CHANNEL0_MOD_4 == 1
    data = (FLOAT4)(data0.s3, data1.s0, data1.s1, data1.s2);
#elif CHANNEL0_MOD_4 == 2
    data = (FLOAT4)(data0.s2, data0.s3, data1.s0, data1.s1);
#else
    data = (FLOAT4)(data0.s1, data0.s2, data0.s3, data1.s0);
#endif
  } 
  
  const int pos = mad24(channel_block_idx, width, width_idx);
  WI_F(output, (int2)(pos, hb_idx), data);
}

__kernel void ConcatChannel4X(
                             GLOBAL_SIZE_3_DIMS
                             __read_only image2d_t input0,
                             __read_only image2d_t input1,
                             __private const int input0_channel,
                             __private const int output_channel,
                             __write_only image2d_t output) {
  const int channel_block_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

  DEAL_NON_UNIFORM_DIM3(channel_block_idx, width_idx, hb_idx);

  const int width = global_size_dim1;
  const int input0_channel_blk = input0_channel >> 2;

  FLOAT4 data = 0;
  if (channel_block_idx < input0_channel_blk) {
    data = RI_F(input0, SAMPLER, (int2)(mad24(channel_block_idx, width, width_idx), hb_idx));
  } else {
    const int input1_channel_idx = channel_block_idx - input0_channel_blk;
    data = RI_F(input1, SAMPLER, (int2)(mad24(input1_channel_idx, width, width_idx), hb_idx));
  } 
  
  WI_F(output, (int2)(mad24(channel_block_idx, width, width_idx), hb_idx), data);
}
