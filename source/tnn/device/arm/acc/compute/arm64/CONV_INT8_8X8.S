#ifdef __aarch64__

#include "tnn/device/arm/acc/compute/asm_func_name.S"

.text
.align 5

asm_function ASMConvInt8Unit8x8
//void ConvInt8Unit8x8(int32_t mr, int32_t nr, int32_t input_channel, int32_t kernel_size, const int32_t* indirect,
//                                            const void* weight, int8_t* output, int32_t channel_stride,
//                                            const float* scales, long relu, const int8_t* add_input,
//                                            const float* add_scale, const int8_t* zero, const int8_t* real_input)
//x0(int32_t mr), 8
//x1(int32_t nr), 1
//x2(int32_t input_channel),9
//x3(int32_t kernel_size),9
//x4(int32* indirect),
//x5(void* weight),
//x6(int8_t* output),
//x7(int32_t channel_stride),
//x8 from stack(float* scales), sp+0
//x9 from stack(long relu), sp+8
//x10 from stack(int8_t* add_input), sp+16
//x11 from stack(float* add_scale),sp+24
//x12 from stack(int8_t* zero) , sp+32
//x13 from stack(int8_t* real_input), sp+40
//保存堆栈信息，用于函数返回时，弹出堆栈
sub sp, sp, #176
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
stp x19, x20, [sp], #32
str x21, [sp], #16

ldr x8, [sp, #0]    //x8 scale float

//load bias
// 4s = int32x4_t
ld1 {v8.4s, v9.4s}, [x5], #32   // x5 weight void*
mov v10.16b, v8.16b             // v8: vacc0x0123 v9: vacc0x4567
mov v11.16b, v9.16b
mov v12.16b, v8.16b
mov v13.16b, v9.16b
mov v14.16b, v8.16b
mov v15.16b, v9.16b
mov v16.16b, v8.16b
mov v17.16b, v9.16b
mov v18.16b, v8.16b
mov v19.16b, v9.16b
mov v20.16b, v8.16b
mov v21.16b, v9.16b
mov v22.16b, v8.16b
mov v23.16b, v9.16b

ldr x19, [sp, #32] // int8* zero
ldr x20, [sp, #40] // int8* real_input
0:
    ldr w9,  [x4], #4   //a0 ,x4-> int32*
    cmp w9,  #-1
    sxtw x9, w9
    add x21, x20, x9
    csel x9, x19, x21, eq // x9 代表 a0

    ldr w10, [x4], #4   //a1
    cmp w10,  #-1
    sxtw x10, w10
    add x21, x20, x10
    csel x10, x19, x21, eq

    ldr w11, [x4], #4   //a2
    cmp w11, #-1
    sxtw x11, w11
    add x21, x20, x11
    csel x11, x19, x21, eq

    ldr w12, [x4], #4   //a3
    cmp w12, #-1
    sxtw x12, w12
    add x21, x20, x12
    csel x12, x19, x21, eq

    ldr w13, [x4], #4   //a4
    cmp w13, #-1
    sxtw x13, w13
    add x21, x20, x13
    csel x13, x19, x21, eq


    ldr w14, [x4], #4   //a5
    cmp w14, #-1
    sxtw x14, w14
    add x21, x20, x14
    csel x14, x19, x21, eq

    ldr w15, [x4], #4   //a6
    cmp w15, #-1
    sxtw x15, w15
    add x21, x20, x15
    csel x15, x19, x21, eq

    ldr w16, [x4], #4   //a7
    cmp w16, #-1
    sxtw x16, w16
    add x21, x20, x16
    csel x16, x19, x21, eq

    subs x17, x2, #8    // x2: input channel
    blt 2f              // if x2 < 8, jump 2f
1:
    // load input from indirect buffer
    // x5 : weight
    // 8b : int8x8_t
    ld1 {v27.8b}, [x5], #8 // load weight
    sxtl v27.8h, v27.8b     // weight int8x8_t -> int16x8_t
    // load indirect buffer
    ld1 {v0.8b}, [x9], #8   // va0 int8x8_t -> int16x8_t
    sxtl v0.8h, v0.8b
    ld1 {v1.8b}, [x10], #8  //va1
    sxtl v1.8h, v1.8b
    ld1 {v2.8b}, [x11], #8  //va2
    sxtl v2.8h, v2.8b
    ld1 {v3.8b}, [x12], #8  //va3
    sxtl v3.8h, v3.8b
    ld1 {v4.8b}, [x13], #8  //va4
    sxtl v4.8h, v4.8b
    ld1 {v5.8b}, [x14], #8  //va5
    sxtl v5.8h, v5.8b
    ld1 {v6.8b}, [x15], #8  //va6
    sxtl v6.8h, v6.8b
    ld1 {v7.8b}, [x16], #8  //va7
    sxtl v7.8h, v7.8b

    // COMPUTE_UNIT_LOW(0)
    ld1 {v28.8b}, [x5], #8
    smlal  v8.4s, v27.4h, v0.h[0] //vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb01234567), vget_low_s16(va0), i);
    smlal2 v9.4s, v27.8h, v0.h[0] //vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb01234567), vget_low_s16(va0), i);
    smlal  v10.4s, v27.4h, v1.h[0]
    smlal2 v11.4s, v27.8h, v1.h[0]
    smlal  v12.4s, v27.4h, v2.h[0]
    smlal2 v13.4s, v27.8h, v2.h[0]
    smlal  v14.4s, v27.4h, v3.h[0]
    smlal2 v15.4s, v27.8h, v3.h[0]
    sxtl v28.8h, v28.8b
    smlal  v16.4s, v27.4h, v4.h[0]
    smlal2 v17.4s, v27.8h, v4.h[0]
    smlal  v18.4s, v27.4h, v5.h[0]
    smlal2 v19.4s, v27.8h, v5.h[0]
    smlal  v20.4s, v27.4h, v6.h[0]
    smlal2 v21.4s, v27.8h, v6.h[0]
    smlal  v22.4s, v27.4h, v7.h[0]
    smlal2 v23.4s, v27.8h, v7.h[0]

    // COMPUTE_UNIT_LOW(1)
    ld1 {v27.8b}, [x5], #8
    smlal  v8.4s, v28.4h, v0.h[1]
    smlal2 v9.4s, v28.8h, v0.h[1]
    smlal  v10.4s, v28.4h, v1.h[1]
    smlal2 v11.4s, v28.8h, v1.h[1]
    smlal  v12.4s, v28.4h, v2.h[1]
    smlal2 v13.4s, v28.8h, v2.h[1]
    smlal  v14.4s, v28.4h, v3.h[1]
    smlal2 v15.4s, v28.8h, v3.h[1]
    sxtl v27.8h, v27.8b
    smlal  v16.4s, v28.4h, v4.h[1]
    smlal2 v17.4s, v28.8h, v4.h[1]
    smlal  v18.4s, v28.4h, v5.h[1]
    smlal2 v19.4s, v28.8h, v5.h[1]
    smlal  v20.4s, v28.4h, v6.h[1]
    smlal2 v21.4s, v28.8h, v6.h[1]
    smlal  v22.4s, v28.4h, v7.h[1]
    smlal2 v23.4s, v28.8h, v7.h[1]

    // COMPUTE_UNIT_LOW(2)
    ld1 {v28.8b}, [x5], #8
    smlal  v8.4s, v27.4h, v0.h[2]
    smlal2 v9.4s, v27.8h, v0.h[2]
    smlal  v10.4s, v27.4h, v1.h[2]
    smlal2 v11.4s, v27.8h, v1.h[2]
    smlal  v12.4s, v27.4h, v2.h[2]
    smlal2 v13.4s, v27.8h, v2.h[2]
    smlal  v14.4s, v27.4h, v3.h[2]
    smlal2 v15.4s, v27.8h, v3.h[2]
    sxtl v28.8h, v28.8b
    smlal  v16.4s, v27.4h, v4.h[2]
    smlal2 v17.4s, v27.8h, v4.h[2]
    smlal  v18.4s, v27.4h, v5.h[2]
    smlal2 v19.4s, v27.8h, v5.h[2]
    smlal  v20.4s, v27.4h, v6.h[2]
    smlal2 v21.4s, v27.8h, v6.h[2]
    smlal  v22.4s, v27.4h, v7.h[2]
    smlal2 v23.4s, v27.8h, v7.h[2]

    // COMPUTE_UNIT_LOW(3)
    ld1 {v27.8b}, [x5], #8
    smlal  v8.4s, v28.4h, v0.h[3]
    smlal2 v9.4s, v28.8h, v0.h[3]
    smlal  v10.4s, v28.4h, v1.h[3]
    smlal2 v11.4s, v28.8h, v1.h[3]
    smlal  v12.4s, v28.4h, v2.h[3]
    smlal2 v13.4s, v28.8h, v2.h[3]
    smlal  v14.4s, v28.4h, v3.h[3]
    smlal2 v15.4s, v28.8h, v3.h[3]
    sxtl v27.8h, v27.8b
    smlal  v16.4s, v28.4h, v4.h[3]
    smlal2 v17.4s, v28.8h, v4.h[3]
    smlal  v18.4s, v28.4h, v5.h[3]
    smlal2 v19.4s, v28.8h, v5.h[3]
    smlal  v20.4s, v28.4h, v6.h[3]
    smlal2 v21.4s, v28.8h, v6.h[3]
    smlal  v22.4s, v28.4h, v7.h[3]
    smlal2 v23.4s, v28.8h, v7.h[3]

    //c4
    ld1 {v28.8b}, [x5], #8
    smlal  v8.4s, v27.4h, v0.h[4]
    smlal2 v9.4s, v27.8h, v0.h[4]
    smlal  v10.4s, v27.4h, v1.h[4]
    smlal2 v11.4s, v27.8h, v1.h[4]
    smlal  v12.4s, v27.4h, v2.h[4]
    smlal2 v13.4s, v27.8h, v2.h[4]
    smlal  v14.4s, v27.4h, v3.h[4]
    smlal2 v15.4s, v27.8h, v3.h[4]
    sxtl v28.8h, v28.8b
    smlal  v16.4s, v27.4h, v4.h[4]
    smlal2 v17.4s, v27.8h, v4.h[4]
    smlal  v18.4s, v27.4h, v5.h[4]
    smlal2 v19.4s, v27.8h, v5.h[4]
    smlal  v20.4s, v27.4h, v6.h[4]
    smlal2 v21.4s, v27.8h, v6.h[4]
    smlal  v22.4s, v27.4h, v7.h[4]
    smlal2 v23.4s, v27.8h, v7.h[4]

    //c5
    ld1 {v27.8b}, [x5], #8
    smlal  v8.4s, v28.4h, v0.h[5]
    smlal2 v9.4s, v28.8h, v0.h[5]
    smlal  v10.4s, v28.4h, v1.h[5]
    smlal2 v11.4s, v28.8h, v1.h[5]
    smlal  v12.4s, v28.4h, v2.h[5]
    smlal2 v13.4s, v28.8h, v2.h[5]
    smlal  v14.4s, v28.4h, v3.h[5]
    smlal2 v15.4s, v28.8h, v3.h[5]
    sxtl v27.8h, v27.8b
    smlal  v16.4s, v28.4h, v4.h[5]
    smlal2 v17.4s, v28.8h, v4.h[5]
    smlal  v18.4s, v28.4h, v5.h[5]
    smlal2 v19.4s, v28.8h, v5.h[5]
    smlal  v20.4s, v28.4h, v6.h[5]
    smlal2 v21.4s, v28.8h, v6.h[5]
    smlal  v22.4s, v28.4h, v7.h[5]
    smlal2 v23.4s, v28.8h, v7.h[5]

    //c6
    ld1 {v28.8b}, [x5], #8
    smlal  v8.4s, v27.4h, v0.h[6]
    smlal2 v9.4s, v27.8h, v0.h[6]
    smlal  v10.4s, v27.4h, v1.h[6]
    smlal2 v11.4s, v27.8h, v1.h[6]
    smlal  v12.4s, v27.4h, v2.h[6]
    smlal2 v13.4s, v27.8h, v2.h[6]
    smlal  v14.4s, v27.4h, v3.h[6]
    smlal2 v15.4s, v27.8h, v3.h[6]
    sxtl v28.8h, v28.8b
    smlal  v16.4s, v27.4h, v4.h[6]
    smlal2 v17.4s, v27.8h, v4.h[6]
    smlal  v18.4s, v27.4h, v5.h[6]
    smlal2 v19.4s, v27.8h, v5.h[6]
    smlal  v20.4s, v27.4h, v6.h[6]
    smlal2 v21.4s, v27.8h, v6.h[6]
    smlal  v22.4s, v27.4h, v7.h[6]
    smlal2 v23.4s, v27.8h, v7.h[6]

    //c7
    subs  x17, x17, #8 // for(;;k-8);  1-8 = -7
    smlal  v8.4s, v28.4h, v0.h[7]
    smlal2 v9.4s, v28.8h, v0.h[7]
    smlal  v10.4s, v28.4h, v1.h[7]
    smlal2 v11.4s, v28.8h, v1.h[7]
    smlal  v12.4s, v28.4h, v2.h[7]
    smlal2 v13.4s, v28.8h, v2.h[7]
    smlal  v14.4s, v28.4h, v3.h[7]
    smlal2 v15.4s, v28.8h, v3.h[7]
    smlal  v16.4s, v28.4h, v4.h[7]
    smlal2 v17.4s, v28.8h, v4.h[7]
    smlal  v18.4s, v28.4h, v5.h[7]
    smlal2 v19.4s, v28.8h, v5.h[7]
    smlal  v20.4s, v28.4h, v6.h[7]
    smlal2 v21.4s, v28.8h, v6.h[7]
    smlal  v22.4s, v28.4h, v7.h[7]
    smlal2 v23.4s, v28.8h, v7.h[7]

    bge 1b // >= 0

2:
    cmp x17, #-8
    beq 3f              // IC 是 8 的倍数
    add x9, x9, x17     // a0 - a_predecrement
    add x10, x10, x17   // a1 - a_predecrement
    add x11, x11, x17
    add x12, x12, x17
    add x13, x13, x17
    add x14, x14, x17
    add x15, x15, x17
    add x16, x16, x17   // a7 - a_predecrement

    lsl x17, x17, #3    // x17 代表:va_shift = -56, x17 = x17*8 = -56
    fmov d29, x17       // 这一行没有问题

    ld1 {v0.8b}, [x9], #8
    sshl d0, d0, d29        // signed shift left;有符号左移
    sxtl v0.8h, v0.8b       // int8x8_t -> int16x8_t

    ld1 {v1.8b}, [x10], #8
    sshl d1, d1, d29
    sxtl v1.8h, v1.8b

    ld1 {v2.8b}, [x11], #8
    sshl d2, d2, d29
    sxtl v2.8h, v2.8b

    ld1 {v3.8b}, [x12], #8
    sshl d3, d3, d29
    sxtl v3.8h, v3.8b

    ld1 {v4.8b}, [x13], #8
    sshl d4, d4, d29
    sxtl v4.8h, v4.8b

    ld1 {v5.8b}, [x14], #8
    sshl d5, d5, d29
    sxtl v5.8h, v5.8b

    ld1 {v6.8b}, [x15], #8
    sshl d6, d6, d29
    sxtl v6.8h, v6.8b

    ld1 {v7.8b}, [x16], #8
    sshl d7, d7, d29
    sxtl v7.8h, v7.8b

    //COMPUTE_UNIT_LOW(0)
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b
    smlal  v8.4s, v27.4h, v0.h[0]
    smlal2 v9.4s, v27.8h, v0.h[0]
    smlal  v10.4s, v27.4h, v1.h[0]
    smlal2 v11.4s, v27.8h, v1.h[0]
    smlal  v12.4s, v27.4h, v2.h[0]
    smlal2 v13.4s, v27.8h, v2.h[0]
    smlal  v14.4s, v27.4h, v3.h[0]
    smlal2 v15.4s, v27.8h, v3.h[0]
    smlal  v16.4s, v27.4h, v4.h[0]
    smlal2 v17.4s, v27.8h, v4.h[0]
    smlal  v18.4s, v27.4h, v5.h[0]
    smlal2 v19.4s, v27.8h, v5.h[0]
    smlal  v20.4s, v27.4h, v6.h[0]
    smlal2 v21.4s, v27.8h, v6.h[0]
    smlal  v22.4s, v27.4h, v7.h[0]
    smlal2 v23.4s, v27.8h, v7.h[0]

    cmp x17, #-48
    blt 3f

    //COMPUTE_UNIT_LOW(1)
    ld1 {v28.8b}, [x5], #8
    sxtl v28.8h, v28.8b
    smlal  v8.4s, v28.4h, v0.h[1]
    smlal2 v9.4s, v28.8h, v0.h[1]
    smlal  v10.4s, v28.4h, v1.h[1]
    smlal2 v11.4s, v28.8h, v1.h[1]
    smlal  v12.4s, v28.4h, v2.h[1]
    smlal2 v13.4s, v28.8h, v2.h[1]
    smlal  v14.4s, v28.4h, v3.h[1]
    smlal2 v15.4s, v28.8h, v3.h[1]
    smlal  v16.4s, v28.4h, v4.h[1]
    smlal2 v17.4s, v28.8h, v4.h[1]
    smlal  v18.4s, v28.4h, v5.h[1]
    smlal2 v19.4s, v28.8h, v5.h[1]
    smlal  v20.4s, v28.4h, v6.h[1]
    smlal2 v21.4s, v28.8h, v6.h[1]
    smlal  v22.4s, v28.4h, v7.h[1]
    smlal2 v23.4s, v28.8h, v7.h[1]

    ble 3f

    //c2
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b
    smlal  v8.4s, v27.4h, v0.h[2]
    smlal2 v9.4s, v27.8h, v0.h[2]
    smlal  v10.4s, v27.4h, v1.h[2]
    smlal2 v11.4s, v27.8h, v1.h[2]
    smlal  v12.4s, v27.4h, v2.h[2]
    smlal2 v13.4s, v27.8h, v2.h[2]
    smlal  v14.4s, v27.4h, v3.h[2]
    smlal2 v15.4s, v27.8h, v3.h[2]
    smlal  v16.4s, v27.4h, v4.h[2]
    smlal2 v17.4s, v27.8h, v4.h[2]
    smlal  v18.4s, v27.4h, v5.h[2]
    smlal2 v19.4s, v27.8h, v5.h[2]
    smlal  v20.4s, v27.4h, v6.h[2]
    smlal2 v21.4s, v27.8h, v6.h[2]
    smlal  v22.4s, v27.4h, v7.h[2]
    smlal2 v23.4s, v27.8h, v7.h[2]

    cmp x17, #-32
    blo 3f

    //c3
    ld1 {v28.8b}, [x5], #8
    sxtl v28.8h, v28.8b
    smlal  v8.4s, v28.4h, v0.h[3]
    smlal2 v9.4s, v28.8h, v0.h[3]
    smlal  v10.4s, v28.4h, v1.h[3]
    smlal2 v11.4s, v28.8h, v1.h[3]
    smlal  v12.4s, v28.4h, v2.h[3]
    smlal2 v13.4s, v28.8h, v2.h[3]
    smlal  v14.4s, v28.4h, v3.h[3]
    smlal2 v15.4s, v28.8h, v3.h[3]
    smlal  v16.4s, v28.4h, v4.h[3]
    smlal2 v17.4s, v28.8h, v4.h[3]
    smlal  v18.4s, v28.4h, v5.h[3]
    smlal2 v19.4s, v28.8h, v5.h[3]
    smlal  v20.4s, v28.4h, v6.h[3]
    smlal2 v21.4s, v28.8h, v6.h[3]
    smlal  v22.4s, v28.4h, v7.h[3]
    smlal2 v23.4s, v28.8h, v7.h[3]

    bls 3f

    //c4
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b
    smlal  v8.4s, v27.4h, v0.h[4]
    smlal2 v9.4s, v27.8h, v0.h[4]
    smlal  v10.4s, v27.4h, v1.h[4]
    smlal2 v11.4s, v27.8h, v1.h[4]
    smlal  v12.4s, v27.4h, v2.h[4]
    smlal2 v13.4s, v27.8h, v2.h[4]
    smlal  v14.4s, v27.4h, v3.h[4]
    smlal2 v15.4s, v27.8h, v3.h[4]
    smlal  v16.4s, v27.4h, v4.h[4]
    smlal2 v17.4s, v27.8h, v4.h[4]
    smlal  v18.4s, v27.4h, v5.h[4]
    smlal2 v19.4s, v27.8h, v5.h[4]
    smlal  v20.4s, v27.4h, v6.h[4]
    smlal2 v21.4s, v27.8h, v6.h[4]
    smlal  v22.4s, v27.4h, v7.h[4]
    smlal2 v23.4s, v27.8h, v7.h[4]

    cmp x17, #-16
    blo 3f

    //c5
    ld1 {v28.8b}, [x5], #8
    sxtl v28.8h, v28.8b
    smlal  v8.4s, v28.4h, v0.h[5]
    smlal2 v9.4s, v28.8h, v0.h[5]
    smlal  v10.4s, v28.4h, v1.h[5]
    smlal2 v11.4s, v28.8h, v1.h[5]
    smlal  v12.4s, v28.4h, v2.h[5]
    smlal2 v13.4s, v28.8h, v2.h[5]
    smlal  v14.4s, v28.4h, v3.h[5]
    smlal2 v15.4s, v28.8h, v3.h[5]
    smlal  v16.4s, v28.4h, v4.h[5]
    smlal2 v17.4s, v28.8h, v4.h[5]
    smlal  v18.4s, v28.4h, v5.h[5]
    smlal2 v19.4s, v28.8h, v5.h[5]
    smlal  v20.4s, v28.4h, v6.h[5]
    smlal2 v21.4s, v28.8h, v6.h[5]
    smlal  v22.4s, v28.4h, v7.h[5]
    smlal2 v23.4s, v28.8h, v7.h[5]

    bls 3f

    //c6
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b
    smlal  v8.4s, v27.4h, v0.h[6]
    smlal2 v9.4s, v27.8h, v0.h[6]
    smlal  v10.4s, v27.4h, v1.h[6]
    smlal2 v11.4s, v27.8h, v1.h[6]
    smlal  v12.4s, v27.4h, v2.h[6]
    smlal2 v13.4s, v27.8h, v2.h[6]
    smlal  v14.4s, v27.4h, v3.h[6]
    smlal2 v15.4s, v27.8h, v3.h[6]
    smlal  v16.4s, v27.4h, v4.h[6]
    smlal2 v17.4s, v27.8h, v4.h[6]
    smlal  v18.4s, v27.4h, v5.h[6]
    smlal2 v19.4s, v27.8h, v5.h[6]
    smlal  v20.4s, v27.4h, v6.h[6]
    smlal2 v21.4s, v27.8h, v6.h[6]
    smlal  v22.4s, v27.4h, v7.h[6]
    smlal2 v23.4s, v27.8h, v7.h[6]

3:
    subs x3, x3, #1 // kernel size -1
    bne 0b

    ld1 {v24.4s}, [x8], #16  //v24: weight_scale
    movi v25.4s, #0
    cmp x1, #4
    ble 4f                  // nr <=4, jump 4f
    ld1 {v25.4s}, [x8]      // v25: weight_scale

4:
    ldr x8, [sp, #8]    //lonh relu
    cmp x8, #0
    bge 42f             // relu >=0 jump 42f
    movi v0.16b, #0
    smax v8.4s, v8.4s, v0.4s
    smax v9.4s, v9.4s, v0.4s
    smax v10.4s, v10.4s, v0.4s
    smax v11.4s, v11.4s, v0.4s
    smax v12.4s, v12.4s, v0.4s
    smax v13.4s, v13.4s, v0.4s
    smax v14.4s, v14.4s, v0.4s
    smax v15.4s, v15.4s, v0.4s
    smax v16.4s, v16.4s, v0.4s
    smax v17.4s, v17.4s, v0.4s
    smax v18.4s, v18.4s, v0.4s
    smax v19.4s, v19.4s, v0.4s
    smax v20.4s, v20.4s, v0.4s
    smax v21.4s, v21.4s, v0.4s
    smax v22.4s, v22.4s, v0.4s
    smax v23.4s, v23.4s, v0.4s


42: //  vacc0x0123 = vcvtaq_s32_f32(vfacc0x0123);
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    scvtf v10.4s, v10.4s
    scvtf v11.4s, v11.4s
    scvtf v12.4s, v12.4s
    scvtf v13.4s, v13.4s
    scvtf v14.4s, v14.4s
    scvtf v15.4s, v15.4s
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s

    ldr x8, [sp, #16]               // load add_input address
    fmul v8.4s, v8.4s, v24.4s       // val = val * scale
    fmul v10.4s, v10.4s, v24.4s
    fmul v12.4s, v12.4s, v24.4s
    fmul v14.4s, v14.4s, v24.4s
    fmul v16.4s, v16.4s, v24.4s
    fmul v18.4s, v18.4s, v24.4s
    fmul v20.4s, v20.4s, v24.4s
    fmul v22.4s, v22.4s, v24.4s
    fmul v9.4s, v9.4s, v25.4s
    fmul v11.4s, v11.4s, v25.4s
    fmul v13.4s, v13.4s, v25.4s
    fmul v15.4s, v15.4s, v25.4s
    fmul v17.4s, v17.4s, v25.4s
    fmul v19.4s, v19.4s, v25.4s
    fmul v21.4s, v21.4s, v25.4s
    fmul v23.4s, v23.4s, v25.4s

    cbz x8, 44f // if x8 == 0, jump 44f, no add fuse
    add x9, x8, x7
    cmp x0, #2
    csel x9, x8, x9, lo
    add x10, x9, x7
    csel x10, x9, x10, ls
    add x11, x10, x7
    cmp x0, #4
    csel x11, x10, x11, lo
    // load add input to v0 ~ v7
    ld1 {v28.8b}, [x8]
    sxtl v28.8h, v28.8b
    sxtl v0.4s, v28.4h
    sxtl2 v1.4s, v28.8h
    ld1 {v29.8b}, [x9]
    sxtl  v29.8h, v29.8b
    sxtl  v2.4s,  v29.4h
    sxtl2 v3.4s,  v29.8h
    ld1 {v28.8b}, [x10]
    sxtl  v28.8h, v28.8b
    sxtl  v4.4s,  v28.4h
    sxtl2 v5.4s,  v28.8h
    ld1 {v29.8b}, [x11]
    sxtl  v29.8h, v29.8b
    sxtl  v6.4s,  v29.4h
    sxtl2 v7.4s,  v29.8h

    ldr x8, [sp, #24]   //add_scale
    ld1 {v26.4s}, [x8], #16
    movi v27.4s, #0
    cmp x1, #4
    ble 43f
    ld1 {v27.4s}, [x8]
43:
    scvtf v0.4s, v0.4s
    scvtf v1.4s, v1.4s
    scvtf v2.4s, v2.4s
    scvtf v3.4s, v3.4s
    scvtf v4.4s, v4.4s
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s

    fmla v8.4s, v0.4s, v26.4s
    fmla v9.4s, v1.4s, v27.4s
    fmla v10.4s, v2.4s, v26.4s
    fmla v11.4s, v3.4s, v27.4s
    fmla v12.4s, v4.4s, v26.4s
    fmla v13.4s, v5.4s, v27.4s
    fmla v14.4s, v6.4s, v26.4s
    fmla v15.4s, v7.4s, v27.4s

44:
    cbz x8, 45f             // x8 == 0, jump 45f
    cmp x0, #4
    add  x12, x11, x7
    csel x12, x11, x12, ls
    add  x13, x12, x7
    cmp  x0,  #6
    csel x13, x12, x13, lo
    add  x14, x13, x7
    csel x14, x13, x14, ls
    add  x15, x14, x7
    cmp  x0,  #8
    csel x15, x14, x15, ne

    ld1 {v28.8b}, [x12]
    sxtl  v28.8h, v28.8b
    sxtl  v0.4s,  v28.4h
    sxtl2 v1.4s,  v28.8h
    ld1 {v29.8b}, [x13]
    sxtl  v29.8h, v29.8b
    sxtl  v2.4s,  v29.4h
    sxtl2 v3.4s,  v29.8h
    ld1 {v28.8b}, [x14]
    sxtl  v28.8h, v28.8b
    sxtl  v4.4s,  v28.4h
    sxtl2 v5.4s,  v28.8h
    ld1 {v29.8b}, [x15]
    sxtl  v29.8h, v29.8b
    sxtl  v6.4s,  v29.4h
    sxtl2 v7.4s,  v29.8h

    scvtf v0.4s, v0.4s
    scvtf v1.4s, v1.4s
    scvtf v2.4s, v2.4s
    scvtf v3.4s, v3.4s
    scvtf v4.4s, v4.4s
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s

    fmla v16.4s, v0.4s, v26.4s
    fmla v17.4s, v1.4s, v27.4s
    fmla v18.4s, v2.4s, v26.4s
    fmla v19.4s, v3.4s, v27.4s
    fmla v20.4s, v4.4s, v26.4s
    fmla v21.4s, v5.4s, v27.4s
    fmla v22.4s, v6.4s, v26.4s
    fmla v23.4s, v7.4s, v27.4s

45:
    fcvtas v8.4s, v8.4s         //    vacc0x0123 = vcvtaq_s32_f32(vfacc0x0123);
    fcvtas v9.4s, v9.4s
    fcvtas v10.4s, v10.4s
    fcvtas v11.4s, v11.4s
    fcvtas v12.4s, v12.4s
    fcvtas v13.4s, v13.4s
    fcvtas v14.4s, v14.4s
    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s

    sqxtn v8.4h, v8.4s
    sqxtn v10.4h, v10.4s
    sqxtn v12.4h, v12.4s
    sqxtn v14.4h, v14.4s
    sqxtn v16.4h, v16.4s
    sqxtn v18.4h, v18.4s
    sqxtn v20.4h, v20.4s
    sqxtn v22.4h, v22.4s
    sqxtn2 v8.8h, v9.4s
    sqxtn2 v10.8h, v11.4s
    sqxtn2 v12.8h, v13.4s
    sqxtn2 v14.8h, v15.4s
    sqxtn2 v16.8h, v17.4s
    sqxtn2 v18.8h, v19.4s
    sqxtn2 v20.8h, v21.4s
    sqxtn2 v22.8h, v23.4s

    sqxtn v8.8b, v8.8h
    sqxtn v10.8b, v10.8h
    sqxtn v12.8b, v12.8h
    sqxtn v14.8b, v14.8h
    sqxtn v16.8b, v16.8h
    sqxtn v18.8b, v18.8h
    sqxtn v20.8b, v20.8h
    sqxtn v22.8b, v22.8h

    ldr x8, [sp, #8]        // long relu
    cmp x8, #0              // x8 <= 0, jump 5
    ble 5f
    movi v0.16b, #0
    smax v8.8b, v8.8b, v0.8b
    smax v10.8b, v10.8b, v0.8b
    smax v12.8b, v12.8b, v0.8b
    smax v14.8b, v14.8b, v0.8b
    smax v16.8b, v16.8b, v0.8b
    smax v18.8b, v18.8b, v0.8b
    smax v20.8b, v20.8b, v0.8b
    smax v22.8b, v22.8b, v0.8b

5:
    add x9, x6, x7          // x6: output  x7: channel_stride
    cmp x0, #2              // x0 代表 mr: 8
    csel x9, x6, x9, lo     // if (mr < 2) ? x9 = x6 : x9 = x6+x7

    add x10, x9, x7
    csel x10, x9, x10, ls

    add x11, x10, x7
    cmp x0, #4
    csel x11, x10, x11, lo

    add x12, x11, x7
    csel x12, x11, x12, ls

    add x13, x12, x7
    cmp x0, #6
    csel x13, x12, x13, lo

    add x14, x13, x7
    csel x14, x13, x14, ls

    add x15, x14, x7
    cmp x0, #8
    csel x15, x14, x15, ne

    cmp x1, #8              // if (nr == 8) ?
    bne 6f                  // not equal then jump f6

    st1 {v8.d}[0], [x6]
    st1 {v10.d}[0], [x9]
    st1 {v12.d}[0], [x10]
    st1 {v14.d}[0], [x11]
    st1 {v16.d}[0], [x12]
    st1 {v18.d}[0], [x13]
    st1 {v20.d}[0], [x14]
    st1 {v22.d}[0], [x15]
    b 9f                    // jump return

6:
    cmp x1, #4              //
    blo 7f                  // if (x1 < 4) then jump to 7f

    st1 {v8.s}[0], [x6], #4
    st1 {v10.s}[0], [x9], #4
    st1 {v12.s}[0], [x10], #4
    st1 {v14.s}[0], [x11], #4
    st1 {v16.s}[0], [x12], #4
    st1 {v18.s}[0], [x13], #4
    st1 {v20.s}[0], [x14], #4
    st1 {v22.s}[0], [x15], #4

    sub x1, x1, #4
    ext v8.8b, v8.8b, v8.8b, #4
    ext v10.8b, v10.8b, v10.8b, #4
    ext v12.8b, v12.8b, v12.8b, #4
    ext v14.8b, v14.8b, v14.8b, #4
    ext v16.8b, v16.8b, v16.8b, #4
    ext v18.8b, v18.8b, v18.8b, #4
    ext v20.8b, v20.8b, v20.8b, #4
    ext v22.8b, v22.8b, v22.8b, #4

7:
    cmp x1, #2
    blo 8f                   // if (x1 < 2) then jump to 8f

    st1 {v8.h}[0], [x6], #2
    st1 {v10.h}[0], [x9], #2
    st1 {v12.h}[0], [x10], #2
    st1 {v14.h}[0], [x11], #2
    st1 {v16.h}[0], [x12], #2
    st1 {v18.h}[0], [x13], #2
    st1 {v20.h}[0], [x14], #2
    st1 {v22.h}[0], [x15], #2

    sub x1, x1, #2
    ext v8.8b, v8.8b, v8.8b, #2
    ext v10.8b, v10.8b, v10.8b, #2
    ext v12.8b, v12.8b, v12.8b, #2
    ext v14.8b, v14.8b, v14.8b, #2
    ext v16.8b, v16.8b, v16.8b, #2
    ext v18.8b, v18.8b, v18.8b, #2
    ext v20.8b, v20.8b, v20.8b, #2
    ext v22.8b, v22.8b, v22.8b, #2

8:
    cmp x1, #1
    blo 9f                      // if (x1 < 1) then jump to 9f

    st1 {v8.b}[0], [x6]
    st1 {v10.b}[0], [x9]
    st1 {v12.b}[0], [x10]
    st1 {v14.b}[0], [x11]
    st1 {v16.b}[0], [x12]
    st1 {v18.b}[0], [x13]
    st1 {v20.b}[0], [x14]
    st1 {v22.b}[0], [x15]

9:
    sub sp, sp, #176
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ldp x19, x20, [sp], #32
    ldr x21, [sp], #16
    ret

#endif
