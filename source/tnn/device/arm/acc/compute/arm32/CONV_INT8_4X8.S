#ifdef __arm__

#include "tnn/device/arm/acc/compute/asm_func_name.S"

.text

asm_function ASMConvInt8Unit4x8
//void ConvInt8Unit8x8(long mr, long nr, long kc, long ks, const int8_t** a,
//                    const void* w, int8_t* c, long c_stride, const float* scales,
//                    long relu, const int8_t* add_input, const float* add_scale)
//x0(mr),
//x1(nr),
//x2(kc),
//x3(ks),
//x4(a),
//x5(w),
//x6(c),
//x7(c_stride),
//from stack(scales),
//from stack(relu)

# Load w
# - ip = w
LDR ip, [sp, #4]
PUSH {r4, r5, r6, r7, r8, r9, r10, r11}

VPUSH {d8-d15}

# Load bias0123, bias4567
VLDM ip!, {d16-d19}

# q10 := vacc1x0123
VMOV.I32 q10, q8
MOV r4, #4
# q11 := vacc1x4567
VMOV.I32 q11, q9
# Load a
# - r8 = a
LDR r8, [sp, #96]
# q12 := vacc2x0123
VMOV.I32 q12, q8
# q13 := vacc2x4567
VMOV.I32 q13, q9
# q14 := vacc3x0123
VMOV.I32 q14, q8
# q15 := vacc3x4567
VMOV.I32 q15, q9

VMOV.I8 d15, #0

.p2align 5

0:
	LDM r8!, {r4-r7}
	LDR r9, [sp, #128]
	LDR r11, [sp, #132]

	CMP r4, #-1
	ADD r4, r11, r4
	MOVEQ r4, r9

	CMP r5, #-1
	ADD r5, r11, r5
	MOVEQ r5, r9

	CMP r6, #-1
	ADD r6, r11, r6
	MOVEQ r6, r9

	CMP r7, #-1
	ADD r7, r11, r7
	MOVEQ r7, r9

	SUBS r10, r2, #8

	# Load a0, a1, a2, a3
	# - r4 = a0
	# - r5 = a1
	# - r6 = a2
	# - r7 = a3
	#LDM r8!, {r4-r7}


	BLO 2f

1:
	# Load va0
	# - d1 = va0
	VLD1.8 {d1}, [r4]!

	# Load va1
	# - d3 = va1
	VLD1.8 {d3}, [r5]!

	# Load vb0-vb7 (channel 0)
	# - d9 = vb0-vb7
	VLD1.8 {d9}, [ip:64]!

	# Load va2
	# - d5 = va2
	VLD1.8 {d5}, [r6]!

	# q0 = va0 = a0
	VMOVL.S8 q0, d1

	# Load va3
	# - d7 = va3
	VLD1.8 {d7}, [r7]!

	# q1 = va1 = a1
	VMOVL.S8 q1, d3

	# q4 = b0:7 - vb_zero_point
	# - d8 = vb0123 (channel 0)
	# - d9 = vb4567 (channel 0)
	VSUBL.S8 q4, d9, d15

	# q2 = va2 = a2
	VMOVL.S8 q2, d5
	# q3 = va3 = a3
	VMOVL.S8 q3, d7

	### Channel 0 ###

	# Load b0-b7 (channel 1)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# vacc0x0123 += vb0123 * va0[0]
	VMLAL.S16 q8, d8, d0[0]
	# vacc0x4567 += vb4567 * va0[0]
	VMLAL.S16 q9, d9, d0[0]

	# vacc1x0123 += vb0123 * va1[0]
	VMLAL.S16 q10, d8, d2[0]
	# vacc1x4567 += vb4567 * va1[0]
	VMLAL.S16 q11, d9, d2[0]

	# vacc2x0123 += vb0123 * va2[0]
	VMLAL.S16 q12, d8, d4[0]
	# vacc2x4567 += vb4567 * va2[0]
	VMLAL.S16 q13, d9, d4[0]

	# q5 = b0:7 - vb_zero_point
	# - d10 = vb0123 (channel 1)
	# - d11 = vb4567 (channel 1)
	VSUBL.S8 q5, d11, d15

	# vacc3x0123 += vb0123 * va3[0]
	VMLAL.S16 q14, d8, d6[0]
	# vacc3x4567 += vb4567 * va3[0]
	VMLAL.S16 q15, d9, d6[0]

	### Channel 1 ###

	# Load b0-b7 (channel 2)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# vacc0x0123 += vb0123 * va0[1]
	VMLAL.S16 q8, d10, d0[1]
	# vacc0x4567 += vb4567 * va0[1]
	VMLAL.S16 q9, d11, d0[1]

	# vacc1x0123 += vb0123 * va1[1]
	VMLAL.S16 q10, d10, d2[1]
	# vacc1x4567 += vb4567 * va1[1]
	VMLAL.S16 q11, d11, d2[1]

	# vacc2x0123 += vb0123 * va2[1]
	VMLAL.S16 q12, d10, d4[1]
	# vacc2x4567 += vb4567 * va2[1]
	VMLAL.S16 q13, d11, d4[1]

	# q4 = b0:7 - vb_zero_point
	# - d8 = vb0123 (channel 2)
	# - d9 = vb4567 (channel 2)
	VSUBL.S8 q4, d9, d15

	# vacc3x0123 += vb0123 * va3[1]
	VMLAL.S16 q14, d10, d6[1]
	# vacc3x4567 += vb4567 * va3[1]
	VMLAL.S16 q15, d11, d6[1]

	### Channel 2 ###

	# Load b0-b7 (channel 3)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# vacc0x0123 += vb0123 * va0[2]
	VMLAL.S16 q8, d8, d0[2]
	# vacc0x4567 += vb4567 * va0[2]
	VMLAL.S16 q9, d9, d0[2]

	# vacc1x0123 += vb0123 * va1[2]
	VMLAL.S16 q10, d8, d2[2]
	# vacc1x4567 += vb4567 * va1[2]
	VMLAL.S16 q11, d9, d2[2]

	# vacc2x0123 += vb0123 * va2[2]
	VMLAL.S16 q12, d8, d4[2]
	# vacc2x4567 += vb4567 * va2[2]
	VMLAL.S16 q13, d9, d4[2]

	# q5 = b0:7 - vb_zero_point
	# - d10 = vb0123 (channel 3)
	# - d11 = vb4567 (channel 3)
	VSUBL.S8 q5, d11, d15

	# vacc3x0123 += vb0123 * va3[2]
	VMLAL.S16 q14, d8, d6[2]
	# vacc3x4567 += vb4567 * va3[2]
	VMLAL.S16 q15, d9, d6[2]

	### Channel 3 ###

	# Load b0-b7 (channel 4)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# vacc0x0123 += vb0123 * va0[3]
	VMLAL.S16 q8, d10, d0[3]
	# vacc0x4567 += vb4567 * va0[3]
	VMLAL.S16 q9, d11, d0[3]

	# vacc1x0123 += vb0123 * va1[3]
	VMLAL.S16 q10, d10, d2[3]
	# vacc1x4567 += vb4567 * va1[3]
	VMLAL.S16 q11, d11, d2[3]

	# vacc2x0123 += vb0123 * va2[3]
	VMLAL.S16 q12, d10, d4[3]
	# vacc2x4567 += vb4567 * va2[3]
	VMLAL.S16 q13, d11, d4[3]

	# q5 = b0:7 - vb_zero_point
	# - d10 = vb0123 (channel 4)
	# - d11 = vb4567 (channel 4)
	VSUBL.S8 q4, d9, d15

	# vacc3x0123 += vb0123 * va3[3]
	VMLAL.S16 q14, d10, d6[3]
	# vacc3x4567 += vb4567 * va3[3]
	VMLAL.S16 q15, d11, d6[3]

	### Channel 4 ###

	# Load b0-b7 (channel 5)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# vacc0x0123 += vb0123 * va0[4]
	VMLAL.S16 q8, d8, d1[0]
	# vacc0x4567 += vb4567 * va0[4]
	VMLAL.S16 q9, d9, d1[0]

	# vacc1x0123 += vb0123 * va1[4]
	VMLAL.S16 q10, d8, d3[0]
	# vacc1x4567 += vb4567 * va1[4]
	VMLAL.S16 q11, d9, d3[0]

	# vacc2x0123 += vb0123 * va2[4]
	VMLAL.S16 q12, d8, d5[0]
	# vacc2x4567 += vb4567 * va2[4]
	VMLAL.S16 q13, d9, d5[0]

	# q4 = b0:7 - vb_zero_point
	# - d8 = vb0123 (channel 5)
	# - d9 = vb4567 (channel 5)
	VSUBL.S8 q5, d11, d15

	# vacc3x0123 += vb0123 * va3[4]
	VMLAL.S16 q14, d8, d7[0]
	# vacc3x4567 += vb4567 * va3[4]
	VMLAL.S16 q15, d9, d7[0]

	### Channel 5 ###

	# Load b0-b7 (channel 6)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# vacc0x0123 += vb0123 * va0[5]
	VMLAL.S16 q8, d10, d1[1]
	# vacc0x4567 += vb4567 * va0[5]
	VMLAL.S16 q9, d11, d1[1]

	# vacc1x0123 += vb0123 * va1[5]
	VMLAL.S16 q10, d10, d3[1]
	# vacc1x4567 += vb4567 * va1[5]
	VMLAL.S16 q11, d11, d3[1]

	# vacc2x0123 += vb0123 * va2[5]
	VMLAL.S16 q12, d10, d5[1]
	# vacc2x4567 += vb4567 * va2[5]
	VMLAL.S16 q13, d11, d5[1]

	# q4 = b0:7 - vb_zero_point
	# - d8 = vb0123 (channel 6)
	# - d9 = vb4567 (channel 6)
	VSUBL.S8 q4, d9, d15

	# vacc3x0123 += vb0123 * va3[5]
	VMLAL.S16 q14, d10, d7[1]
	# vacc3x4567 += vb4567 * va3[5]
	VMLAL.S16 q15, d11, d7[1]

	### Channel 6 ###

	# Load b0-b7 (channel 7)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# vacc0x0123 += vb0123 * va0[6]
	VMLAL.S16 q8, d8, d1[2]
	# vacc0x4567 += vb4567 * va0[6]
	VMLAL.S16 q9, d9, d1[2]

	# vacc1x0123 += vb0123 * va1[6]
	VMLAL.S16 q10, d8, d3[2]
	# vacc1x4567 += vb4567 * va1[6]
	VMLAL.S16 q11, d9, d3[2]

	# vacc2x0123 += vb0123 * va2[6]
	VMLAL.S16 q12, d8, d5[2]

	# q5 = b0:7 - vb_zero_point
	# - d10 = vb0123 (channel 7)
	# - d11 = vb4567 (channel 7)
	VSUBL.S8 q5, d11, d15

	# vacc2x4567 += vb4567 * va2[6]
	VMLAL.S16 q13, d9, d5[2]

	# vacc3x0123 += vb0123 * va3[6]
	VMLAL.S16 q14, d8, d7[2]
	# vacc3x4567 += vb4567 * va3[6]
	VMLAL.S16 q15, d9, d7[2]

	### Channel 8 ###
	SUBS r10, r10, #8

	# vacc0x0123 += vb0123 * va0[7]
	VMLAL.S16 q8, d10, d1[3]
	# vacc0x4567 += vb4567 * va0[7]
	VMLAL.S16 q9, d11, d1[3]

	# vacc1x0123 += vb0123 * va1[7]
	VMLAL.S16 q10, d10, d3[3]
	# vacc1x4567 += vb4567 * va1[7]
	VMLAL.S16 q11, d11, d3[3]

	# vacc2x0123 += vb0123 * va2[7]
	VMLAL.S16 q12, d10, d5[3]
	# vacc2x4567 += vb4567 * va2[7]
	VMLAL.S16 q13, d11, d5[3]

	# vacc3x0123 += vb0123 * va3[7]
	VMLAL.S16 q14, d10, d7[3]
	# vacc3x4567 += vb4567 * va3[7]
	VMLAL.S16 q15, d11, d7[3]

	BHS 1b

2:
	CMP r10, #-8
	BEQ 3f
	
	# Adjust a0, a1, a2, a3
	ADD r4, r10
	ADD r5, r10
	ADD r6, r10
	ADD r7, r10

	# a_shift = 8 * k - 64
	LSL r10, r10, #3
	VDUP.32 d13, r10

	# Load va0
	# - d1 = va0
	VLD1.8 {d1}, [r4]

	# Load va1
	# - d3 = va1
	VLD1.8 {d3}, [r5]

	# Load b0-b7 (channel 0)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# Load a2
	# - d5 = a2
	VLD1.8 {d5}, [r6]

	# q0 = va0 = a0
	VSHL.S64 d1, d1, d13
	VMOVL.S8 q0, d1

	# Load a3
	# - d7 = a3
	VLD1.8 {d7}, [r7]

	# q1 = va1 = a1
	VSHL.S64 d3, d3, d13
	VMOVL.S8 q1, d3

	# q4 = b0:7 - vb_zero_point
	# - d8 = vb0123 (channel 0)
	# - d9 = vb4567 (channel 0)
	VSUBL.S8 q4, d9, d15

	# q2 = va2 = a2
	VSHL.S64 d5, d5, d13
	VMOVL.S8 q2, d5
	# q3 = va3 = a3
	VSHL.S64 d7, d7, d13
	VMOVL.S8 q3, d7

	### Channel 0 ###

	# vacc0x0123 += vb0123 * va0[0]
	VMLAL.S16 q8, d8, d0[0]
	# vacc0x4567 += vb4567 * va0[0]
	VMLAL.S16 q9, d9, d0[0]

	# vacc1x0123 += vb0123 * va1[0]
	VMLAL.S16 q10, d8, d2[0]
	# vacc1x4567 += vb4567 * va1[0]
	VMLAL.S16 q11, d9, d2[0]

	# vacc2x0123 += vb0123 * va2[0]
	VMLAL.S16 q12, d8, d4[0]
	# vacc2x4567 += vb4567 * va2[0]
	VMLAL.S16 q13, d9, d4[0]

	# vacc3x0123 += vb0123 * va3[0]
	VMLAL.S16 q14, d8, d6[0]
	# vacc3x4567 += vb4567 * va3[0]
	VMLAL.S16 q15, d9, d6[0]

	CMP r10, #-48
	BLO 3f

	### Channel 1 ###

	# Load b0-b7 (channel 1)
	# - d11 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# q5 = b0:7 - vb_zero_point
	# - d10 = vb0123 (channel 1)
	# - d11 = vb4567 (channel 1)
	VSUBL.S8 q5, d11, d15

	# vacc0x0123 += vb0123 * va0[1]
	VMLAL.S16 q8, d10, d0[1]
	# vacc0x4567 += vb4567 * va0[1]
	VMLAL.S16 q9, d11, d0[1]

	# vacc1x0123 += vb0123 * va1[1]
	VMLAL.S16 q10, d10, d2[1]
	# vacc1x4567 += vb4567 * va1[1]
	VMLAL.S16 q11, d11, d2[1]

	# vacc2x0123 += vb0123 * va2[1]
	VMLAL.S16 q12, d10, d4[1]
	# vacc2x4567 += vb4567 * va2[1]
	VMLAL.S16 q13, d11, d4[1]

	# vacc3x0123 += vb0123 * va3[1]
	VMLAL.S16 q14, d10, d6[1]
	# vacc3x4567 += vb4567 * va3[1]
	VMLAL.S16 q15, d11, d6[1]

	### Channel 2 ###
	BLS 3f

	# Load b0-b7 (channel 2)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# q4 = b0:7 - vb_zero_point
	# - d8 = vb0123 (channel 2)
	# - d9 = vb4567 (channel 2)
	VSUBL.S8 q4, d9, d15

	# vacc0x0123 += vb0123 * va0[2]
	VMLAL.S16 q8, d8, d0[2]
	# vacc0x4567 += vb4567 * va0[2]
	VMLAL.S16 q9, d9, d0[2]

	# vacc1x0123 += vb0123 * va1[2]
	VMLAL.S16 q10, d8, d2[2]
	# vacc1x4567 += vb4567 * va1[2]
	VMLAL.S16 q11, d9, d2[2]

	# vacc2x0123 += vb0123 * va2[2]
	VMLAL.S16 q12, d8, d4[2]
	# vacc2x4567 += vb4567 * va2[2]
	VMLAL.S16 q13, d9, d4[2]

	# vacc3x0123 += vb0123 * va3[2]
	VMLAL.S16 q14, d8, d6[2]
	# vacc3x4567 += vb4567 * va3[2]
	VMLAL.S16 q15, d9, d6[2]

	### Channel 3 ###
	CMP r10, #-32
	BLO 3f

	# Load b0-b7 (channel 3)
	# - d9 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# q4 = b0:7 - vb_zero_point
	# - d8 = vb0123 (channel 3)
	# - d9 = vb4567 (channel 3)
	VSUBL.S8 q5, d11, d15

	# vacc0x0123 += vb0123 * va0[3]
	VMLAL.S16 q8, d10, d0[3]
	# vacc0x4567 += vb4567 * va0[3]
	VMLAL.S16 q9, d11, d0[3]

	# vacc1x0123 += vb0123 * va1[3]
	VMLAL.S16 q10, d10, d2[3]
	# vacc1x4567 += vb4567 * va1[3]
	VMLAL.S16 q11, d11, d2[3]

	# vacc2x0123 += vb0123 * va2[3]
	VMLAL.S16 q12, d10, d4[3]
	# vacc2x4567 += vb4567 * va2[3]
	VMLAL.S16 q13, d11, d4[3]

	# vacc3x0123 += vb0123 * va3[3]
	VMLAL.S16 q14, d10, d6[3]
	# vacc3x4567 += vb4567 * va3[3]
	VMLAL.S16 q15, d11, d6[3]

	### Channel 4 ###
	BLS 3f

	# Load b0-b7 (channel 4)
	# - d11 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# q5 = b0:7 - vb_zero_point
	# - d10 = vb0123 (channel 4)
	# - d11 = vb4567 (channel 4)
	VSUBL.S8 q4, d9, d15

	# vacc0x0123 += vb0123 * va0[4]
	VMLAL.S16 q8, d8, d1[0]
	# vacc0x4567 += vb4567 * va0[4]
	VMLAL.S16 q9, d9, d1[0]

	# vacc1x0123 += vb0123 * va1[4]
	VMLAL.S16 q10, d8, d3[0]
	# vacc1x4567 += vb4567 * va1[4]
	VMLAL.S16 q11, d9, d3[0]

	# vacc2x0123 += vb0123 * va2[4]
	VMLAL.S16 q12, d8, d5[0]
	# vacc2x4567 += vb4567 * va2[4]
	VMLAL.S16 q13, d9, d5[0]

	# vacc3x0123 += vb0123 * va3[4]
	VMLAL.S16 q14, d8, d7[0]
	# vacc3x4567 += vb4567 * va3[4]
	VMLAL.S16 q15, d9, d7[0]

	### Channel 5 ###
	CMP r10, #-16
	BLO 3f

	# Load b0-b7 (channel 5)
	# - d13 = b0-b7
	VLD1.8 {d11}, [ip:64]!

	# q5 = b0:7 - vb_zero_point
	# - d10 = vb0123 (channel 5)
	# - d11 = vb4567 (channel 5)
	VSUBL.S8 q5, d11, d15

	# vacc0x0123 += vb0123 * va0[5]
	VMLAL.S16 q8, d10, d1[1]
	# vacc0x4567 += vb4567 * va0[5]
	VMLAL.S16 q9, d11, d1[1]

	# vacc1x0123 += vb0123 * va1[5]
	VMLAL.S16 q10, d10, d3[1]
	# vacc1x4567 += vb4567 * va1[5]
	VMLAL.S16 q11, d11, d3[1]

	# vacc2x0123 += vb0123 * va2[5]
	VMLAL.S16 q12, d10, d5[1]
	# vacc2x4567 += vb4567 * va2[5]
	VMLAL.S16 q13, d11, d5[1]

	# vacc3x0123 += vb0123 * va3[5]
	VMLAL.S16 q14, d10, d7[1]
	# vacc3x4567 += vb4567 * va3[5]
	VMLAL.S16 q15, d11, d7[1]

	### Channel 6 ###
	BLS 3f

	# Load b0-b7 (channel 6)
	# - d9 = b0-b7
	VLD1.8 {d9}, [ip:64]!

	# q4 = b0:7 - vb_zero_point
	# - d8 = vb0123 (channel 6)
	# - d9 = vb4567 (channel 6)
	VSUBL.S8 q4, d9, d15

	# vacc0x0123 += vb0123 * va0[6]
	VMLAL.S16 q8, d8, d1[2]
	# vacc0x4567 += vb4567 * va0[6]
	VMLAL.S16 q9, d9, d1[2]

	# vacc1x0123 += vb0123 * va1[6]
	VMLAL.S16 q10, d8, d3[2]
	# vacc1x4567 += vb4567 * va1[6]
	VMLAL.S16 q11, d9, d3[2]

	# vacc2x0123 += vb0123 * va2[6]
	VMLAL.S16 q12, d8, d5[2]

	# vacc2x4567 += vb4567 * va2[6]
	VMLAL.S16 q13, d9, d5[2]

	# vacc3x0123 += vb0123 * va3[6]
	VMLAL.S16 q14, d8, d7[2]
	# vacc3x4567 += vb4567 * va3[6]
	VMLAL.S16 q15, d9, d7[2]

	.p2align 4

3:  
    SUBS r3, r3, #1
	BNE 0b

    LDR r9, [sp, #112]
    VLD1.32 {d12, d13}, [r9]!
	cmp r1, #4
	ble 4f
	VLD1.32 {d14, d15}, [r9]!

4:
    LDR r4, [sp, #116]    //relu
    CMP r4, #0
    BGE 42f
    VMOV.I16 q0, #0
    VMAX.S16 q8, q8, q0
    VMAX.S16 q9, q9, q0
    VMAX.S16 q10, q10, q0
    VMAX.S16 q11, q11, q0
    VMAX.S16 q12, q12, q0
    VMAX.S16 q13, q13, q0
    VMAX.S16 q14, q14, q0
    VMAX.S16 q15, q15, q0

42:
    LDR r4, [sp, #120]
    VCVT.F32.S32 q8, q8
    VCVT.F32.S32 q9, q9
    VCVT.F32.S32 q10, q10
    VCVT.F32.S32 q11, q11
    VCVT.F32.S32 q12, q12
    VCVT.F32.S32 q13, q13
    VCVT.F32.S32 q14, q14
    VCVT.F32.S32 q15, q15

    VMUL.F32  q8, q8, q6
	VMUL.F32  q10, q10, q6
	VMUL.F32 q12, q12, q6
	VMUL.F32 q14, q14, q6

    VMUL.F32 q9, q9, q7
	VMUL.F32 q11, q11, q7
	VMUL.F32 q13, q13, q7
	VMUL.F32 q15, q15, q7

	LDR r3, [sp, #108]

	CMP r4, #0
    BEQ 44f
    ADD r5, r4, r3
    CMP r0, #2
	MOVLO r5, r4
    ADD r6, r5, r3
	MOVLS r6, r5
    ADD r7, r6, r3
    CMP r0, #4
	MOVLO r7, r6

    VLD1.8 {d0}, [r4]
    VMOVL.S8 q0, d0
    VMOVL.S16 q4, d0
	VMOVL.S16 q5, d1

    VLD1.8 {d0}, [r5]
    VMOVL.S8 q0, d0
    VMOVL.S16 q6, d0
	VMOVL.S16 q7, d1
    
    LDR r8, [sp, #124]
    VLD1.32 {d2, d3}, [r8]!
	VMOV.F32 q2, #0.0
	cmp r1, #4
	ble 43f
	VLD1.32 {d4, d5}, [r8]!

43:
    VCVT.F32.S32 q4, q4
    VCVT.F32.S32 q5, q5
	VCVT.F32.S32 q6, q6
    VCVT.F32.S32 q7, q7

    VMLA.F32 q8, q4, q1
	VMLA.F32 q9, q5, q2
	VMLA.F32 q10, q6, q1
	VMLA.F32 q11, q7, q2

    VLD1.8 {d0}, [r6]
    VMOVL.S8 q0, d0
    VMOVL.S16 q4, d0
	VMOVL.S16 q5, d1

    VLD1.8 {d0}, [r7]
    VMOVL.S8 q0, d0
    VMOVL.S16 q6, d0
	VMOVL.S16 q7, d1

	VCVT.F32.S32 q4, q4
    VCVT.F32.S32 q5, q5
	VCVT.F32.S32 q6, q6
    VCVT.F32.S32 q7, q7

    VMLA.F32 q12, q4, q1
	VMLA.F32 q13, q5, q2
	VMLA.F32 q14, q6, q1
	VMLA.F32 q15, q7, q2

44:
	ldr r9, .L4
    vdup.32 q0, r9
    vadd.f32 q8, q8, q0
    vadd.f32 q9, q9, q0
    vadd.f32 q10, q10, q0
    vadd.f32 q11, q11, q0
    vadd.f32 q12, q12, q0
    vadd.f32 q13, q13, q0
    vadd.f32 q14, q14, q0
    vadd.f32 q15, q15, q0

    vsub.s32 q8, q8, q0
    vsub.s32 q9, q9, q0
    vsub.s32 q10, q10, q0
    vsub.s32 q11, q11, q0
    vsub.s32 q12, q12, q0
    vsub.s32 q13, q13, q0
    vsub.s32 q14, q14, q0
    vsub.s32 q15, q15, q0

	LDRD r2, r3, [sp, #104]

	VQMOVN.S32 d16, q8
	VQMOVN.S32 d17, q9
	VQMOVN.S32 d18, q10
	VQMOVN.S32 d19, q11
	VQMOVN.S32 d20, q12
	VQMOVN.S32 d21, q13
	VQMOVN.S32 d22, q14
	VQMOVN.S32 d23, q15

	VQMOVN.S16 d16, q8
	VQMOVN.S16 d17, q9
	VQMOVN.S16 d18, q10
	VQMOVN.S16 d19, q11

	LDR r9, [sp, #116]
	ADD r4, r2, r3
	CMP r9, #0
	BLE 5f
    VMOV.I8 d0, #0
	VMAX.S8 d16, d16, d0
	VMAX.S8 d17, d17, d0
	VMAX.S8 d18, d18, d0
	VMAX.S8 d19, d19, d0

5:
	CMP r0, #2
	MOVLO r4, r2
	ADD r5, r4, r3
	MOVLS r5, r4
	CMP r0, #4
	ADD r3, r5, r3
	MOVNE r3, r5
	CMP r1, #8
	BNE 6f

	VST1.8 {d16}, [r2]
	VST1.8 {d17}, [r4]
	VST1.8 {d18}, [r5]
	VST1.8 {d19}, [r3]

	VPOP {d8-d15}
	POP {r4, r5, r6, r7, r8, r9, r10, r11}
	BX lr

	.p2align 3

6:
	CMP r1, #4
	BLO 7f

	VST1.32 {d16[0]}, [r2]!
	VST1.32 {d17[0]}, [r4]!
	VST1.32 {d18[0]}, [r5]!
	VST1.32 {d19[0]}, [r3]!

	SUB r1, #4
	VEXT.8 q8, q8, q8, #4
	VEXT.8 q9, q9, q9, #4

7:
	CMP r1, #2
	BLO 8f

	VST1.16 {d16[0]}, [r2]!
	VST1.16 {d17[0]}, [r4]!
	VST1.16 {d18[0]}, [r5]!
	VST1.16 {d19[0]}, [r3]!

	SUB r1, #2
	VEXT.8 q8, q8, q8, #2
	VEXT.8 q9, q9, q9, #2

8:
	TEQ r1, #0
	BEQ 9f

	VST1.8 {d16[0]}, [r2]
	VST1.8 {d17[0]}, [r4]
	VST1.8 {d18[0]}, [r5]
	VST1.8 {d19[0]}, [r3]

9:
	VPOP {d8-d15}
	POP {r4, r5, r6, r7, r8, r9, r10, r11}
	BX lr

.L4:
    .word 0x4B400000
	
#endif








