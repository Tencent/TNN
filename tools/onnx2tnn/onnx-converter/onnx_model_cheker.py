# -*- coding: utf-8 -*-
# @Author: Dandi Ding
# @Date:   2019-11-29 17:40:11
# @Last Modified by:   Dandiding
# @Last Modified time: 2019-11-29 17:43:41

import onnx
import onnxruntime as rt
import sys, traceback

if len(sys.argv) < 2:
    print("Usage {:s} onnx_model".format(sys.argv[0]))
    exit(0)

try:
    model = model = onnx.load(sys.argv[1])
    rt.InferenceSession(model.SerializeToString())
    print("Check passed.")
except RuntimeError:
    traceback.print_exc(file=sys.stdout)
    print("Check failed.")
