# 如何正确放置HiAI ddk

1. 到华为开发者联盟下载DDK[https://developer.huawei.com/consumer/cn/doc/overview/HUAWEI_HiAI]
2. 解压缩
3. 进入到下载文件夹下的`ddk/ai_ddk_lib`目录
4. 在`<path_to_tnn>/third_party/npu/hiai_ddk_latest/`下创建`armeabi-v7a`文件夹， 并将ai_ddk_lib目录下的lib文件夹中所有文件复制到 `<path_to_tnn>/third_party/npu/hiai_ddk_latest/armeabi-v7a`
5. 在`<path_to_tnn>/third_party/npu/hiai_ddk_latest/`下创建`arm64-v8a`文件夹，并将ai_ddk_lib目录下的lib64文件夹中所有文件复制到 `<path_to_tnn>/third_party/npu/hiai_ddk_latest/arm64-v8a`
6. 将ai_ddk_lib目录下include`文件夹`复制到 `<path_to_tnn>/third_party/npu/hiai_ddk_latest/`目录下
7. 最终 `<path_to_tnn>/third_party/npu/hiai_ddk_latest/`文件结构应该如下：

```
hiai_ddk_latest
├── arm64-v8a 
│   ├── libcpucl.so 
│   ├── libhcl.so
│   ├── libhiai.so
│   ├── libhiai_ir.so
│   └── libhiai_ir_build.so
├── armeabi-v7a
│   ├── libcpucl.so
│   ├── libhcl.so
│   ├── libhiai.so
│   ├── libhiai_ir.so
│   └── libhiai_ir_build.so
└── include
    ├── HiAiAippPara.h
    ├── HiAiModelManagerService.h
    ├── HiAiModelManagerType.h
    ├── graph
    │   ├── attr_value.h
    │   ├── buffer.h
    │   ├── common
    │   │   └── secures\tl.h
    │   ├── debug
    │   │   └── ge_error_codes.h
    │   ├── detail
    │   │   └── attributes_holder.h
    │   ├── graph.h
    │   ├── model.h
    │   ├── op
    │   │   ├── all_ops.h
    │   │   ├── array_defs.h
    │   │   ├── const_defs.h
    │   │   ├── detection_defs.h
    │   │   ├── image_defs.h
    │   │   ├── math_defs.h
    │   │   ├── nn_defs.h
    │   │   └── random_defs.h
    │   ├── operator.h
    │   ├── operator_reg.h
    │   ├── tensor.h 
    │   └── types.h
    └── hiai_ir_build.h
```

