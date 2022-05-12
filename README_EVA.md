## 项目说明

本项目初始化自 [TNN 开源版本 2022_05_12](https://github.com/Tencent/TNN/tree/a69033975dc717c475f6e353974bcdb3fefd6c6d)

在使用过程中，TNN 开源版本存在一部分无法正常使用/对齐的问题，因此创建该项目，更新并记录对 TNN 库的响应修改，以支持 Numerous2TNN 工具的正常使用。

## 具体修改内容

### 1. GEMM 算子

- **算子逻辑** 

  该算子的计算逻辑为，输入 A，B，C，输出 A * B + C

- **存在问题**

   在 TNN 源码中，矩阵 B 和 C 的值仅可来源于 initializer 或 input，忽略了来源于中间结果的可能。

- **解决办法**

  修改 `/opt/TNN/tools/onnx2tnn/src/core/layer/onnx_converter_gemm.cc` 文件 153 行，bias 变量(即 C) 对应来源逻辑。

### 2. Selu 算子

- **算子逻辑**

  $$y = \left\{ \begin{array}{rcl} \gamma \cdot (\alpha \cdot e^x-\alpha)  & \mbox{for} & x \leq 0 \\ \gamma \cdot x & \mbox{for} & x \gt 0  \end{array}\right. $$

- **存在问题**

  在 TNN 源码中，$\gamma$ 与 $\alpha$ 初始化值均为 0，但在 ONNX 实现时，存在 default 值 
   
  - $\gamma = 1.0507009873554804934193349852946$
  - $\alpha = 1.6732632423543772848170429916717$

- **解决办法**

  修改 `/opt/TNN/source/tnn/device/cpu/acc/cpu_selu_layer_acc.cc` 文件 30 行，初始化 $\gamma$ 与 初始化 $\alpha$ 的逻辑。