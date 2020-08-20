TNN为使用者提供了内存读取的模型的转换工具。首先，在TNN编译时打开开关

```
mkdir build
cd build
cmake ..  -DTNN_TNN2MEM_ENABLE=ON 
```

然后就可以在tools/tnn2mem 目录下得到可执行工具tnn2mem,这里我们以常见的mobilenetv2为例

```
cd tools/tnn2mem
./tnn2mem mobilenetv2.tnnproto mobilenetv2.tnnmodel mobilenetv2.h
```

模型的参数会以非明文的形式保存在生成的mobilenetv2.h文件中

为了读取模型，我们需要头文件

```
#include "mobilenetv2.h"
#include "tnn/core/common.h"
#include "tnn/utils/string_utils.h"
#include <string>
```

在加载模型时，我们需要定义模型变量

```
ModelConfig model_config;
std::string mobilenetv2_tnnproto_string = UcharToString(mobilenetv2_tnnproto,mobilenetv2_tnnproto_length);
std::string mobilenetv2_tnnmodel_string = UcharToString(mobilenetv2_tnnmodel,mobilenetv2_model_length);
model_config.params.push_back(mobilenetv2_tnnproto_string);
model_config.params.push_back(mobilenetv2_tnnmodel_string);
```

在变量model_config中储存模型信息，之后便可按照所需要的补齐其他参数进行推理