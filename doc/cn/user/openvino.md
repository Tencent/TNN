# TNN 封装 Openvino X86 使用文档
1.  使用 scripts/build_openvino.sh 脚本编译 openvino 和 tnn
    ```
    $ cd scripts/
    $ sh build_openvino.sh
    ```
2.  进入 build_openvino/test/ 目录，使用 TNNTest 运行模型，并指定 device_type=NAIVE（必须） 和 network_type=OPENVINO（必须）。如果不用 ```-ip```（非必须） 指定输入文件，则 TNN 会随机生成输入信息，用 ```-op```（非必须） 指定输出文件
    ```
    $ cd build_openvino/test/
    $ ./TNNTest -mp PATH_TO_MODEL -dt NAIVE -nt OPENVINO -ip PATH_TO_INPUT -op PATH_TO_OUTPUT 
    ```
