# v0.3 benchmark

* huawei P30 Pro(Kirin 980, Mali-G76 MP10)

| benchmark model | cpu time(thread 1,fp16, ms) | gpu time(ms) |
|-----------------|-----------------------------|--------------|
| DenseNet 121    | 65.70                       | 45.83        |
| Inception v3    | 130.98                      | 67.36        |
| Inception v4    | 310.67                      | 129.59       |
| MnasNet         | 11.74                       | 9.16         |
| MobileNet v1    | 16.39                       | 11.18        |
| MobileNet v2    | 14.81                       | 11.24        |
| ResNet50 v1     | 77.11                       | 44.29        |
| ResNet50 v2     | 90.53                       | 48.63        |
| ShuffleNet v2   | 7.66                        | 10.39        |
| SqueezeNet 1.0  | 8.38                        | 8.90         |
| SqueezeNet 1.1  | 8.37                        | 8.66         |

* xiaomi 6(Snapdragon 835, Adreno 540)

| benchmark model | cpu time(thread 1,fp16, ms) | gpu time(ms) |
|-----------------|-----------------------------|--------------|
| DenseNet 121    | 349.65                      | 86.81        |
| Inception v3    | 924.54                      | 77.01        |
| Inception v4    | 2286.02                     | 229.54       |
| MnasNet         | 61.80                       | 16.64        |
| MobileNet v1    | 95.46                       | 12.30        |
| MobileNet v2    | 82.85                       | 11.58        |
| ResNet50 v1     | 465.54                      | 65.77        |
| ResNet50 v2     | 575.29                      | 72.23        |
| ShuffleNet v2   | 36.93                       | 22.30        |
| SqueezeNet 1.0  | 53.37                       | 11.60        |
| SqueezeNet 1.1  | 53.47                       | 12.18        |

* samsung Galaxy S9+(Snapdragon 845, Adreno 630)

| benchmark model | cpu time(thread 1,fp16, ms) | gpu time(ms) |
|-----------------|-----------------------------|--------------|
| DenseNet 121    | 128.19                      | 63.65        |
| Inception v3    | 245.01                      | 71.00        |
| Inception v4    | 591.45                      | 145.76       |
| MnasNet         | 21.86                       | 9.35         |
| MobileNet v1    | 31.91                       | 10.15        |
| MobileNet v2    | 28.22                       | 9.89         |
| ResNet50 v1     | 152.59                      | 39.94        |
| ResNet50 v2     | 177.18                      | 45.34        |
| ShuffleNet v2   | 13.78                       | 9.41         |
| SqueezeNet 1.0  | 15.71                       | 6.58         |
| SqueezeNet 1.1  | 15.64                       | 7.00         |

* Oppo K3(Snapdragon 710, Adreno 616)

| benchmark model | cpu time(thread 1,fp16, ms) | gpu time(ms) |
|-----------------|-----------------------------|--------------|
| DenseNet 121    | 157.61                      | 114.56       |
| Inception v3    | 299.34                      | 163.22       |
| Inception v4    | 711.74                      | 345.85       |
| MnasNet         | 26.08                       | 18.69        |
| MobileNet v1    | 39.69                       | 23.10        |
| MobileNet v2    | 34.20                       | 22.21        |
| ResNet50 v1     | 184.75                      | 94.61        |
| ResNet50 v2     | 216.65                      | 107.23       |
| ShuffleNet v2   | 16.29                       | 12.90        |
| SqueezeNet 1.0  | 19.81                       | 15.70        |
| SqueezeNet 1.1  | 19.74                       | 15.74        |

* Intel(R) Xeon(R) Gold 6133 CPU

| benchmark model | cpu time(thread 1,fp32, ms) |
|-----------------|-----------------------------|
| Resnet50        | 151.00                      |
| YoloV5          | 2428.00                     |
| Bert-Based      | 832.00                      |
| Bert-Squad10    | 1093.00                     |

* TITAN Xp GPU

| benchmark model | gpu time(fp32, ms) |
|-----------------|--------------------|
| Resnet50        | 2.22               |
| YoloV5          | 17.47              |
| Bert-Based      | 8.16               |
| Bert-Squad10    | 9.60               |


# v0.1 benchmark

* Kirin970：

| model                     | cpu time(single thread, ms) | gpu time(ms) |
|---------------------------|--------------|--------------|
| Mobilenet_v1              | 88           |   12         |
| Mobilenet_v1_int8         | 55           |              |
| Mobilenet_v2              | 58           |   11         |
| Mobilenet_v2_int8         | 41           |              |
| squeezenet_v1.0           | 127          |   20         |
| squeezenet_v1.0_int8      | 82           |              |

* Snapdragon 835：

| model                     | cpu time(single thread, ms) | gpu time(ms) |
|---------------------------|--------------|--------------|
| Mobilenet_v1              | 94           |   16         |
| Mobilenet_v1_int8         | 62           |              |
| Mobilenet_v2              | 61           |   14         |
| Mobilenet_v2_int8         | 47           |              |
| squeezenet_v1.0           | 122          |   28         |
| squeezenet_v1.0_int8      | 93           |              |

* Snapdragon 845：

| model                     | cpu time(single thread, ms) | gpu time(ms) |
|---------------------------|--------------|--------------|
| Mobilenet_v1              | 60           |   10         |
| Mobilenet_v1_int8         | 37           |              |
| Mobilenet_v2              | 39           |   8          |
| Mobilenet_v2_int8         | 28           |              |
| squeezenet_v1.0           | 74           |   14         |
| squeezenet_v1.0_int8      | 56           |              |
