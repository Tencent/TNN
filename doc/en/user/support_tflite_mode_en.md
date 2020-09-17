

# TFLite 算子的支持


| tflite operator         | tnn operatpr         | support |
|-------------------------+----------------------+---------|
| Add                     | Add                  | yes     |
| Average_Pool_2d         | Pooling              | yes     |
| Concatenation           | Concat               | yes     |
| Conv_2d                 | Convolution          | yes     |
| Cos                     | Cos                  | yes     |
| Depthwise_Conv_2d       | Convolution          | yes     |
| Detetion_Post_Process   | DetectionPostProcess | yes     |
| Div                     | Div                  | yes     |
| Exp                     | Exp                  | yes     |
| Full_Connected          | InnerProduct         | yes     |
| LeakyRelu               | Prelu                | yes     |
| Log                     | Logistic             | yes     |
| Logistic                | Sigmoid              | yes     |
| Max_Pool_2d             | Pooling              | yes     |
| Maximum                 | Maximum              | yes     |
| Mean                    | ReduceMean           | yes     |
| Minimum                 | Minimum              | yes     |
| Mul                     | Mul                  | yes     |
| Neg                     | Neg                  | yes     |
| Pad                     | Pad                  | yes     |
| Padv2                   | Pad                  | yes     |
| Prelu                   | Prelu                | yes     |
| Reshape                 | Reshape              | yes     |
| Resize_Biliner          | Upsample             | yes     |
| Resize_Nearest_Neighbor | Upsample             | yes     |
| Sin                     | Sin                  | yes     |
| Softmax                 | Softmax              | yes     |
| Split                   | SplitV               | yes     |
| SplitV                  | SplitV               | yes     |
| Squeeze                 | Squeeze              | yes     |
| StridedSlice            | StridedSlice         | yes     |
| Sub                     | Sub                  | yes     |
| Tanh                    | Tanh                 | yes     |
| Transpose_Conv          | Deconvolution        | yes     |


# TFLite 模型的支持


| tflite model                           | support align |
|----------------------------------------+---------------|
| alexnet                                | yes           |
| densenet_2018_04_27                    | yes           |
| face_landmark(media pipe)              | yes           |
| inception_v3_2018_04_27                | yes           |
| inception_v4_2018_04_27                | yes           |
| mobiletnet_v1_1.0_224                  | yes           |
| mobiletnet_v2_1.0_224                  | yes           |
| object_detection_3d(shoes, media pipe) | yes           |
| resnet_v2_101_229                      | yes           |
| squeezenet_2018_04_26                  | yes           |
| ssd                                    | yes           |
| vgg16                                  | yes           |
| yolo_tiny                              | yes           |
| yolov2_tiny                            | yes           |
