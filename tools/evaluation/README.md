# Evaluation
## I. Image Classify(ILSVRC2012)
### 1. Download ILSVRC
In order to use this tool to run evaluation on the 50K ImageNet dataset, download the data set from http://image-net.org/request.

### 2. Ground truth label id generation
Use the python script to generate validation id labels.
```
python generate_validation_labels.py \
    --ilsvrc_devkit_dir=${ILSVRC_2012_DEVKIT_DIR} \
    --synset_words_file=${TNN_ROOT_PATH}/examples/assets/synset.txt \
    --validation_labels_output=val_id.txt
```

### 3.Image Classification evaluation
- Turn on `TNN_EVALUATION_ENABLE` and build `eval_cmd`
```
-DTNN_EVALUATION_ENABLE=ON
```
- Usage 
```
./eval_cmd  [-h] [-p] [-m] [-i] [-g] <param>
```
- Parameter Description  

| option             | mandatory | with value | description                                  |
| :----------------- | :-------: | :--------: | :------------------------------------------- |
| -h, --help         |           |            | Output command prompt.                       |
| -p, --proto        |  &radic;  |  &radic;   | Specify tnnproto model description file.     |
| -m, --model        |  &radic;  |  &radic;   | Specify the tnnmodel model parameter file.   |
| -i, --input_path   |  &radic;  |  &radic;   | Specify the path of the image input folder.  |
| -g, --ground_truth |  &radic;  |  &radic;   | Specify the validation ground truth ID file. |
### 4. Result
SqueezeNet model from `${TNN_ROOT_PATH}/model/SqueezeNet/squeezenet_v1.1.tnnproto`. 500 pictures with id ending in `00` (`ILSVRC2012_val_*00.JPEG`) are chosen for calibration.
|   blob_method    |  weight_method  |   merge_type    | Top-1 Accuracy |
| :--------------: | :-------------: | :-------------: | :------------: |
|       FP32       |      FP32       |        -        |     53.01%     |
|   0-(Min-Max)    |   0-(Min-Max)   | 0-(Per-Channel) |     52.71%     |
|   0-(Min-Max)    |   0-(Min-Max)   |     1-(Mix)     |     52.56%     |
|   0-(Min-Max)    |   0-(Min-Max)   | 2-(Per-Tensor)  |     47.29%     |
| 3-(ASY_MIN_MAX)  | 3-(ASY_MIN_MAX) | 0-(Per-Channel) |     52.78%     |
| 3-(ASY_MIN_MAX)  | 3-(ASY_MIN_MAX) |     1-(Mix)     |     52.94%     |
| 3-(ASY_MIN_MAX)  | 3-(ASY_MIN_MAX) | 2-(Per-Tensor)  |     47.93%     |
|  4-(ACIQ_GAUS)   |   0-(Min-Max)   | 0-(Per-Channel) |     52.76%     |
|  4-(ACIQ_GAUS)   |   0-(Min-Max)   |     1-(Mix)     |     52.63%     |
| 5-(ACIQ_LAPLACE) |   0-(Min-Max)   | 0-(Per-Channel) |     40.25%     |

### 5. Note
- defaut input scale/bias `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`
```
MatConvertParam ImageClassifier::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
    input_cvt_param.bias  = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
    return input_cvt_param;
}
```
- quantization_cmd mean `-n 123.675,116.28,103.53` and scale `-s 0.01712475383,0.0175070028,0.01742919389`



