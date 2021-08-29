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
SqueezeNet model result from `${TNN_ROOT_PATH}/model/SqueezeNet/squeezenet_v1.1.tnnproto`
| blob_method | weight_method |   merge_type    | Top-1 Accuracy |
| :---------: | :-----------: | :-------------: | :------------: |
|    FP32     |     FP32      |      FP32       |     53.01%     |
| 0-(Min-Max) |  0-(Min-Max)  | 0-(Per-Channel) |                |
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

