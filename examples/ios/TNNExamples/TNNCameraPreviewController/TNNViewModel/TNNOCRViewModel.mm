// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#import "TNNOCRViewModel.h"

#if HAS_OPENCV

#import "ocr_textbox_detector.h"
#import "ocr_angle_predictor.h"
#import "ocr_text_recognizer.h"
#import "ocr_driver.h"

using namespace std;

@interface TNNOCRViewModel ()
@property (nonatomic, strong) UILabel *label;
@property (nonatomic, assign) std::array<std::shared_ptr<TNNSDKSample>, 3> ocrPredictors;
@end

@implementation TNNOCRViewModel

- (std::shared_ptr<OCRTextboxDetector>) loadTextboxDetector:(TNNComputeUnits)units {
    std::shared_ptr<OCRTextboxDetector> predictor = nullptr;
    
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/chinese-ocr/dbnet.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/chinese-ocr/dbnet.tnnproto"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    auto option = std::make_shared<OCRTextboxDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;

        option->padding = 10;
        option->box_score_threshold = 0.6f;
        option->scale_down_ratio    = 0.75f;
    }
        
    predictor = std::make_shared<OCRTextboxDetector>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

- (std::shared_ptr<OCRAnglePredictor>) loadAnglePredictor:(TNNComputeUnits)units {
    std::shared_ptr<OCRAnglePredictor> predictor = nullptr;
    
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/chinese-ocr/angle_net.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/chinese-ocr/angle_net.tnnproto"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
    }
        
    predictor = std::make_shared<OCRAnglePredictor>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

- (std::shared_ptr<OCRTextRecognizer>)loadTextRecognizer:(TNNComputeUnits)units {
    std::shared_ptr<OCRTextRecognizer> predictor = nullptr;
    
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/chinese-ocr/crnn_lite_lstm.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/chinese-ocr/crnn_lite_lstm.tnnproto"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }
    auto vocab_path = [[NSBundle mainBundle] pathForResource:@"model/chinese-ocr/keys.txt"
                                                      ofType:nil];
    if (vocab_path.length <= 0) {
        NSLog(@"Error: vocabulary file path is invalid");
        return predictor;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    auto option = std::make_shared<OCRTextRecognizerOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
        
        option->vocab_path = vocab_path.UTF8String;
    }
        
    predictor = std::make_shared<OCRTextRecognizer>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }

    return predictor;
}

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    auto textbox_detector = [self loadTextboxDetector:units];
    RETURN_VALUE_ON_NEQ(!textbox_detector,
                        false,
                        Status(TNNERR_MODEL_ERR,
                               "loadTextboxDetector failed: pls make sure the pose detect model is downloaded"));
    
    auto angle_predictor = [self loadAnglePredictor:units];
    RETURN_VALUE_ON_NEQ(!angle_predictor,
                            false,
                            Status(TNNERR_MODEL_ERR,
                                   "loadAnglePredictor failed: pls make sure the pose landmark model is downloaded"));
    auto text_recognizer = [self loadTextRecognizer:units];
    RETURN_VALUE_ON_NEQ(!text_recognizer,
                         false,
                         Status(TNNERR_MODEL_ERR,
                                "loadTextRecognizer failed: pls make sure the pose landmark model is downloaded"));

    self.ocrPredictors = {textbox_detector, angle_predictor, text_recognizer};

    auto predictor = std::make_shared<OCRDriver>();
    status = predictor->Init({textbox_detector, angle_predictor, text_recognizer});
    RETURN_ON_NEQ(status, TNN_OK);

    self.predictor = predictor;

    return status;
}

-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    std::vector<std::shared_ptr<ObjectInfo> > text_list;
    if (sdk_output && dynamic_cast<OCROutput *>(sdk_output.get())) {
        auto ocr_output = dynamic_cast<OCROutput *>(sdk_output.get());
        for(int i=0; i<ocr_output->texts.size(); ++i) {
            auto textbox = std::make_shared<ObjectInfo>();

            // fill ObjectInfo lines to support angles
            auto box_offset = ocr_output->box.begin() + i * 4;
            auto box_end    = box_offset + 4;
            textbox->key_points.insert(textbox->key_points.begin(),
                                       box_offset, box_end);
            textbox->lines = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

            textbox->x1 = box_offset->first;
            textbox->x2 = box_offset->first;
            textbox->y1 = box_offset->second;
            textbox->y2 = box_offset->second;
            for(const auto&p:textbox->key_points) {
                textbox->x1 = std::min(textbox->x1, p.first);
                textbox->x2 = std::max(textbox->x2, p.first);
                textbox->y1 = std::min(textbox->y1, p.second);
                textbox->y2 = std::max(textbox->y2, p.second);
            }

            textbox->image_width = ocr_output->image_width;
            textbox->image_height = ocr_output->image_height;

            textbox->label = ocr_output->texts[i].c_str();
            text_list.push_back(textbox);
        }
    }

    return text_list;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    return nil;
}

@end

#endif
