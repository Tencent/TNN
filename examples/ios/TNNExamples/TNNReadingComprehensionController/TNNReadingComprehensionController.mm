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

#import "TNNReadingComprehensionController.h"
#import "bert_tokenizer.h"
#import "UIImage+Utility.h"
#import <Metal/Metal.h>
#import <cstdlib>
#import <sstream>
#import <string>
#import <tnn/tnn.h>

using namespace std;
using namespace TNN_NS;

@interface TNNReadingComprehensionController ()

@property (weak, nonatomic) IBOutlet UIButton *btnExample;
@property (weak, nonatomic) IBOutlet UILabel *labelResult;
@property (weak, nonatomic) IBOutlet UILabel *labelContext;

@property (strong, nonatomic) NSString *context;
@property (strong, nonatomic) NSString *question;

@end

@implementation TNNReadingComprehensionController
;

- (void)viewDidLoad {
    [super viewDidLoad];
}

- (void)viewWillAppear:(BOOL) animated {
    [super viewWillAppear:animated];
    /*
     Testcast1:
     std::string context = "TNN: A high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab. It also has many outstanding advantages such as cross-platform, high performance, model compression, and code tailoring. The TNN framework further strengthens the support and performance optimization of mobile devices on the basis of the original Rapidnet and ncnn frameworks. At the same time, it refers to the high performance and good scalability characteristics of the industry's mainstream open source frameworks, and expands the support for X86 and NV GPUs. On the mobile phone, TNN has been used by many applications such as mobile QQ, weishi, and Pitu. As a basic acceleration framework for Tencent Cloud AI, TNN has provided acceleration support for the implementation of many businesses. Everyone is welcome to participate in the collaborative construction to promote the further improvement of the TNN inference framework.";
     std::string question = "what advantages does TNN have?"; // cross-platform, high performance, model compression, and code tailoring
     std::string question = "who developed TNN?"; // Tencent Youtu Lab
     std::string question = "which applications use TNN?"; // mobile QQ, weishi, and Pitu
     std::string question = "what is TNN?"; // A high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab

     Testcase2:
     std::string context = "Pumas are large, cat-like animals which are found in America. When reports came into London Zoo that a wild puma had been spotted forty-five miles south of London, they were not taken seriously. However, as the evidence began to accumulate, experts from the Zoo felt obliged to investigate, for the descriptions given by people who claimed to have seen the puma were extraordinarily similar.";
     std::string question = "where is puma from?";  // America
     std::string question = "where has puma been spotted?"; // forty-five miles south of London"
     std::string question = "what are pumas?"; // large, cat-like animals

     Testcase3:
     std::string context = "This paper introduces TIRAMISU, a polyhedral compiler with a scheduling language. TIRAMISU is designed not only for the area of deep learning but also for the areas of image processing and tensor algebra. TIRAMISU relies on the flexible polyhedral representation which allows many advanced capabilities such as expressing complex code transformations, expressing non-rectangular iteration spaces and performing dependence analysis to check the correctness of transformations all of which are difficult to express in non-polyhedral compilers.";
     std::string question = "what is tiramisu?";  // a polyhedral compiler with a scheduling language
     std::string question = "what are tiramisu designed for?"; // deep learning
     */
    self.context = [NSString stringWithUTF8String:"TNN: A high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab. It also has many outstanding advantages such as cross-platform, high performance, model compression, and code tailoring. The TNN framework further strengthens the support and performance optimization of mobile devices on the basis of the original Rapidnet and ncnn frameworks. At the same time, it refers to the high performance and good scalability characteristics of the industry's mainstream open source frameworks, and expands the support for X86 and NV GPUs. On the mobile phone, TNN has been used by many applications such as mobile QQ, weishi, and Pitu. As a basic acceleration framework for Tencent Cloud AI, TNN has provided acceleration support for the implementation of many businesses. Everyone is welcome to participate in the collaborative construction to promote the further improvement of the TNN inference framework."];
    self.question = [NSString stringWithUTF8String:"what is TNN?"];

    self.labelContext.text = [NSString stringWithFormat:@"Context:\n%s", [self.context UTF8String]];
    self.labelResult.text = [NSString stringWithFormat:@"Q:%s", [self.question UTF8String]];
}
- (void)onSwitchChanged:(id)sender {
    self.labelResult.text = [NSString stringWithFormat:@"Q:%s", [self.question UTF8String]];
}

- (IBAction)onBtnTNNExamples:(id)sender {
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认已经调整到release模式
    
    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/tiny-bert/tiny-bert-squad-fixed-256.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/tiny-bert/tiny-bert-squad-fixed-256.tnnproto"
                                                          ofType:nil];
    auto vocab_path = [[NSBundle mainBundle] pathForResource:@"model/tiny-bert/vocab.txt"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0 || vocab_path.length <= 0) {
        self.labelResult.text = @"proto or model or vocab path is invalid";
        NSLog(@"Error: proto or model or vocab path is invalid");
        return;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        self.labelResult.text = @"proto or model path is invalid";
        NSLog(@"Error: proto or model path is invalid");
        return;
    }

    auto units = [self getComputeUnitsForIndex:self.switchDevice.selectedSegmentIndex];
    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
    }

    auto predictor = std::make_shared<TNNSDKSample>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
            return;
    }
    auto tokenizer = std::make_shared<BertTokenizer>();
    tokenizer->Init([vocab_path UTF8String]);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
            return;
    }

    BenchOption bench_option;
    bench_option.forward_count = 10;
    predictor->SetBenchOption(bench_option);
    
    auto actual_units = predictor->GetComputeUnits();
    auto bertInput = std::make_shared<BertTokenizerInput>(DEVICE_ARM, "input_ids", "attention_mask", "token_type_ids");
    auto bertOutput = predictor->CreateSDKOutput();
    tokenizer->buildInput([self.context UTF8String], [self.question UTF8String], bertInput);
    status = predictor->Predict(bertInput, bertOutput);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }

    std::string ans;
    tokenizer->ConvertResult(bertOutput, "output_0", "output_1", ans);

    auto bench_result     = predictor->GetBenchResult();
    self.labelResult.text = [NSString stringWithFormat:@"Q:%s\nA: %s\ndevice: %@     time:\n%s",
                             [self.question UTF8String],
                             ans.c_str(),
                             [self getNSSTringForComputeUnits:actual_units],
                             bench_result.Description().c_str()];
}

@end

