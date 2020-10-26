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

#import "TNNHairSegmentationViewModel.h"
#import "hair_segmentation.h"

#import "DYFlatButton.h"
#import "UIColor+Utility.h"

using namespace std;

@interface TNNHairSegmentationViewModel ()
@property (nonatomic, assign) vector<RGBA> colors;
@property (nonatomic, strong) NSArray<DYFlatButton *> *colorButtons;
@property (nonatomic, assign) RGBA active_color;
@end

@implementation TNNHairSegmentationViewModel

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认已经调整到release模式
    
    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/hair_segmentation/segmentation.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/hair_segmentation/segmentation.tnnproto"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }

    NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
    encoding:NSUTF8StringEncoding
       error:nil];
    string proto_content =
        protoFormat.UTF8String;
    NSData *data = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data length] > 0 ? string((const char *)[data bytes], [data length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }

    auto option = std::make_shared<HairSegmentationOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;

        option->mode = 1;
    }
        
    auto predictor = std::make_shared<HairSegmentation>();
    status = predictor->Init(option);
    
    BenchOption bench_option;
    bench_option.forward_count = 1;
    predictor->SetBenchOption(bench_option);
    
    //考虑多线程安全，最好初始化完全没问题后再赋值给成员变量
    //for muti-thread safety, copy to member variable after allocate
    self.predictor = predictor;
    
    // color blue
    [self setHairSegmentationRGBA:self.active_color];
    return status;
}


-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    return {};
}

-(ImageInfo)getImage:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    //std::shared_ptr<char> image_data = nullptr;
    ImageInfo image;
    if (sdk_output && dynamic_cast<HairSegmentationOutput *>(sdk_output.get())) {
        auto output = dynamic_cast<HairSegmentationOutput *>(sdk_output.get());
        //auto merged_image = output->merged_image;
        image = output->merged_image;
    }
    return image;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    return nil;
}

-(void) setHairSegmentationRGBA:(RGBA) color{
    if (self.predictor) {
        auto* predictor_cast = dynamic_cast<HairSegmentation *>(self.predictor.get());
        predictor_cast->SetHairColor(color);
    }
}

#pragma mark - UI control
- (void)setupCustomView:(UIView *)view
           layoutHeight:(NSLayoutConstraint *)viewLayoutHeight {
    self.colors = {
        //蓝色
        {0,0,185,90},
        //青色
        {0,185,185,40},
        //绿色
        {0,185,0,50},
        //紫色
        {185,0,185,64},
        //红色
        {185,0,0,64},
    };
    self.active_color = self.colors[0];
    
    viewLayoutHeight.constant = 60;
    
    //label
    auto label = [[UILabel alloc] initWithFrame:CGRectMake(15, 0, 80, viewLayoutHeight.constant)];
    label.font = [UIFont systemFontOfSize:14];
    label.text = @"头发颜色：";
    label.textColor = [UIColor whiteColor];
    [view addSubview:label];
    
    auto colorButtons = [NSMutableArray new];
    for (int i=0; i<self.colors.size(); i++) {
        auto color = self.colors[i];
        auto button = [[DYFlatButton alloc] initWithFrame:CGRectMake(15 + 80 + i*(36 + 12),
                                                                 12, 36, 36)];
        button.tag = i;
        
        button.autoHighlightMode = DYAutoHighlightModeAll;
        [button setBackgroundColor:[UIColor colorWithRed:color.r/255.0 green:color.g/255.0 blue:color.b/255.0 alpha:0.85]
                          forState:UIControlStateNormal];
        [button setBackgroundColor:[UIColor colorWithRed:color.r/255.0 green:color.g/255.0 blue:color.b/255.0 alpha:0.85]
                          forState:UIControlStateSelected];
        [button setBorderColor:[UIColor clearColor] forState:UIControlStateNormal];
        [button setBorderColor:[UIColor whiteColor] forState:UIControlStateSelected];
        [button addTarget:self action:@selector(onButtonClick:) forControlEvents:UIControlEventTouchUpInside];
        [view addSubview:button];
        [colorButtons addObject:button];
        button.selected = i == 0;
    }
    
    self.colorButtons = colorButtons;
}

- (void)onButtonClick:(DYFlatButton *)button {
    auto selected = button.selected;
    if (selected) {
        for (DYFlatButton *item in self.colorButtons) {
            item.selected = NO;
        }
    } else {
        for (DYFlatButton *item in self.colorButtons) {
            item.selected = NO;
        }
        button.selected = YES;
    }
    
    RGBA color = {0,0,0,0};
    if (button.selected) {
        color = self.colors[button.tag];
    }
    
    self.active_color = color;
    [self setHairSegmentationRGBA:color];
}

@end

