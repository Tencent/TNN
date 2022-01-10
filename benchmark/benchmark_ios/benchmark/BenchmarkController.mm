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

#import "BenchmarkController.h"
#import <tnn/tnn.h>
#include <fstream>
#include <cmath>
#include <sys/time.h>
#include <float.h>
#include <sstream>


using namespace std;
using namespace TNN_NS;

struct BenchModel {
    string name;
    string tnn_proto_content;
    string tnn_model_content;
    string coreml;
};

struct BenchOption {
    int warm_count = 10;
    int forward_count = 20;
    int create_count = 1;
    
    string description() {
        ostringstream ostr;
        ostr << "create_count = " << create_count
        << "  warm_count = " << warm_count
        << "  forward_count = " << forward_count;
        
        ostr << std::endl;
        return ostr.str();
    };
};

struct BenchResult {
    Status status;
    
    //time
    float min = FLT_MAX;
    float max = FLT_MIN;
    float avg = 0;
    float total = 0;
    int count = 0;
    
    float diff = 0;
    
    int addTime(float time){
        count++;
        total += time;
        min = std::min(min, time);
        max = std::max(max, time);
        avg = total/count;
        return 0;
    };
    
    string description() {
        ostringstream ostr;
        ostr << "min = " << min << "  max = " << max << "  avg = " <<avg;
        
        if (status != TNN_OK) {
            ostr << "\nerror = "<<status.description();
        }
        ostr << std::endl;
        
        return ostr.str();
    };
};

@interface BenchmarkController () {
}
@property (nonatomic, weak) IBOutlet UIButton *btnBenchmark;
@property (nonatomic, weak) IBOutlet UITextView *textViewResult;
@end

@implementation BenchmarkController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    
}

- (vector<BenchModel>)getAllModels {
    NSString *modelZone = [[NSBundle mainBundle] pathForResource:@"model"
                                                          ofType:nil];
    NSArray *modelList = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:modelZone
                                                                             error:nil];
    
    NSPredicate *predicateProto = [NSPredicate predicateWithFormat:@"self ENDSWITH 'tnnproto'"];
    NSPredicate *predicateModel = [NSPredicate predicateWithFormat:@"self ENDSWITH 'tnnmodel'"];
    NSPredicate *predicateCoreML = [NSPredicate predicateWithFormat:@"self ENDSWITH 'mlmodel'"];
    NSPredicate *predicateCoreMLC = [NSPredicate predicateWithFormat:@"self ENDSWITH 'mlmodelc'"];
    
    vector<BenchModel> netmodels;
    
    for (NSString *modelDir in modelList) {
//        if (![modelDir hasPrefix:@"mobilenetv1-ssd"]) {
//            continue;
//        }
       NSString *modelDirPath = [modelZone stringByAppendingPathComponent:modelDir];
       BOOL isDirectory = NO;

       if ([[NSFileManager defaultManager] fileExistsAtPath:modelDirPath
                                                isDirectory:&isDirectory]) {
           if (!isDirectory) {
               continue;
           }
           
           NSComparator sort = ^(NSString *obj1,NSString *obj2){
               auto range = NSMakeRange(0,obj1.length);
               return [obj1 compare:obj2 options:NSCaseInsensitiveSearch|NSNumericSearch|
                       NSWidthInsensitiveSearch|NSForcedOrderingSearch range:range];
           };
           
           BenchModel model;
           model.name = modelDir.UTF8String;
           
           NSArray *modelFiles = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:modelDirPath
                                                                                     error:nil];
           NSArray<NSString *> *protos = [[modelFiles filteredArrayUsingPredicate:predicateProto] sortedArrayUsingComparator:sort];
           NSArray<NSString *> *models = [[modelFiles filteredArrayUsingPredicate:predicateModel] sortedArrayUsingComparator:sort];
           
           //support multiple models in the same directory
           for (size_t index = 0; index < std::min(protos.count, models.count); index++) {
               auto proto_prefix = [protos[index] substringToIndex:protos[index].length - @".tnnproto".length];
               auto model_prefix = [models[index] substringToIndex:models[index].length - @".tnnmodel".length];
               if (![proto_prefix isEqualToString:model_prefix]) {
                   continue;
               }
               model.name = proto_prefix.UTF8String;
               
               auto proto = [NSString stringWithContentsOfFile:[modelDirPath stringByAppendingPathComponent:protos[index]]
                                                        encoding:NSUTF8StringEncoding
                                                           error:nil];
               if (proto.length > 0) {
                   model.tnn_proto_content = proto.UTF8String;
                   
    //               model.tnn_model_content = [modelDirPath stringByAppendingPathComponent:models[0]].UTF8String;
                   NSData *data = [NSData dataWithContentsOfFile:[modelDirPath
                                                                  stringByAppendingPathComponent:models[index]]];
                   model.tnn_model_content = string((const char *)[data bytes], [data length]);
               }
               
               netmodels.push_back(model);
           }

           NSArray<NSString *> *coremls = [[modelFiles filteredArrayUsingPredicate:predicateCoreML] sortedArrayUsingComparator:sort];
           if (coremls.count > 0) {
               model.tnn_proto_content = "";
               model.tnn_model_content = "";
               model.coreml = [modelDirPath stringByAppendingPathComponent:coremls[0]].UTF8String;
               netmodels.push_back(model);
           }
           coremls = [modelFiles filteredArrayUsingPredicate:predicateCoreMLC];
           if (coremls.count > 0) {
               model.tnn_proto_content = "";
               model.tnn_model_content = "";
               model.coreml = [modelDirPath stringByAppendingPathComponent:coremls[0]].UTF8String;
               netmodels.push_back(model);
           }
       }
    }
    return netmodels;
}

- (IBAction)onBtnBenchmark:(id)sender {
    //check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认意见调整到release模式
    
    //搜索model目录下的所有模型
    auto allModels = [self getAllModels];
    
    BenchOption option;
    option.warm_count = 5;
    option.forward_count = 10;
    option.create_count = 1;
    
    //Get metallib path from app bundle
    //PS：A script(Build Phases -> Run Script) is added to copy the metallib file in tnn framework project to benchmark app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto pathLibrary = [[NSBundle mainBundle] pathForResource:@"tnn.metallib"
                                                       ofType:nil];
    pathLibrary = pathLibrary ? pathLibrary : @"";
    
    NSString *allResult = [NSString string];
    for (auto model : allModels) {
        NSLog(@"model: %s", model.name.c_str());
        allResult = [allResult stringByAppendingFormat:@"model: %s\n", model.name.c_str()];
    
        //benchmark on cpu
        auto result_cpu = [self benchmarkWithProtoContent:model.tnn_proto_content
                                                model:model.tnn_model_content
                                               coreml:model.coreml
                                              library:pathLibrary.UTF8String
                                              netType:NETWORK_TYPE_DEFAULT
                                              deviceType:DEVICE_ARM
                                               option:option];
        NSLog(@"cpu: \ntime: %s", result_cpu.description().c_str());
        allResult = [allResult stringByAppendingFormat:@"cpu: \ntime: %s\n",
                     result_cpu.description().c_str()];

        //benchmark on gpu
        auto result_gpu = [self benchmarkWithProtoContent:model.tnn_proto_content
                                                model:model.tnn_model_content
                                               coreml:model.coreml
                                              library:pathLibrary.UTF8String
                                              netType:NETWORK_TYPE_DEFAULT
                                              deviceType:DEVICE_METAL
                                               option:option];
        NSLog(@"gpu: \ntime: %s", result_gpu.description().c_str());
        allResult = [allResult stringByAppendingFormat:@"gpu: \ntime: %s\n",
                     result_gpu.description().c_str()];

        //benchmark on npu
        auto result_npu = [self benchmarkWithProtoContent:model.tnn_proto_content
                                                model:model.tnn_model_content
                                               coreml:model.coreml
                                              library:pathLibrary.UTF8String
                                              netType:NETWORK_TYPE_COREML
                                              deviceType:DEVICE_APPLE_NPU
                                               option:option];
        NSLog(@"npu: \ntime: %s", result_npu.description().c_str());
        allResult = [allResult stringByAppendingFormat:@"npu: \ntime: %s\n",
                     result_npu.description().c_str()];
        
    }
    
    self.textViewResult.text = allResult;
}

- (BenchResult)benchmarkWithProtoContent:(string)protoContent
                                   model:(string)modelPathOrContent
                                  coreml:(string)coremlDir
                                 library:(string)metallibPath
                                 netType:(NetworkType)net_type
                              deviceType:(DeviceType)device_type
                                  option:(BenchOption)option {
    BenchResult result;
    
    net_type = net_type == NETWORK_TYPE_COREML ? NETWORK_TYPE_COREML : NETWORK_TYPE_DEFAULT;
    
    //network init
    //网络初始化
    TNN net;
    {
        ModelConfig config;
        if (protoContent.length() > 0 && modelPathOrContent.length() > 0) {
            config.model_type = MODEL_TYPE_TNN;
            config.params = {protoContent, modelPathOrContent};
        } else if (coremlDir.length() > 0) {
            config.model_type = MODEL_TYPE_COREML;
            config.params = {coremlDir};
        }
        
        result.status = net.Init(config);
        if (result.status != TNN_OK) {
            NSLog(@"net.Init Error: %s", result.status.description().c_str());
            return result;
        }
    }
    
    //create instance
    //创建实例instance
    std::shared_ptr<TNN_NS::Instance> instance = nullptr;
    {
        NetworkConfig network_config;
        network_config.network_type = net_type;
        network_config.library_path = {metallibPath};
        network_config.device_type =  device_type;
        instance = net.CreateInst(network_config, result.status);
        if (result.status != TNN_OK || !instance) {
            NSLog(@"net.CreateInst Error: %s", result.status.description().c_str());
            return result;
        }
        
    }
    
    //warm cpu, only used when benchmark
    for (int cc=0; cc<option.warm_count; cc++) {
        result.status = instance->Forward();
        if (result.status != TNN_OK) {
            NSLog(@"instance.Forward Error: %s", result.status.description().c_str());
            return result;
        }
    }
    
    //inference
    //前向推断
    bool profile_layer_time = false;
#if TNN_PROFILE
    if (profile_layer_time) {
        instance->StartProfile();
    }
#endif
    for (int cc=0; cc<option.forward_count; cc++) {
        timeval tv_begin, tv_end;
        gettimeofday(&tv_begin, NULL);
        
        result.status = instance->Forward();
        
        gettimeofday(&tv_end, NULL);
        double elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
        result.addTime(elapsed);
    }
#if TNN_PROFILE
    if (profile_layer_time) {
        instance->FinishProfile(true);
    }
#endif

    return result;
}

@end


