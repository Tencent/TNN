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

#import "TNNFacemeshViewModel.h"
#import "BlazeFaceDetector.h"
#import "UIImage+Utility.h"

#import <Metal/Metal.h>
#import <memory>

using namespace std;

@implementation TNNFacemeshViewModel

- (std::shared_ptr<BlazeFaceDetector>) loadFaceDetector:(TNNComputeUnits)units {
    std::shared_ptr<BlazeFaceDetector> predictor = nullptr;
    
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface.tnnmodel"
                                                      ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface.tnnproto"
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
    //blazeface requires input with shape 128*128
    const int target_height = 128;
    const int target_width  = 128;
    DimsVector target_dims  = {1, 3, target_height, target_width};
    
    auto option = std::make_shared<BlazeFaceDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        
        option->input_width = target_width;
        option->input_height = target_height;
        //min_score_thresh
        option->min_score_threshold = 0.75;
        //min_suppression_thresh
        option->min_suppression_threshold = 0.3;
        //predefined anchor file path
        option->anchor_path = string([[[NSBundle mainBundle] pathForResource:@"blazeface_anchors.txt" ofType:nil] UTF8String]);
    }
    
    predictor = std::make_shared<BlazeFaceDetector>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

- (std::shared_ptr<Facemesh>) loadFacemesh:(TNNComputeUnits)units {
    std::shared_ptr<Facemesh> predictor = nullptr;
    
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/face_mesh/face_mesh.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/face_mesh/face_mesh.tnnproto"
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
    
    auto option = std::make_shared<FacemeshOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        
        //TODO: set parameters
        option->face_presence_threshold = 0.1;
        option->flip_vertically = false;
        option->flip_horizontally = false;
        option->norm_z = 1.0f;
        option->ignore_rotation = false;
    }
    
    predictor = std::make_shared<Facemesh>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    auto face_detector = [self loadFaceDetector:units];
    auto face_mesh = [self loadFacemesh:units];
    
    self.predictor = face_detector;
    self.face_mesh = face_mesh;
    
    return status;
}

-(Status)Run:(std::shared_ptr<char>)image_data
             height:(int) height
             width :(int) width
             output:(std::shared_ptr<TNNSDKOutput>&) sdk_output
            counter:(std::shared_ptr<TNNFPSCounter>) counter {
    Status status = TNN_OK;
    
    //for muti-thread safety, increase ref count, to insure predictor is not released while detecting object
    auto face_detector = self.predictor;
    auto face_mesh = self.face_mesh;
    
    auto units = self.predictor->GetComputeUnits();
    
    const int image_orig_height = height;
    const int image_orig_width  = width;
    TNN_NS::DimsVector orig_image_dims = {1, 3, image_orig_height, image_orig_width};

    counter->Begin("Copy");
    // mat for the input image
    shared_ptr<TNN_NS::Mat> image_mat = nullptr;
    // construct image_mat
    if (units == TNNComputeUnitsGPU) {
        image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, orig_image_dims);

        id<MTLTexture> texture_rgba = (__bridge id<MTLTexture>)image_mat->GetData();
        if (!texture_rgba) {
            status = Status(TNNERR_NO_RESULT, "Error texture input rgba is nil");
            return status;
        }
        [texture_rgba replaceRegion:MTLRegionMake2D(0, 0, orig_image_dims[3], orig_image_dims[2])
                        mipmapLevel:0
                          withBytes:image_data.get()
                        bytesPerRow:orig_image_dims[3] * 4];
    } else if (units == TNNComputeUnitsCPU) {
        image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, orig_image_dims, image_data.get());
    }
    counter->End("Copy");

    counter->Begin("Phase1");
    std::vector<BlazeFaceInfo> face_info;
    // phase1: face detector
    {
        auto input_shape = face_detector->GetInputShape();
        auto input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), N8UC4, input_shape);
        face_detector->Resize(image_mat, input_mat, TNNInterpLinear);
        
        std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
        status = face_detector->Predict(std::make_shared<BlazeFaceDetectorInput>(input_mat), sdk_output);

        if (status != TNN_OK) {
            return status;
        }
        
        if (sdk_output && dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output.get()))
        {
            auto face_output = dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output.get());
            face_info = face_output->face_list;
        }
        if(face_info.size() <= 0) {
            //no faces, return
            NSLog(@"Error no faces found!");
            return status;
        }
    }
    counter->End("Phase1");

    counter->Begin("Phase2");
    // phase2: face mesh
    {
        // devan: only consider the 1st face
        auto face = face_info[0];
        auto face_orig = face.AdjustToViewSize(image_orig_height, image_orig_width, 2);
        //1.5*crop
        int crop_h = face_orig.y2 - face_orig.y1;
        int crop_w = face_orig.x2 - face_orig.x1;
        auto crop_rect = CGRectMake(face_orig.x1-0.25*crop_w,
                                    face_orig.y1-0.25*crop_h,
                                    1.5*crop_w,
                                    1.5*crop_h);
        crop_rect.origin.x    = std::max(0.0, crop_rect.origin.x);
        crop_rect.origin.y    = std::max(0.0, crop_rect.origin.y);
        crop_rect.size.width  = std::min(crop_rect.size.width,  image_orig_width-crop_rect.origin.x);
        crop_rect.size.height = std::min(crop_rect.size.height, image_orig_height-crop_rect.origin.y);
        DimsVector crop_dims = {1, 3, static_cast<int>(crop_rect.size.height), static_cast<int>(crop_rect.size.width)};
        std::shared_ptr<TNN_NS::Mat> croped_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, crop_dims);
        status = face_detector->Crop(image_mat, croped_mat, crop_rect.origin.x, crop_rect.origin.y);
        if (status != TNN_OK) {
            return status;
        }
        
        auto input_shape = face_mesh->GetInputShape();
        std::shared_ptr<TNN_NS::Mat> input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, input_shape);
        status = face_detector->Resize(croped_mat, input_mat, TNNInterpLinear);
        if (status != TNN_OK) {
            return status;
        }
        
        std::shared_ptr<TNNSDKOutput> facemesh_output = nullptr;
        status = face_mesh->Predict(std::make_shared<FacemeshInput>(input_mat), facemesh_output);
        if (status != TNN_OK) {
            return status;
        }
        
        if (facemesh_output && dynamic_cast<FacemeshOutput *>(facemesh_output.get()))
        {
            auto face_output = dynamic_cast<FacemeshOutput *>(facemesh_output.get());
            auto face_mesh_list = face_output->face_list;
            
            if (face_mesh_list.size() <=0 ){
                // no faces, return
                return status;
            }
            // only consider the first result
            auto face_mesh = face_mesh_list[0];
            auto face_mesh_cropped = face_mesh.AdjustToViewSize(crop_rect.size.height, crop_rect.size.width, 2);
            face_mesh_cropped = face_mesh_cropped.AddOffset(crop_rect.origin.x, crop_rect.origin.y);
            // set result
            sdk_output = std::make_shared<FacemeshOutput>();
            auto output = dynamic_cast<FacemeshOutput *>(sdk_output.get());
            output->face_list.push_back(*reinterpret_cast<FacemeshInfo*>(&face_mesh_cropped));
        }
    }
    counter->End("Phase2");
    return status;
}

-(ObjectInfo)getFace:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    ObjectInfo face;
    if (sdk_output && dynamic_cast<FacemeshOutput *>(sdk_output.get())) {
        auto face_output = dynamic_cast<FacemeshOutput *>(sdk_output.get());
        face = face_output->face_list[0];
    }
    return face;
}

@end
