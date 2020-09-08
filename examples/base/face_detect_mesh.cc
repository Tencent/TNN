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

#include "face_detect_mesh.h"
#include "tnn/utils/dims_vector_utils.h"
#include "blazeface_detector.h"
#include "tnn_sdk_sample.h"
#include "face_mesh.h"

namespace TNN_NS {
Status FaceDetectMesh::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    if (sdks.size() < 2) {
        return Status(TNNERR_INST_ERR, "FaceDetectAligner::Init has invalid sdks, its size < 2");
    }

    predictor_detect_ = sdks[0];
    predictor_mesh_ = sdks[1];
    return TNNSDKComposeSample::Init(sdks);
}

Status FaceDetectMesh::Predict(std::shared_ptr<TNNSDKInput> sdk_input,
                                  std::shared_ptr<TNNSDKOutput> &sdk_output) {
    Status status = TNN_OK;

    if (!sdk_input || sdk_input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    auto predictor_detect_async = predictor_detect_;
    auto predictor_mesh_async = predictor_mesh_;
    auto predictor_detect_cast = dynamic_cast<BlazeFaceDetector *>(predictor_detect_async.get());
    auto predictor_mesh_cast = dynamic_cast<Facemesh *>(predictor_mesh_async.get());

    auto image_mat = sdk_input->GetMat();
    const int image_orig_height = image_mat->GetHeight();
    const int image_orig_width = image_mat->GetWidth();

    // output of each model
    std::shared_ptr<TNNSDKOutput> sdk_output_face = nullptr;
    std::shared_ptr<TNNSDKOutput> sdk_output_mesh = nullptr;

    std::vector<BlazeFaceInfo> face_list;
    // phase1: face detector
    {
        status = predictor_detect_cast->Predict(std::make_shared<BlazeFaceDetectorInput>(image_mat), sdk_output_face);
        RETURN_ON_NEQ(status, TNN_OK);

        if (sdk_output_face && dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output_face.get())) {
            auto face_output = dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output_face.get());
            face_list = face_output->face_list;
        }
        if(face_list.size() <= 0) {
            //no faces, return
            LOGD("Error no faces found!\n");
            return status;
        }
    }

    int crop_height = -1, crop_width = -1;
    int crop_x = -1, crop_y = -1;
    // phase2: face mesh
    {
        // devan: only consider the 1st face
        auto face = face_list[0];
        auto face_orig = face.AdjustToViewSize(image_orig_height, image_orig_width, 2);
        //1.5*crop
        crop_height = 1.5 * (face_orig.y2 - face_orig.y1);
        crop_width  = 1.5 * (face_orig.x2 - face_orig.x1);
        crop_x = std::max(0.0, face_orig.x1 - 0.25 * crop_width);
        crop_y = std::max(0.0, face_orig.y1 - 0.25 * crop_height);
        crop_width  = std::min(crop_width,  image_orig_width-crop_x);
        crop_height = std::min(crop_height, image_orig_height-crop_y);

        DimsVector crop_dims = {1, 3, static_cast<int>(crop_height), static_cast<int>(crop_width)};
        std::shared_ptr<TNN_NS::Mat> croped_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, crop_dims);
        status = predictor_mesh_cast->Crop(image_mat, croped_mat, crop_x, crop_y);
        RETURN_ON_NEQ(status, TNN_OK);

        status = predictor_mesh_cast->Predict(std::make_shared<FacemeshInput>(croped_mat), sdk_output_mesh);
        RETURN_ON_NEQ(status, TNN_OK);
    }
    //get output
    {
        if (sdk_output_mesh && dynamic_cast<FacemeshOutput *>(sdk_output_mesh.get())) {
            auto face_output = dynamic_cast<FacemeshOutput *>(sdk_output_mesh.get());
            auto face_mesh_list = face_output->face_list;

            if (face_mesh_list.size() <= 0 ){
                LOGD("Error no faces found!\n");
                return status;
            }
            // only consider the first result
            auto face_mesh = face_mesh_list[0];
            auto face_mesh_cropped = face_mesh.AdjustToViewSize(crop_height, crop_width, 2);
            face_mesh_cropped = face_mesh_cropped.AddOffset(crop_x, crop_y);

            // set result
            sdk_output = std::make_shared<FacemeshOutput>();
            auto output = dynamic_cast<FacemeshOutput *>(sdk_output.get());
            face_mesh_cropped.image_width = image_orig_width;
            face_mesh_cropped.image_height = image_orig_height;
            output->face_list.push_back(*reinterpret_cast<FacemeshInfo*>(&face_mesh_cropped));
        }
    }
    return TNN_OK;
}
}
