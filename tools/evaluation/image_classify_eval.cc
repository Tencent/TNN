#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>

#include <cstring>

#include "image_classifier_sdk_eval.h"
#include "macro.h"
#include "stb_image.h"
#include "utils.h"
// #include "stb_image_resize.h"
// #include "stb_image_write.h"
#include "tnn/utils/mat_converter_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn_sdk_sample.h"

using namespace TNN_NS;

void PrintConfig() {
    printf(
        "usage:\n./eval_cmd [-h] [-p] <proto file> [-m] <model file> [-i] <input folder> [-g] <groundtruth_file>\n"
        "\t-h, --help        \t show this message\n"
        "\t-p, --proto       \t(require) tnn proto file name\n"
        "\t-m, --model       \t(require) tnn model file name\n"
        "\t-i, --input_path  \t(require) the folder of input files\n"
        "\t-g, --ground_truth  \t(require) the validation ground truth ID file\n");
}

int main(int argc, char** argv) {
    // Init parameters
    std::string proto_file_name;
    std::string model_file_name;
    std::string input_path;
    std::string ground_truth;

    struct option long_options[] = {{"proto", required_argument, 0, 'p'},
                                    {"model", required_argument, 0, 'm'},
                                    {"input_path", required_argument, 0, 'i'},
                                    {"ground_truth", required_argument, 0, 'g'},
                                    {"help", no_argument, 0, 'h'},
                                    {0, 0, 0, 0}};

    const char* optstring = "p:m:i:g:h";

    if (argc == 1) {
        PrintConfig();
        return 0;
    }

    while (1) {
        int c = getopt_long(argc, argv, optstring, long_options, nullptr);
        if (c == -1)
            break;

        switch (c) {
            case 'p':
                printf("proto: %s\n", optarg);
                proto_file_name = optarg;
                break;
            case 'm':
                printf("model: %s\n", optarg);
                model_file_name = optarg;
                break;
            case 'i':
                printf("input path: %s\n", optarg);
                input_path = optarg;
                break;
            case 'g':
                printf("ground truth path: %s\n", optarg);
                ground_truth = optarg;
                break;
            case 'h':
            case '?':
                PrintConfig();
                return 0;
            default:
                PrintConfig();
                break;
        }
    }
    auto proto_content = fdLoadFile(proto_file_name);
    auto model_content = fdLoadFile(model_file_name);

    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path  = "";
        // option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        option->compute_units = TNN_NS::TNNComputeUnitsNaive;
#ifdef _CUDA_
        // option->compute_units = TNN_NS::TNNComputeUnitsTensorRT;
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
#endif
    }

    // Init predictor
    auto predictor = std::make_shared<ImageClassifierEval>();
    std::shared_ptr<TNNSDKOutput> sdk_output = predictor->CreateSDKOutput();
    CHECK_TNN_STATUS(predictor->Init(option));

    // read ground truth label id
    std::vector<int> groundTruthId;
    std::ifstream inputOs(ground_truth);
    std::string line;
    while (std::getline(inputOs, line)) {
        groundTruthId.emplace_back(std::atoi(line.c_str()));
    }

    // read image files
    int count = 0;
    std::vector<std::string> files;
    struct stat s;
    lstat(input_path.c_str(), &s);
    struct dirent* filename;
    DIR* dir;
    dir = opendir(input_path.c_str());
    while ((filename = readdir(dir)) != nullptr) {
        if (strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0) {
            continue;
        }
        files.push_back(filename->d_name);
        count++;
    }
    printf("total images:%d\n", count);
    std::sort(files.begin(), files.end());

    int tested_num   = 0;
    int top1         = 0;
    int batch_size   = predictor->GetInputShape()[0];
    auto target_dims = {batch_size, 3, 224, 224};
    auto image_dims  = {1, 3, 224, 224};

    // set openmp NumThreads 
    // CHECK_TNN_STATUS(predictor->SetOMPThreads(8));
    CHECK_TNN_STATUS(predictor->SetOMPThreads(batch_size));

    // Predict images
    for (int count_idx = 0; count_idx < count / batch_size; count_idx++) {
        std::shared_ptr<Mat> mat_new(new Mat(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, target_dims));
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            const auto image = input_path + "/" + files[count_idx * batch_size + batch_idx];
            int image_width, image_height, image_channel;
            unsigned char* data   = stbi_load(image.c_str(), &image_width, &image_height, &image_channel, 3);
            std::vector<int> nchw = {1, image_channel, image_height, image_width};
            auto input_mat        = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, data);
            auto target_mat =
                std::make_shared<TNN_NS::Mat>(input_mat->GetDeviceType(), input_mat->GetMatType(), image_dims);
            CHECK_TNN_STATUS(predictor->Resize(input_mat, target_mat, TNNInterpLinear));
            int bytesize_perbatch = DimsVectorUtils::Count(target_dims, 1) * GetMatElementSize(target_mat.get());
            memcpy((char*)mat_new->GetData() + batch_idx * bytesize_perbatch, target_mat->GetData(), bytesize_perbatch);
            free(data);
        }
        predictor->Predict(std::make_shared<TNNSDKInput>(mat_new), sdk_output);

        int* class_id;
        if (sdk_output && dynamic_cast<ImageClassifierEvalOutput*>(sdk_output.get())) {
            auto classfy_output = dynamic_cast<ImageClassifierEvalOutput*>(sdk_output.get());
            class_id            = classfy_output->class_id;
        }
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            if (class_id[batch_idx] == groundTruthId[count_idx * batch_size + batch_idx]) {
                top1++;
            }
        }
        tested_num += batch_size;
        if (count_idx % 2 == 0) {
            printf("==> tested:%d/%d, Top1: %f\n", tested_num, count, (float)top1 / (float)tested_num * 100.0);
        }
        free(class_id);
    }
    printf("==> Done! Final Top1: %f\n", (float)top1 / (float)tested_num * 100.0);
    return 0;
}
