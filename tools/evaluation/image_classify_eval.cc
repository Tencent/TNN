#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>

#include <cstring>

#include "image_classifier.h"
#include "macro.h"
#include "stb_image.h"
#include "utils.h"
#include "tnn/utils/omp_utils.h"
// #include "stb_image_resize.h"
// #include "stb_image_write.h"

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
    omp_set_num_threads(8);

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
        option->compute_units = TNN_NS::TNNComputeUnitsTensorRT;
#endif
    }


    // Init predictor
    auto predictor = std::make_shared<ImageClassifier>();
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

    int tested_num = 0;
    int top1     = 0;

    // Predict images
    for (int i=0; i< count; i++) {
        const auto& file = files[i];
        const auto image = input_path + "/" + file;

        int image_width, image_height, image_channel;
        unsigned char* data   = stbi_load(image.c_str(), &image_width, &image_height, &image_channel, 3);
        std::vector<int> nchw = {1, image_channel, image_height, image_width};
        auto image_mat        = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, data);
        predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output);

        int class_id = -1;
        if (sdk_output && dynamic_cast<ImageClassifierOutput*>(sdk_output.get())) {
            auto classfy_output = dynamic_cast<ImageClassifierOutput*>(sdk_output.get());
            class_id            = classfy_output->class_id;
        }
        if (class_id == groundTruthId[i]) {
            top1++;
        }
        if (++tested_num % 10 == 0) {
            printf("==> tested:%d/%d, Top1: %f\n", tested_num, count, (float)top1 / (float)tested_num * 100.0);
        }
        free(data);
    }
    printf("==> Done! Final Top1: %f\n", (float)top1 / (float)tested_num * 100.0);
    return 0;
}
