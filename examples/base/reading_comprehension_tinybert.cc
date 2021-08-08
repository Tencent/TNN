#include "reading_comprehension_tinybert.h"
#include "sample_timer.h"
#include <cmath>

namespace TNN_NS{
    ReadingComprehensionTinyBertOutput::~ReadingComprehensionTinyBertOutput() {}

    ReadingComprehensionTinyBert::~ReadingComprehensionTinyBert() {}

    std::shared_ptr<Mat> ReadingComprehensionTinyBert::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                          std::string name) {
        return TNNSDKSample::ResizeToInputShape(input_mat, name);
    }

    std::shared_ptr<TNNSDKOutput> ReadingComprehensionTinyBert::CreateSDKOutput() {
        return std::make_shared<ReadingComprehensionTinyBertOutput>();
    }

    Status ReadingComprehensionTinyBert::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
        Status status = TNN_OK;

        auto output = dynamic_cast<ReadingComprehensionTinyBertOutput *>(output_.get());
        RETURN_VALUE_ON_NEQ(!output, false,
                            Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

        auto output_mat = output->GetMat();
        RETURN_VALUE_ON_NEQ(!output_mat, false,
                            Status(TNNERR_PARAM_ERR, "GetMat is invalid"));

        auto input_shape = GetInputShape();
        RETURN_VALUE_ON_NEQ(input_shape.size() == 2, true,
                            Status(TNNERR_PARAM_ERR, "GetInputShape is invalid"));

        return status;
    }


}