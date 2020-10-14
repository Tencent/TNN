/*
* @Author: Dandi Ding
* @Date:   2020-10-14 11:26:15
* @Last Modified by:   Dandiding
* @Last Modified time: 2020-10-14 14:19:46
*/

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <memory>
#include <stdio.h>

#include "tnn_sdk_sample.h"
#include "worker.h"
#include "utils/utils.h"
#include "macro.h"

using namespace TNN_NS;

// #define FAKE_FRAME

int main(int argc, char** argv)
{
    cv::Mat frame;
    
#ifndef FAKE_FRAME
    cv::VideoCapture cap;

    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    cap.open(deviceID, apiID);
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
#else
    frame = cv::imread("1.jpg");
#endif

    Worker worker;
    auto status = worker.Init("../../../model/");
    if (status != TNN_OK) {
        LOGERROR(status);
        return -1;
    }

    while(true)
    {

#ifndef FAKE_FRAME
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
#endif
        cv::Mat frame_rgb;
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);

        cv::Mat frame_paint = frame.clone();
        BREAK_ON_NEQ(worker.FrocessFrame(frame_rgb, frame_paint), TNN_OK);

#ifdef FAKE_FRAME
        cv::imwrite("result.jpg", frame_paint);
        break;
#else
        cv::imshow("Live", frame_paint);
        int key = cv::waitKey(5);
        if (key == 'c')
            break;
#endif

    }

    return 0;
}