//  Copyright © 2020 tencent. All rights reserved.

#include "tnn_fps_counter.h"
#include "sample_timer.h"

const std::string kFPSCounterDefaultTag = "fps.default.tag";

TNNFPSCounter::TNNFPSCounter(){
    map_fps_ = {};
    map_start_time_ = {};
}

std::string TNNFPSCounter::RetifiedTag(std::string tag) {
    return tag = tag.length() <= 0 ? kFPSCounterDefaultTag : tag;
}

void TNNFPSCounter::Begin(std::string tag) {
    tag = RetifiedTag(tag);
    
    map_timer["tag"] = std::make_shared<TNN_NS::SampleTimer>();
    map_timer["tag"]->Start(); 
}

void TNNFPSCounter::End(std::string tag) {
    tag = RetifiedTag(tag);
    
    map_timer["tag"]->Stop();
    double time = map_timer["tag"]->GetTime();
    
    if (time > 0.1) {
        double fps = GetFPS(tag);
        double fps_current = 1000.0 / time;
        const double smoothing = 0.75;
        fps = smoothing*fps + (1 - smoothing)*fps_current;
        //smoothing time
        double time_history = GetTime(tag);
        double smoothing_time = smoothing*time_history + (1-smoothing)*time;
        
        map_fps_[tag] = fps;
        map_time_[tag] = smoothing_time;
    }
}

double TNNFPSCounter::GetFPS(std::string tag) {
    tag = RetifiedTag(tag);
    
    if (map_fps_.find(tag) != map_fps_.end()) {
        return map_fps_[tag];
    }
    return 0;
}

double TNNFPSCounter::GetTime(std::string tag) {
    tag = RetifiedTag(tag);

    if (map_time_.find(tag) != map_time_.end()) {
        return map_time_[tag];
    }
    return 0;
}

std::map<std::string, double> TNNFPSCounter::GetAllFPS() {
    std::map<std::string, double> map_all;
    for (auto iter : map_fps_) {
        map_all[RetifiedTag(iter.first)] = iter.second;
    }
    return map_all;
}

std::map<std::string, double> TNNFPSCounter::GetAllTime() {
    std::map<std::string, double> map_all;
    for(auto iter : map_time_) {
        map_all[RetifiedTag(iter.first)] = iter.second;
    }
    return map_all;
}

double TNNFPSCounter::GetStartTime(std::string tag) {
    tag = RetifiedTag(tag);
    
    if (map_start_time_.find(tag) != map_start_time_.end()) {
        return map_start_time_[tag];
    }
    return 0;
}
