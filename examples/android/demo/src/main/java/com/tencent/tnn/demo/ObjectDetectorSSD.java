package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class ObjectDetectorSSD {
    public static final String[] label_list = {
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"};

    public native int init(String modelPath, int width, int height, float scoreThreshold, float iouThreshold, int topk, int computeType);
    public native boolean checkNpu(String modelPath);
    public native int deinit();
    public native ObjectInfo[] detectFromStream(byte[] yuv420sp, int width, int height, int rotate);
    public native ObjectInfo[] detectFromImage(Bitmap bitmap, int width, int height);

}
