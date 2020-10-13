package com.tencent.tnn.demo;

import android.graphics.Bitmap;
import android.util.Pair;

public class BlazeFaceDetector {
    public native int init(String modelPath, int width, int height, float scoreThreshold, float iouThreshold, int topk, int computeType);
    public native boolean checkNpu(String modelPath);
    public native int deinit();
    public native FaceInfo[] detectFromStream(byte[] yuv420sp, int width, int height, int view_width, int view_height, int rotate);
    public native FaceInfo[] detectFromImage(Bitmap bitmap, int width, int height);

}
