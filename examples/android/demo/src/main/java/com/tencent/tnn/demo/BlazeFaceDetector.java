package com.tencent.tnn.demo;

import android.graphics.Bitmap;
import android.util.Pair;

public class BlazeFaceDetector {
    public static class BlazeFaceInfo  extends FaceDetector.FaceInfo {
//        public Pair<Float, Float>[] keypoints;
        public float[][] keypoints;
    }

    public native int init(String modelPath, int width, int height, float scoreThreshold, float iouThreshold, int topk, int computeType);
    public native boolean checkNpu(String modelPath);
    public native int deinit();
    public native BlazeFaceInfo[] detectFromStream(byte[] yuv420sp, int width, int height, int rotate);
    public native BlazeFaceInfo[] detectFromImage(Bitmap bitmap, int width, int height);

}
