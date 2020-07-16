package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class FaceDetector {
    public static class FaceInfo {
        public float x1;
        public float y1;
        public float x2;
        public float y2;
        public float score;
        public float[] landmarks;
    }
    public native int init(String modelPath, int width, int height, float scoreThreshold, float iouThreshold, int topk, int computeType);
    public native boolean checkNpu(String modelPath);
    public native int deinit();
    public native FaceInfo[] detectFromStream(byte[] yuv420sp, int width, int height, int rotate);
    public native FaceInfo[] detectFromImage(Bitmap bitmap, int width, int height);
}