package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class ObjectDetector {
    public static class ObjectInfo {
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
    public native ObjectInfo[] detectFromStream(byte[] yuv420sp, int width, int height, int rotate);
    public native ObjectInfo[] detectFromImage(Bitmap bitmap, int width, int height);

}