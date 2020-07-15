package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class ImageClassify {
    public native int init(String modelPath, int width, int height, int computeUnitType);
    public native boolean checkNpu(String modelPath);
    public native int deinit();
    public native int[] detectFromImage(Bitmap image, int width, int height);
}
