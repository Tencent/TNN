package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class ImageClassifyOpenGL {
    public native int init(String modelPath, int width, int height, int computeUnitType);
    public native int deinit();
    public native int[] detectFromImage(Bitmap image, int width, int height);
}