package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class HairSegmentation {
    public native int init(String modelPath, int width, int height, int computeType);
    public native boolean checkNpu(String modelPath);
    public native int deinit();
    public native int setHairColor(byte[] rgba);
    public native ImageInfo[] predictFromStream(byte[] yuv420sp, int width, int height, int rotate);
}