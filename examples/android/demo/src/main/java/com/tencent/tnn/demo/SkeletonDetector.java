package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class SkeletonDetector {

    public native int init(String modelPath, int width, int height, int computeType);
    public native boolean checkNpu(String modelPath);
    public native int deinit();
    public native ObjectInfo[] detectFromStream(byte[] yuv420sp, int width, int height, int view_width, int view_height, int rotate, int detector_type);

}
