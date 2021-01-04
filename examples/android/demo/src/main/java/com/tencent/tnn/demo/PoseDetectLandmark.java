package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class PoseDetectLandmark {

    public native int init(String modelPath, int computeType);
    public native boolean checkNpu(String modelPath);
    public native int deinit();
    public native ObjectInfo[] detectFromStream(byte[] yuv420sp, int width, int height, int view_width, int view_height, int rotate, int detector_type);

}
