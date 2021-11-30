package com.tencent.tnn.demo;

import android.graphics.Bitmap;

public class ReadingComprehensionTinyBert {
    public native int init(String modelPath, int computeUnitType);
    public native int deinit();
    public native String ask(String modelPath, String material, String question);
}
