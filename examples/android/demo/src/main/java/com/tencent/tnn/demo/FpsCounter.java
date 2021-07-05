package com.tencent.tnn.demo;

public class FpsCounter {
    public native int init();
    public native int deinit();
    public native int begin(String tag);
    public native int end(String tag);
    public native double getFps(String tag);
    public native double getTime(String tag);
}
