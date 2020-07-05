package com.tencent.tnn.demo;

import android.graphics.Bitmap;
import android.util.Log;

public class TNNLib {

    private long nativePtr;

    static {
        try {
            System.loadLibrary("tnn_wrapper");
        }catch(Exception e) {
        }catch(Error e) {
        } finally {
        }
    }

    public void setNativePtr(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    public long getNativePtr(){
        return nativePtr;
    }


    public TNNLib() {}

    public native int init(String protoFilePath, String modelFilePath, String device_type);

    public native float[] forward(Bitmap imageSrc);

    public native int deinit();

}
