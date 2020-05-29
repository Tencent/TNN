package com.tencent.tnn.demo.common.activity;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;

public abstract class DemoBaseActivity extends Activity {
    private static final String TAG = "DemoBaseActivity";
    private static final int FACE_PERMISSION_QUEST_CAMERA = 1024;

    public void askForPermission() {
        //检测权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "didnt get permission,ask for it!");
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO, Manifest.permission.READ_PHONE_STATE, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    FACE_PERMISSION_QUEST_CAMERA);
        } else {
            updateUI();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                           int[] grantResults) {
        switch (requestCode) {
            case FACE_PERMISSION_QUEST_CAMERA:
                //If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0) {
                    if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                        Log.i(TAG, "get camera permission!");
                        if (grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                            //permission was granted，yay！Do the
                            // mic-related task u need to do.
                            Log.i(TAG, "get mic permission!");
                            if (grantResults[2] == PackageManager.PERMISSION_GRANTED) {
                                Log.i(TAG, "get read_phone permission!");
                                Log.i(TAG, "get all permission! Go on Verify!");
                                if (grantResults[3] == PackageManager.PERMISSION_GRANTED) {
                                    updateUI();
                                    return;
                                }else {
                                    askReadPhonePermissionError();
                                    return;
                                }
                            } else {
                                askReadPhonePermissionError();
                                return;
                            }
                        } else {
                            askAudioPermissionError();
                            return;
                        }
                    } else {
                        askCameraPermissionError();
                        return;
                    }
                }
                break;
        }
    }

    public abstract void updateUI();

    private void askCameraPermissionError() {
        Log.e(TAG, "Didn't get camera permission!");
        String msg = "用户没有授权相机权限";
        askPermissionError(msg);
    }

    private void askAudioPermissionError() {
        Log.e(TAG, "Didn't get mic permission!");
        String msg = "用户没有授权录音权限";
        askPermissionError(msg);
    }

    private void askReadPhonePermissionError() {
        Log.e(TAG, "Didn't get read_phone permission!");
        String msg = "用户没有授权读取手机状态权限";
        askPermissionError(msg);
    }

    private void askPermissionError(String msg) {
        Log.w(TAG,"设备授权验证失败");
        finish();
    }

}
