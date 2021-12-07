package com.tencent.tnn.demo.StreamHairSegmentation;

import android.hardware.Camera;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.RadioGroup;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.tencent.tnn.demo.HairSegmentation;
import com.tencent.tnn.demo.FpsCounter;
import com.tencent.tnn.demo.ImageInfo;
import com.tencent.tnn.demo.FileUtils;
import com.tencent.tnn.demo.Helper;
import com.tencent.tnn.demo.R;
import com.tencent.tnn.demo.common.component.CameraSetting;
import com.tencent.tnn.demo.common.component.DrawView;
import com.tencent.tnn.demo.common.fragment.BaseFragment;
import com.tencent.tnn.demo.common.sufaceHolder.DemoSurfaceHolder;

import java.io.IOException;


public class StreamHairSegmentationFragment extends BaseFragment {

    private final static String TAG = StreamHairSegmentationFragment.class.getSimpleName();

    /**********************************     Define    **********************************/

    private SurfaceView mPreview;

    private DrawView mDrawView;
    private int mCameraWidth;
    private int mCameraHeight;

    Camera mOpenedCamera;
    int mOpenedCameraId = 0;
    DemoSurfaceHolder mDemoSurfaceHolder = null;

    int mCameraFacing = -1;
    int mRotate = -1;
    SurfaceHolder mSurfaceHolder;

    private HairSegmentation mHairSegmentation = new HairSegmentation();
    private FpsCounter mFpsCounter = new FpsCounter();
    private boolean mIsSegmentingHair = false;
    private boolean mIsCountFps = false;
    private RadioGroup color_button;

    private ToggleButton mGPUSwitch;
    private boolean mUseGPU = false;
    //add for npu
    private ToggleButton mHuaweiNPUswitch;
    private boolean mUseHuaweiNpu = false;
    private TextView HuaweiNpuTextView;

    private boolean mDeviceSwitched = false;
    private byte[] mColor = {(byte)0, (byte)0, (byte)185, (byte)90};

    /**********************************     Get Preview Advised    **********************************/

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        System.loadLibrary("tnn_wrapper");
        //start SurfaceHolder
        mDemoSurfaceHolder = new DemoSurfaceHolder(this);
        String modelPath = initModel();
        NpuEnable = mHairSegmentation.checkNpu(modelPath);
    }

    private String initModel() {
        String targetDir =  getActivity().getFilesDir().getAbsolutePath();

        // copy segmentation model to sdcard
        String[] modelPathsSegmentation = {
                "segmentation.tnnmodel",
                "segmentation.tnnproto",
        };

        for (int i = 0; i < modelPathsSegmentation.length; i++) {
            String modelFilePath = modelPathsSegmentation[i];
            String interModelFilePath = targetDir + "/" + modelFilePath ;
            FileUtils.copyAsset(getActivity().getAssets(), "hair_segmentation/"+modelFilePath, interModelFilePath);
        }
        return targetDir;
    }

    @Override
    public void onClick(View view) {
        int i = view.getId();
        if (i == R.id.back_rl) {
            clickBack();
        }
    }

    private void restartCamera() {
        closeCamera();
        openCamera(mCameraFacing);
        startPreview(mSurfaceHolder);
    }

    private void onSwitchGPU(boolean b) {
        if (b && mHuaweiNPUswitch.isChecked()) {
            mHuaweiNPUswitch.setChecked(false);
            mUseHuaweiNpu = false;
        }
        mUseGPU = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
        mDeviceSwitched = true;
    }

    private void onSwitchNPU(boolean b) {
        if (b && mGPUSwitch.isChecked()) {
            mGPUSwitch.setChecked(false);
            mUseGPU = false;
        }
        mUseHuaweiNpu = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
        mDeviceSwitched = true;
    }

    private void clickBack() {
        if (getActivity() != null) {
            (getActivity()).finish();
        }
    }

    @Override
    public void setFragmentView() {
        Log.d(TAG, "setFragmentView");
        setView(R.layout.fragment_stream_hair_segmentation);
        setTitleGone();
        $$(R.id.gpu_switch);
        $$(R.id.back_rl);
        mGPUSwitch = $(R.id.gpu_switch);
        mGPUSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                onSwitchGPU(b);
            }
        });

        $$(R.id.npu_switch);
        mHuaweiNPUswitch = $(R.id.npu_switch);
        mHuaweiNPUswitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                onSwitchNPU(b);
            }
        });
        HuaweiNpuTextView = $(R.id.npu_text);
        if (!NpuEnable) {
            HuaweiNpuTextView.setVisibility(View.INVISIBLE);
            mHuaweiNPUswitch.setVisibility(View.INVISIBLE);
        }

        RadioButton initBtn = $(R.id.button_blue);
        initBtn.setSelected(true);
        final int[] colorList = {R.id.button_blue, R.id.button_cyan, R.id.button_green, R.id.button_purple, R.id.button_red};
        for (int j = 0; j < colorList.length; j++) {
            RadioButton button = $(colorList[j]);
            button.setOnClickListener(new View.OnClickListener(){
                @Override
                public void onClick(View v) {
                    RadioButton btn = $(v.getId());
                    boolean selected = btn.isSelected();
                    // init hair color to black
                    byte[] color = {(byte)0, (byte)0, (byte)0, (byte)0};
                    mColor = color;
                    if (!selected) {
                        switch (v.getId()) {
                            case R.id.button_blue:
                                mColor[0] = (byte)0;
                                mColor[1] = (byte)0;
                                mColor[2] = (byte)185;
                                mColor[3] = (byte)90;
                                break;
                            case R.id.button_cyan:
                                mColor[0] = (byte)0;
                                mColor[1] = (byte)185;
                                mColor[2] = (byte)185;
                                mColor[3] = (byte)40;
                                break;
                            case R.id.button_green:
                                mColor[0] = (byte)0;
                                mColor[1] = (byte)185;
                                mColor[2] = (byte)0;
                                mColor[3] = (byte)50;
                                break;
                            case R.id.button_purple:
                                mColor[0] = (byte)185;
                                mColor[1] = (byte)0;
                                mColor[2] = (byte)185;
                                mColor[3] = (byte)64;
                                break;
                            case R.id.button_red:
                                mColor[0] = (byte)185;
                                mColor[1] = (byte)0;
                                mColor[2] = (byte)0;
                                mColor[3] = (byte)64;
                        }
                    }
                    mHairSegmentation.setHairColor(mColor);
                    btn.setSelected(!selected);

                    for (int j = 0; j < colorList.length; j++) {
                        if (v.getId() != colorList[j]) {
                            RadioButton tmpBtn = $(colorList[j]);
                            tmpBtn.setSelected(false);
                        }
                    }
                }
            });
        }

        init();
    }

    private void init() {
        mPreview = $(R.id.live_detection_preview);

        Button btnSwitchCamera = $(R.id.switch_camera);
        btnSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                closeCamera();
                if (mCameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                    openCamera(Camera.CameraInfo.CAMERA_FACING_BACK);
                }
                else {
                    openCamera(Camera.CameraInfo.CAMERA_FACING_FRONT);
                }
                startPreview(mSurfaceHolder);
            }
        });

        mDrawView = (DrawView) $(R.id.drawView);
    }

    @Override
    public void onStart() {
        Log.d(TAG, "onStart");
        super.onStart();
        if (null != mDemoSurfaceHolder) {
            SurfaceHolder holder = mPreview.getHolder();
            holder.setKeepScreenOn(true);
            mDemoSurfaceHolder.setSurfaceHolder(holder);
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");

        getFocus();
        preview();
    }

    @Override
    public void onPause() {
        Log.d(TAG, "onPause");
        super.onPause();
    }

    @Override
    public void onStop() {
        Log.i(TAG, "onStop");
        super.onStop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.i(TAG, "onDestroy");
    }

    private void preview() {
        Log.i(TAG, "preview");
    }

    private void getFocus() {
        getView().setFocusableInTouchMode(true);
        getView().requestFocus();
        getView().setOnKeyListener(new View.OnKeyListener() {
            @Override
            public boolean onKey(View v, int keyCode, KeyEvent event) {
                if (event.getAction() == KeyEvent.ACTION_UP && keyCode == KeyEvent.KEYCODE_BACK) {
                    clickBack();
                    return true;
                }
                return false;
            }
        });
    }

/**********************************     Camera    **********************************/


    public void openCamera() {
        openCamera(Camera.CameraInfo.CAMERA_FACING_FRONT);
    }

    private void openCamera(int cameraFacing) {
        mIsSegmentingHair = true;
        mCameraFacing = cameraFacing;
        try {
            int numberOfCameras = Camera.getNumberOfCameras();
            if (numberOfCameras < 1) {
                Log.e(TAG, "no camera device found");
            } else if (1 == numberOfCameras) {
                mOpenedCamera = Camera.open(0);
                mOpenedCameraId = 0;
            } else {
                Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                for (int i = 0; i < numberOfCameras; i++) {
                    Camera.getCameraInfo(i, cameraInfo);
                    if (cameraInfo.facing == cameraFacing) {
                        mOpenedCamera = Camera.open(i);
                        mOpenedCameraId = i;
                        break;
                    }
                }
            }
            if (mOpenedCamera == null) {
//                popTip("can't find camera","");
                Log.e(TAG, "can't find camera");
            } else {
                int r = CameraSetting.initCamera(getActivity().getApplicationContext(),mOpenedCamera,mOpenedCameraId);
                if (r == 0) {
                    //设置摄像头朝向
                    CameraSetting.setCameraFacing(cameraFacing);

                    Camera.Parameters parameters = mOpenedCamera.getParameters();
                    mRotate = CameraSetting.getRotate(getActivity().getApplicationContext(), mOpenedCameraId, mCameraFacing);
                    mCameraWidth = parameters.getPreviewSize().width;
                    mCameraHeight = parameters.getPreviewSize().height;
                    String modelPath = initModel();
                    int device = 0;
                    if (mUseHuaweiNpu) {
                        device = 2;
                    } else if (mUseGPU) {
                        device = 1;
                    }
                    int ret = mHairSegmentation.init(modelPath, mCameraHeight, mCameraWidth, device);
                    if (ret == 0) {
                        mIsSegmentingHair = true;
                    } else {
                        mIsSegmentingHair = false;
                        Log.e(TAG, "Hair Segmentation init failed " + ret);
                    }

                    ret = mFpsCounter.init();
                    if (ret == 0) {
                        mIsCountFps = true;
                    } else {
                        mIsCountFps = false;
                        Log.e(TAG, "Fps Counter init failed " + ret);
                    }
                } else {
                    Log.e(TAG, "Failed to init camera");
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "open camera failed:" + e.getLocalizedMessage());
        }
    }

    public void startPreview(SurfaceHolder surfaceHolder) {
        try {
            if (null != mOpenedCamera) {
                Log.i(TAG, "start preview, is previewing");
                mOpenedCamera.setPreviewCallback(new Camera.PreviewCallback() {
                    @Override
                    public void onPreviewFrame(byte[] data, Camera camera) {
                        if (mIsSegmentingHair) {
                            Camera.Parameters mCameraParameters = camera.getParameters();
                            ImageInfo[] imageInfoList;
                            // reinit
                            if (mDeviceSwitched) {
                                String modelPath = getActivity().getFilesDir().getAbsolutePath();
                                int device = 0;
                                if (mUseHuaweiNpu) {
                                    device = 2;
                                } else if (mUseGPU) {
                                    device = 1;
                                }
                                int ret = mHairSegmentation.init(modelPath, mCameraHeight, mCameraWidth, device);
                                if (ret == 0) {
                                    mIsSegmentingHair = true;
                                    mHairSegmentation.setHairColor(mColor);
                                    mFpsCounter.init();
                                } else {
                                    mIsSegmentingHair = false;
                                    Log.e(TAG, "Hair Segmentation init failed " + ret);
                                }
                                mDeviceSwitched = false;
                            }
                            if (mIsCountFps) {
                                mFpsCounter.begin("HairSegmentation");
                            }
                            imageInfoList = mHairSegmentation.predictFromStream(data, mCameraParameters.getPreviewSize().width, mCameraParameters.getPreviewSize().height, mRotate);
                            if (mIsCountFps) {
                                mFpsCounter.end("HairSegmentation");
                                double fps = mFpsCounter.getFps("HairSegmentation");
                                String monitorResult = "device: ";
                                if (mUseGPU) {
                                    monitorResult += "opencl\n";
                                } else if (mUseHuaweiNpu) {
                                    monitorResult += "huawei_npu\n";
                                } else {
                                    monitorResult += "arm\n";
                                }
                                monitorResult += "fps: " + String.format("%.02f", fps);
                                TextView monitor_result_view = (TextView)$(R.id.monitor_result);
                                monitor_result_view.setText(monitorResult);
                            }
                            Log.i(TAG, "predict from stream ret " + imageInfoList);
                            mDrawView.addImageInfo(imageInfoList[1]);
                        }
                        else {
                            Log.i(TAG,"No Hair Segmentating");
                        }
                    }
                });
                mOpenedCamera.setPreviewDisplay(surfaceHolder);
                mOpenedCamera.startPreview();
                mSurfaceHolder = surfaceHolder;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (RuntimeException e) {
            e.printStackTrace();
        }
    }

    public void closeCamera() {
        Log.i(TAG, "closeCamera");
        mIsSegmentingHair = false;
        if (mOpenedCamera != null) {
            try {
                mOpenedCamera.stopPreview();
                mOpenedCamera.setPreviewCallback(null);
                Log.i(TAG, "stop preview, not previewing");
            } catch (Exception e) {
                e.printStackTrace();
                Log.i(TAG, "Error setting camera preview: " + e.toString());
            }
            try {
                mOpenedCamera.release();
                mOpenedCamera = null;
            } catch (Exception e) {
                e.printStackTrace();
                Log.i(TAG, "Error setting camera preview: " + e.toString());
            } finally {
                mOpenedCamera = null;
            }
        }
        mHairSegmentation.deinit();
    }
}
