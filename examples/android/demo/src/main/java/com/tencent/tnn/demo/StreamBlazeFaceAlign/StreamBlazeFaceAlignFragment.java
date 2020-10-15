package com.tencent.tnn.demo.StreamBlazeFaceAlign;

import android.hardware.Camera;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.tencent.tnn.demo.BlazeFaceDetector;
import com.tencent.tnn.demo.FaceAlign;
import com.tencent.tnn.demo.FaceDetector;
import com.tencent.tnn.demo.FaceInfo;
import com.tencent.tnn.demo.FileUtils;
import com.tencent.tnn.demo.Helper;
import com.tencent.tnn.demo.R;
import com.tencent.tnn.demo.common.component.CameraSetting;
import com.tencent.tnn.demo.common.component.DrawView;
import com.tencent.tnn.demo.common.fragment.BaseFragment;
import com.tencent.tnn.demo.common.sufaceHolder.DemoSurfaceHolder;

import java.io.IOException;


public class StreamBlazeFaceAlignFragment extends BaseFragment {

    private final static String TAG = StreamBlazeFaceAlignFragment.class.getSimpleName();

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

    private FaceAlign mFaceAlign = new FaceAlign();
    private boolean mIsDetectingFace = false;

    private ToggleButton mGPUSwitch;
    private boolean mUseGPU = false;
    //add for npu
    private ToggleButton mHuaweiNPUswitch;
    private boolean mUseHuaweiNpu = false;
    private TextView HuaweiNpuTextView;

    /**********************************     Get Preview Advised    **********************************/

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        System.loadLibrary("tnn_wrapper");
        //start SurfaceHolder
        mDemoSurfaceHolder = new DemoSurfaceHolder(this);
        String modelPath = initModel();
        NpuEnable = mFaceAlign.checkNpu(modelPath);
    }

    private String initModel()
    {

        String targetDir =  getActivity().getFilesDir().getAbsolutePath();

        //copy detect model to sdcard
        String[] detectModelPathsDetector = {
                "blazeface.tnnmodel",
                "blazeface.tnnproto"
        };

        for (int i = 0; i < detectModelPathsDetector.length; i++) {
            String modelFilePath = detectModelPathsDetector[i];
            String interModelFilePath = targetDir + "/" + modelFilePath ;
            FileUtils.copyAsset(getActivity().getAssets(), "blazeface/"+modelFilePath, interModelFilePath);
        }

        //copy detect model to sdcard
        String[] alignModelPathsDetector = {
                "youtu_face_alignment_phase1.tnnmodel",
                "youtu_face_alignment_phase1.tnnproto",
                "youtu_face_alignment_phase2.tnnmodel",
                "youtu_face_alignment_phase2.tnnproto"
        };

        for (int i = 0; i < alignModelPathsDetector.length; i++) {
            String modelFilePath = alignModelPathsDetector[i];
            String interModelFilePath = targetDir + "/" + modelFilePath ;
            FileUtils.copyAsset(getActivity().getAssets(), "youtu_face_alignment/"+modelFilePath, interModelFilePath);
        }

        FileUtils.copyAsset(getActivity().getAssets(),"blazeface_anchors.txt", targetDir + "/blazeface_anchors.txt");
        FileUtils.copyAsset(getActivity().getAssets(),"mean_pts_phase1.txt", targetDir + "/mean_pts_phase1.txt");
        FileUtils.copyAsset(getActivity().getAssets(),"mean_pts_phase2.txt", targetDir + "/mean_pts_phase2.txt");

        return targetDir;
    }

    @Override
    public void onClick(View view) {
        int i = view.getId();
        if (i == R.id.back_rl) {
            clickBack();
        }
    }
    private void restartCamera()
    {
        closeCamera();
        openCamera(mCameraFacing);
        startPreview(mSurfaceHolder);
    }
    private void onSwichGPU(boolean b)
    {
        if (b && mHuaweiNPUswitch.isChecked()) {
            mHuaweiNPUswitch.setChecked(false);
            mUseHuaweiNpu = false;
        }
        mUseGPU = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
        restartCamera();
    }

    private void onSwichNPU(boolean b)
    {
        if (b && mGPUSwitch.isChecked()) {
            mGPUSwitch.setChecked(false);
            mUseGPU = false;
        }
        mUseHuaweiNpu = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
        restartCamera();
    }

    private void clickBack() {
        if (getActivity() != null) {
            (getActivity()).finish();
        }
    }

    @Override
    public void setFragmentView() {
        Log.d(TAG, "setFragmentView");
        setView(R.layout.fragment_stream_detector);
        setTitleGone();
        $$(R.id.gpu_switch);
        $$(R.id.back_rl);
        mGPUSwitch = $(R.id.gpu_switch);
        mGPUSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                onSwichGPU(b);
            }
        });

        $$(R.id.npu_switch);
        mHuaweiNPUswitch = $(R.id.npu_switch);
        mHuaweiNPUswitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                onSwichNPU(b);
            }
        });
        HuaweiNpuTextView = $(R.id.npu_text);
        if (!NpuEnable) {
            HuaweiNpuTextView.setVisibility(View.INVISIBLE);
            mHuaweiNPUswitch.setVisibility(View.INVISIBLE);
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
        mIsDetectingFace = true;
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
            }
            else {

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
                    int ret = mFaceAlign.init(modelPath, mCameraHeight, mCameraWidth, 0.975f, 0.23f, 1, device);
                    if (ret == 0) {
                        mIsDetectingFace = true;
                    } else {
                        mIsDetectingFace = false;
                        Log.e(TAG, "Face detector init failed " + ret);
                    }
                } else {
                    Log.e(TAG, "Failed to init camera");
                }
            }
        }
        catch (Exception e) {
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
                        if (mIsDetectingFace) {
                            Camera.Parameters mCameraParameters = camera.getParameters();
                            FaceInfo[] faceInfoList = mFaceAlign.detectFromStream(data, mCameraParameters.getPreviewSize().width, mCameraParameters.getPreviewSize().height, mDrawView.getWidth(), mDrawView.getHeight(), mRotate);
                            Log.i(TAG, "detect from stream ret " + faceInfoList);
                            int faceCount = 0;
                            if (faceInfoList != null) {
                                faceCount = faceInfoList.length;
                            }
                            mDrawView.addFaceRect(faceInfoList);
                        }
                        else {
                            Log.i(TAG,"No face");
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
        mIsDetectingFace = false;
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
        mFaceAlign.deinit();
    }

}
