package com.tencent.tnn.demo.ImageClassifyDetector;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.tencent.tnn.demo.FileUtils;
import com.tencent.tnn.demo.Helper;
import com.tencent.tnn.demo.ImageClassify;
import com.tencent.tnn.demo.R;
import com.tencent.tnn.demo.common.fragment.BaseFragment;

import java.util.ArrayList;


public class ImageClassifyDetectFragment extends BaseFragment {

    private final static String TAG = ImageClassifyDetectFragment.class.getSimpleName();
    private ImageClassify mImageClassify = new ImageClassify();

    private static final String IMAGE = "tiger_cat.jpg";
    private static final String RESULT_LIST = "synset.txt";
    private static final int NET_INPUT = 224;
    private ToggleButton mGPUSwitch;
    private Button mRunButton;
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
        String modelPath = initModel();
        NpuEnable = mImageClassify.checkNpu(modelPath);
    }

    private String initModel() {
        String targetDir =  getActivity().getFilesDir().getAbsolutePath();

        //copy detect model to sdcard
        String[] modelPathsDetector = {
                "squeezenet_v1.1.tnnmodel",
                "squeezenet_v1.1.tnnproto",
        };

        for (int i = 0; i < modelPathsDetector.length; i++) {
            String modelFilePath = modelPathsDetector[i];
            String interModelFilePath = targetDir + "/" + modelFilePath ;
            FileUtils.copyAsset(getActivity().getAssets(), "SqueezeNet/"+modelFilePath, interModelFilePath);
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

    private void onSwitchGPU(boolean b) {
        if (b && mHuaweiNPUswitch.isChecked()) {
            mHuaweiNPUswitch.setChecked(false);
            mUseHuaweiNpu = false;
        }
        mUseGPU = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
    }

    private void onSwitchNPU(boolean b) {
        if (b && mGPUSwitch.isChecked()) {
            mGPUSwitch.setChecked(false);
            mUseGPU = false;
        }
        mUseHuaweiNpu = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
    }

    private void clickBack() {
        if (getActivity() != null) {
            (getActivity()).finish();
        }
    }

    @Override
    public void setFragmentView() {
        Log.d(TAG, "setFragmentView");
        setView(R.layout.fragment_image_detector);
        setTitleGone();
        $$(R.id.back_rl);
        $$(R.id.gpu_switch);
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

        mRunButton = $(R.id.run_button);
        mRunButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startDetect();
            }
        });

        final Bitmap originBitmap = FileUtils.readBitmapFromFile(getActivity().getAssets(), IMAGE);
        ImageView source = (ImageView)$(R.id.origin);
        source.setImageBitmap(originBitmap);
    }

    @Override
    public void openCamera() {

    }

    @Override
    public void startPreview(SurfaceHolder surfaceHolder) {

    }

    @Override
    public void closeCamera() {

    }

    private void startDetect() {
        ArrayList<String> result_list = FileUtils.ReadListFromFile(getActivity().getAssets(), RESULT_LIST);
        final Bitmap originBitmap = FileUtils.readBitmapFromFile(getActivity().getAssets(), IMAGE);

        final Bitmap scaleBitmap = Bitmap.createScaledBitmap(originBitmap, NET_INPUT, NET_INPUT, false);
        ImageView source = (ImageView)$(R.id.origin);
        source.setImageBitmap(originBitmap);
        String modelPath = initModel();
        Log.d(TAG, "Init classify " + modelPath);
        int device = 0;
        if (mUseHuaweiNpu) {
            device = 2;
        } else if (mUseGPU) {
            device = 1;
        }
        int result = mImageClassify.init(modelPath, NET_INPUT, NET_INPUT, device);
        if (result == 0) {
            Log.d(TAG, "detect from image");
            int [] indexArray= mImageClassify.detectFromImage(scaleBitmap, NET_INPUT, NET_INPUT);
            Log.d(TAG, "detect from image result " + result + " index :" + indexArray);
            if (indexArray != null && indexArray.length > 0) {
                Log.d(TAG, "detect index " + indexArray[0]);
                String resultText = "result: " + result_list.get(indexArray[0]) + " " + Helper.getBenchResult();
                TextView result_view = (TextView)$(R.id.result);
                result_view.setText(resultText);
            }
            mImageClassify.deinit();
        } else {
            Log.e(TAG, "failed to init model " + result);
        }
    }

    @Override
    public void onStart() {
        Log.d(TAG, "onStart");
        super.onStart();
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");

        getFocus();
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
}
