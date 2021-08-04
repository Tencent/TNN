package com.tencent.tnn.demo.ImageOCRDetector;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.tencent.tnn.demo.OCRDetector;
import com.tencent.tnn.demo.ObjectInfo;
import com.tencent.tnn.demo.FileUtils;
import com.tencent.tnn.demo.Helper;
import com.tencent.tnn.demo.R;
import com.tencent.tnn.demo.common.component.DrawView;
import com.tencent.tnn.demo.common.fragment.BaseFragment;

import java.util.ArrayList;


public class ImageOCRDetectFragment extends BaseFragment {
    private static final String IMAGES_PATH = Environment.getExternalStorageDirectory().getPath() + "/saveBitmaps";
    private int index = 43;
    private static final String IMAGE = "test_ocr.jpg";

    private final static String TAG = ImageOCRDetectFragment.class.getSimpleName();
    private OCRDetector mOCRDetector = new OCRDetector();
    private Bitmap originBitmap;

    private Paint mPaint = new Paint();
    private DrawView mDrawView;
    private ImageView mImageView;
    private ToggleButton mGPUSwitch;
    private Button mRunButton;
    private boolean mUseGPU = false;
    //add for npu
    private boolean mUseHuaweiNpu = false;

    /**********************************     Get Preview Advised    **********************************/

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        System.loadLibrary("tnn_wrapper");
        String modelPath = initModel();
        NpuEnable = mOCRDetector.checkNpu(modelPath);
    }

    private String initModel() {

        String targetDir = getActivity().getFilesDir().getAbsolutePath();

        // copy ocr related models to sdcard
        String[] modelPathsDetector = {
                "angle_net.tnnmodel",
                "angle_net.tnnproto",
                "crnn_lite_lstm.tnnmodel",
                "crnn_lite_lstm.tnnproto",
                "dbnet.tnnmodel",
                "dbnet.tnnproto",
                "keys.txt",
        };

        for (int i = 0; i < modelPathsDetector.length; i++) {
            String modelFilePath = modelPathsDetector[i];
            String interModelFilePath = targetDir + "/" + modelFilePath;
            FileUtils.copyAsset(getActivity().getAssets(), "chinese-ocr/" + modelFilePath, interModelFilePath);
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

    private void onSwichGPU(boolean b) {

        mUseGPU = b;
        TextView result_view = (TextView) $(R.id.result);
        result_view.setText("");
    }

    private void onSwichNPU(boolean b) {
        if (b && mGPUSwitch.isChecked()) {
            mGPUSwitch.setChecked(false);
            mUseGPU = false;
        }
        mUseHuaweiNpu = b;
        TextView result_view = (TextView) $(R.id.result);
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
                onSwichGPU(b);
            }
        });

        mDrawView = (DrawView) $(R.id.drawView);
        mRunButton = $(R.id.run_button);
        mRunButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startDetect();
            }
        });

        ((Button) $(R.id.btn_next)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                nextImage();
                startDetect();
            }
        });

        originBitmap = FileUtils.readBitmapFromFile(getActivity().getAssets(), IMAGE);
        mImageView = (ImageView) $(R.id.origin);
        mImageView.setImageBitmap(originBitmap);

        ImageView source = (ImageView) $(R.id.origin);
        source.setImageBitmap(originBitmap);

    }

    private void nextImage() {
        String path = IMAGES_PATH + String.format("/image-%05d.jpeg", ++index);
        originBitmap = FileUtils.readBitmapFromSDCard(path);
        mImageView.setImageBitmap(originBitmap);
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
        Bitmap scaleBitmap = Bitmap.createScaledBitmap(originBitmap, originBitmap.getWidth(), originBitmap.getHeight(), false);
        String modelPath = initModel();
        Log.d(TAG, "Init classify " + modelPath);
        int device = 0;
        if (mUseHuaweiNpu) {
            device = 2;
        } else if (mUseGPU) {
            device = 1;
        }
        int result = mOCRDetector.init(modelPath, originBitmap.getWidth(), originBitmap.getHeight(), device);
        String txt_result = "text result:\n";
        if (result == 0) {
            Log.d(TAG, "detect from image");
            ObjectInfo[] objectInfoList = mOCRDetector.detectFromImage(scaleBitmap, originBitmap.getWidth(), originBitmap.getHeight());
            Log.d(TAG, "detect from image result " + objectInfoList);
            int objectCount = 0;
            if (objectInfoList != null) {
                objectCount = objectInfoList.length;
            }
            if (objectInfoList != null && objectInfoList.length > 0) {
                Log.d(TAG, "detect object size " + objectInfoList.length);

                mPaint.setARGB(255, 255, 0, 0);
                mPaint.setStrokeWidth(3);
                mPaint.setFilterBitmap(true);
                mPaint.setStyle(Paint.Style.STROKE);
                mPaint.setTextAlign(Paint.Align.CENTER);
                mPaint.setTextSize(30);
                Bitmap scaleBitmap2 = originBitmap.copy(Bitmap.Config.ARGB_8888, true);
                Canvas canvas = new Canvas(scaleBitmap2);
                ArrayList<float[]> point_lines_list = new ArrayList<float[]>();
                ArrayList<String> labels = new ArrayList<String>();
                for (int i = 0; i < objectInfoList.length; i++) {
                    float[] point_lines = new float[4 * 4];
                    if (objectInfoList[i]==null || objectInfoList[i].key_points == null || objectInfoList[i].key_points.length == 0) {
                        Log.e("julis", "objectInfoList[i].key_points.length == 0");
                        continue;
                    }
                    point_lines[0] = objectInfoList[i].key_points[0][0];
                    point_lines[1] = objectInfoList[i].key_points[0][1];
                    point_lines[2] = objectInfoList[i].key_points[1][0];
                    point_lines[3] = objectInfoList[i].key_points[1][1];
                    point_lines[4] = objectInfoList[i].key_points[1][0];
                    point_lines[5] = objectInfoList[i].key_points[1][1];
                    point_lines[6] = objectInfoList[i].key_points[2][0];
                    point_lines[7] = objectInfoList[i].key_points[2][1];
                    point_lines[8] = objectInfoList[i].key_points[2][0];
                    point_lines[9] = objectInfoList[i].key_points[2][1];
                    point_lines[10] = objectInfoList[i].key_points[3][0];
                    point_lines[11] = objectInfoList[i].key_points[3][1];
                    point_lines[12] = objectInfoList[i].key_points[3][0];
                    point_lines[13] = objectInfoList[i].key_points[3][1];
                    point_lines[14] = objectInfoList[i].key_points[0][0];
                    point_lines[15] = objectInfoList[i].key_points[0][1];

                    point_lines_list.add(point_lines);
                    labels.add(String.format("%s", objectInfoList[i].label));
                    txt_result += objectInfoList[i].label + "\n";
                }
                for (int i = 0; i < point_lines_list.size(); i++) {
                    float[] point_lines = point_lines_list.get(i);
                    canvas.drawLines(point_lines, mPaint);
                    if (labels.size() > 0) {
                        canvas.drawText(labels.get(i), point_lines[0], point_lines[1], mPaint);
                    }
                }
                mImageView.setImageBitmap(scaleBitmap2);
                mImageView.draw(canvas);

            }
            String benchResult = "text box count: " + objectCount + " " + Helper.getBenchResult() + txt_result;
            TextView result_view = (TextView) $(R.id.result);
            result_view.setText(benchResult);
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
