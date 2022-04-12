package com.tencent.tnn.demo.ReadingComprehension;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.tencent.tnn.demo.FileUtils;
import com.tencent.tnn.demo.Helper;
import com.tencent.tnn.demo.R;
import com.tencent.tnn.demo.common.component.DrawView;
import com.tencent.tnn.demo.common.fragment.BaseFragment;
import com.tencent.tnn.demo.ReadingComprehensionTinyBert;

public class ReadingComprehensionFragment extends BaseFragment {

    private final static String TAG = ReadingComprehensionFragment.class.getSimpleName();
    private ReadingComprehensionTinyBert mReadingComprehensionTinyBert = new ReadingComprehensionTinyBert();
    private boolean isOk = false;

    private ToggleButton mGPUSwitch;

    private Button demo1_btn;
    private Button demo2_btn;
    private Button demo3_btn;

    private Button ask_btn;

    private EditText material;
    private EditText question;
    private EditText answer;

    private boolean mUseGPU = false;

    String material1 = "TNN: A high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab. It also has many outstanding advantages such as cross-platform, high performance, model compression, and code tailoring. The TNN framework further strengthens the support and performance optimization of mobile devices on the basis of the original Rapidnet and ncnn frameworks. At the same time, it refers to the high performance and good scalability characteristics of the industry's mainstream open source frameworks, and expands the support for X86 and NV GPUs. On the mobile phone, TNN has been used by many applications such as mobile QQ, weishi, and Pitu. As a basic acceleration framework for Tencent Cloud AI, TNN has provided acceleration support for the implementation of many businesses. Everyone is welcome to participate in the collaborative construction to promote the further improvement of the TNN inference framework.";
    String question1 = "what advantages does TNN have?";

    String material2 = "Pumas are large, cat-like animals which are found in America. When reports came into London Zoo that a wild puma had been spotted forty-five miles south of London, they were not taken seriously. However, as the evidence began to accumulate, experts from the Zoo felt obliged to investigate, for the descriptions given by people who claimed to have seen the puma were extraordinarily similar.";
    String question2 = "where has puma been spotted?";

    String material3 = "This paper introduces TIRAMISU, a polyhedral compiler with a scheduling language. TIRAMISU is designed not only for the area of deep learning but also for the areas of image processing and tensor algebra. TIRAMISU relies on the flexible polyhedral representation which allows many advanced capabilities such as expressing complex code transformations, expressing non-rectangular iteration spaces and performing dependence analysis to check the correctness of transformations all of which are difficult to express in non-polyhedral compilers.";
    String question3 = "what are tiramisu designed for?";

    private String initModel() {
        String targetDir =  getActivity().getFilesDir().getAbsolutePath();

        //copy detect model to sdcard
        String[] modelPathsDetector = {
                "tiny-bert-squad.tnnmodel",
                "tiny-bert-squad.tnnproto",
                "vocab.txt"
        };

        for (int i = 0; i < modelPathsDetector.length; i++) {
            String modelFilePath = modelPathsDetector[i];
            String interModelFilePath = targetDir + "/" + modelFilePath ;
            FileUtils.copyAsset(getActivity().getAssets(), "tiny-bert/"+modelFilePath, interModelFilePath);
        }
        return targetDir;
    }

    private void checkInitStatus(int result) {
        if(result != 0){
            isOk = false;
            answer.setText("failed to init model " + result);
            Log.e(TAG, "failed to init model " + result);
        } else {
            isOk = true;
        }
    }

    private void onSwitchGPU(boolean b) {
        mUseGPU = b;
        String modelPath = getActivity().getFilesDir().getAbsolutePath();
        int result;

        if (mUseGPU) {
            result = mReadingComprehensionTinyBert.init(modelPath, 1); //use gpu
        }else {
            result = mReadingComprehensionTinyBert.init(modelPath, 0); // use cpu
        }
        checkInitStatus(result);
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        System.loadLibrary("tnn_wrapper");
        String modelPath = initModel();

        int result = mReadingComprehensionTinyBert.init(modelPath, 0); // default use cpu
        checkInitStatus(result);
    }

    @Override
    public void setFragmentView() {
        Log.d(TAG, "setFragmentView");
        setView(R.layout.fragment_reading_comprehension);
        setTitleGone();
        $$(R.id.gpu_switch);
        mGPUSwitch = $(R.id.gpu_switch);
        mGPUSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                onSwitchGPU(b);
            }
        });

        demo1_btn = $(R.id.reading_comprehension_demo_1_btn);
        demo2_btn = $(R.id.reading_comprehension_demo_2_btn);
        demo3_btn = $(R.id.reading_comprehension_demo_3_btn);
        ask_btn   = $(R.id.reading_comprehension_ask_btn);

        demo1_btn.setOnClickListener(this);
        demo2_btn.setOnClickListener(this);
        demo3_btn.setOnClickListener(this);
        ask_btn.setOnClickListener(this);

        material = $(R.id.reading_comprehension_material_text);
        question = $(R.id.reading_comprehension_question_text);
        answer = $(R.id.reading_comprehension_answer_text);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.reading_comprehension_demo_1_btn:
                onClickButton(1);
                material.setText(material1);
                question.setText(question1);
                break;
            case R.id.reading_comprehension_demo_2_btn:
                onClickButton(2);
                material.setText(material2);
                question.setText(question2);
                break;
            case R.id.reading_comprehension_demo_3_btn:
                onClickButton(3);
                material.setText(material3);
                question.setText(question3);
                break;
            case R.id.reading_comprehension_ask_btn:
                ask();
                break;
            default:
                break;
        }
    }

    private void onClickButton(int btn_no) {

    }

    private void ask(){
        if(isOk){
            String modelPath = getActivity().getFilesDir().getAbsolutePath();
            String answer_text = mReadingComprehensionTinyBert.ask(modelPath, material.getText().toString(), question.getText().toString());
            answer.setText(answer_text);
        } else {
            Log.e(TAG, "tnn has not been initialized");
            answer.setText("failed to init model maybe you are using GPU");
        }
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

    private void clickBack() {
        if (getActivity() != null) {
            (getActivity()).finish();
        }
    }
}
