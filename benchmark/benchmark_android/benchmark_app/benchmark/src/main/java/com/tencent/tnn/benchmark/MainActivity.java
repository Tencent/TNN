package com.tencent.tnn.benchmark;


import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Debug;
import android.view.View;
import android.widget.TextView;

import android.util.Log;
import com.tencent.tnn.benchmark.BenchmarkModel;

import java.io.IOException;


public class MainActivity extends Activity {

    private TextView lightLiveCheckBtn;

    private static final String TAG = "TNN_BenchmarkModelActivity";
    private BenchmarkModel benchmark = new BenchmarkModel();
    private static final String ARGS_INTENT_KEY_0 = "args";
    private static final String ARGS_INTENT_KEY_1 = "--args";

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        init();

    }

    private void init() {

        System.loadLibrary("tnn_wrapper");
        Intent intent = getIntent();
        Bundle bundle = intent.getExtras();
        String args = bundle.getString(ARGS_INTENT_KEY_0, bundle.getString(ARGS_INTENT_KEY_1));
        String fileDir = initModel();

        Log.i(TAG, "Running TNN Benchmark with args: " + args);
        benchmark.nativeRun(args, fileDir);
    }

    private String initModel() {

        String targetDir = this.getFilesDir().getAbsolutePath();
        String[] files;
        try {
            files = this.getResources().getAssets().list("");
        } catch (IOException exception){
            Log.e(TAG, "Get TNN Benchmark assets failed");
            return targetDir;
        }

        for (int i = 0; i < files.length; i++) {
            if (files[i].contains(".tnnproto") || files[i].contains(".tnnmodel")) {
                String interModelFilePath = targetDir + "/" + files[i];
                boolean ret = FileUtils.copyAsset(this.getResources().getAssets(), files[i], interModelFilePath);
            }
        }
        return targetDir;
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

}
