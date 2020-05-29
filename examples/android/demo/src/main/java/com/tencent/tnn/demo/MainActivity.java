package com.tencent.tnn.demo;


import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import com.tencent.tnn.demo.ImageClassifyDetector.ImageClassifyDetectActivity;
import com.tencent.tnn.demo.ImageFaceDetector.ImageFaceDetectActivity;
import com.tencent.tnn.demo.StreamFaceDetector.StreamFaceDetectActivity;


public class MainActivity extends Activity {

    private TextView lightLiveCheckBtn;

    private boolean isShowedActivity = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        init();

    }

    private void init() {
        findViewById(R.id.stream_detect_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, StreamFaceDetectActivity.class);
                    activity.startActivity(intent);
                }
            }
        });
        findViewById(R.id.image_detect_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, ImageFaceDetectActivity.class);
                    activity.startActivity(intent);
                }
            }
        });

        findViewById(R.id.image_classify_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, ImageClassifyDetectActivity.class);
                    activity.startActivity(intent);
                }
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        isShowedActivity = false;
    }

}
