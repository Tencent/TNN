package com.tencent.tnn.demo;


import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Debug;
import android.view.View;
import android.widget.TextView;

import com.tencent.tnn.demo.ImageBlazeFaceDetector.ImageBlazeFaceDetectActivity;
import com.tencent.tnn.demo.ImageClassifyDetector.ImageClassifyDetectActivity;
import com.tencent.tnn.demo.ImageFaceDetector.ImageFaceDetectActivity;
import com.tencent.tnn.demo.ImageObjectDetectorSSD.ImageObjectDetectSSDActivity;
import com.tencent.tnn.demo.StreamBlazeFaceAlign.StreamBlazeFaceAlignActivity;
import com.tencent.tnn.demo.StreamBlazeFaceDetector.StreamBlazeFaceDetectActivity;
import com.tencent.tnn.demo.StreamFaceDetector.StreamFaceDetectActivity;
import com.tencent.tnn.demo.ImageObjectDetector.ImageObjectDetectActivity;
import com.tencent.tnn.demo.StreamHairSegmentation.StreamHairSegmentationActivity;
import com.tencent.tnn.demo.StreamObjectDetector.StreamObjectDetectActivity;
import com.tencent.tnn.demo.StreamObjectDetectorSSD.StreamObjectDetectSSDActivity;


public class MainActivity extends Activity {

    private TextView lightLiveCheckBtn;

    private boolean isShowedActivity = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//       Debug.waitForDebugger();

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

        findViewById(R.id.image_object_detect_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, ImageObjectDetectActivity.class);
                    activity.startActivity(intent);
                }
            }
        });

        findViewById(R.id.stream_object_detect_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, StreamObjectDetectActivity.class);
                    activity.startActivity(intent);
                }
            }
        });
        findViewById(R.id.image_object_detectssd_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, ImageObjectDetectSSDActivity.class);
                    activity.startActivity(intent);
                }
            }
        });

        findViewById(R.id.stream_object_detectssd_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, StreamObjectDetectSSDActivity.class);
                    activity.startActivity(intent);
                }
            }
        });
        findViewById(R.id.image_facedetect_blaze_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, ImageBlazeFaceDetectActivity.class);
                    activity.startActivity(intent);
                }
            }
        });

        findViewById(R.id.stream_facedetect_blaze_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, StreamBlazeFaceDetectActivity.class);
                    activity.startActivity(intent);
                }
            }
        });

        findViewById(R.id.stream_facealign_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, StreamBlazeFaceAlignActivity.class);
                    activity.startActivity(intent);
                }
            }
        });

        findViewById(R.id.stream_hairsegmentation_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!isShowedActivity) {
                    isShowedActivity = true;
                    Intent intent = new Intent();
                    Activity activity = MainActivity.this;
                    intent.setClass(activity, StreamHairSegmentationActivity.class);
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
