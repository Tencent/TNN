package com.tencent.tnn.demo.common.component;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceView;
import android.graphics.Bitmap;

import com.tencent.tnn.demo.BlazeFaceDetector;
import com.tencent.tnn.demo.FaceDetector;
import com.tencent.tnn.demo.FaceInfo;
import com.tencent.tnn.demo.ImageInfo;
import com.tencent.tnn.demo.ObjectDetector;
import com.tencent.tnn.demo.ObjectDetectorSSD;
import com.tencent.tnn.demo.ObjectInfo;


import java.util.ArrayList;
import java.nio.ByteBuffer;


public class DrawView extends SurfaceView
{
    private static String TAG = DrawView.class.getSimpleName();
    private Paint paint = new Paint();
    private Paint key_paint = new Paint();
    private ArrayList<String> labels = new ArrayList<String>();
    private ArrayList<Rect> rects = new ArrayList<Rect>();
    private ArrayList<float[]> points_list = new ArrayList<float[]>();
    private ArrayList<ImageInfo> image_info_list = new ArrayList<ImageInfo>();

    public DrawView(Context context, AttributeSet attrs)
    {
        super(context, attrs);
        paint.setARGB(255, 0, 255, 0);
        key_paint.setARGB(255, 0, 255, 0);
        paint.setStyle(Paint.Style.STROKE);
        key_paint.setStyle(Paint.Style.STROKE);
        key_paint.setStrokeWidth(5);
        setWillNotDraw(false);
    }

    public void addFaceRect(FaceInfo[] facestatus)
    {
        rects.clear();
        points_list.clear();
        if (facestatus != null && facestatus.length!=0)
        {
            for (int i=0; i<facestatus.length; i++)
            {
                rects.add(new Rect((int)facestatus[i].x1, (int)facestatus[i].y1, (int)facestatus[i].x2, (int)facestatus[i].y2));
                float[][] keypoints = facestatus[i].keypoints;
                if(keypoints != null) {
                    float[] points = new float[facestatus[i].keypoints.length * 2];
                    for(int j = 0; j < keypoints.length; ++j) {
                        points[j * 2] = facestatus[i].keypoints[j][0];
                        points[j * 2 + 1] = facestatus[i].keypoints[j][1];
                    }
                    points_list.add(points);
                }
            }
        }

        postInvalidate();
    }

    public void addObjectRect(ObjectInfo[] objectstatus, String[]  label_list)
    {
        rects.clear();
        labels.clear();
        if (objectstatus != null && objectstatus.length!=0)
        {
            for (int i=0; i<objectstatus.length; i++)
            {
                rects.add(new Rect((int)objectstatus[i].x1, (int)objectstatus[i].y1, (int)objectstatus[i].x2, (int)objectstatus[i].y2));
                labels.add(String.format("%s : %f", label_list[objectstatus[i].class_id], objectstatus[i].score));
            }
        }

        postInvalidate();
    }

    public void addImageInfo(ImageInfo imageInfo)
    {
        image_info_list.clear();
        image_info_list.add(imageInfo);

        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas)
    {
        if (rects.size() > 0)
        {
            for (int i=0; i<rects.size(); i++) {
                Log.d(TAG, "rect " + rects.get(i));
                paint.setARGB(255, 0, 255, 0);
                canvas.drawRect(rects.get(i), paint);
                if(labels.size() > 0) {
                    canvas.drawText(labels.get(i), rects.get(i).left, rects.get(i).top - 5, paint);
                }
            }

        }

        if(points_list.size() > 0) {
            for(int i = 0; i < points_list.size(); ++i) {
                float[] points = points_list.get(i);
                canvas.drawPoints(points, key_paint);
            }
        }

        if (image_info_list.size() > 0) {
            for (int i = 0; i < image_info_list.size(); i++) {
                ImageInfo imageInfo = image_info_list.get(i);
                if (imageInfo.image_channel != 4) {
                    Log.e(TAG, "canvas get invalid image info, image_channel: " + imageInfo.image_channel);
                } else {
                    Bitmap bitmap = Bitmap.createBitmap(imageInfo.image_width, imageInfo.image_height, Bitmap.Config.ARGB_8888);
                    ByteBuffer buffer = ByteBuffer.wrap(imageInfo.data);
                    bitmap.copyPixelsFromBuffer(buffer);
                    Rect rect = new Rect(0, 0, getWidth() - 1, getHeight() -1);
                    canvas.drawBitmap(bitmap, null, rect, null);
                    bitmap.recycle();
                }
            }
        }
    }
}
