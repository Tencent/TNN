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
    private Paint line_paint = new Paint();
    private Paint line_point_paint = new Paint();
    private Paint text_paint = new Paint();
    private ArrayList<String> labels = new ArrayList<String>();
    private ArrayList<Rect> rects = new ArrayList<Rect>();
    private ArrayList<float[]> points_list = new ArrayList<float[]>();
    private ArrayList<float[]> point_lines_list = new ArrayList<float[]>();
    private ArrayList<ImageInfo> image_info_list = new ArrayList<ImageInfo>();

    public DrawView(Context context, AttributeSet attrs)
    {
        super(context, attrs);
        paint.setARGB(255, 0, 255, 0);
        key_paint.setARGB(255, 0, 255, 0);
        paint.setStyle(Paint.Style.STROKE);
        key_paint.setStyle(Paint.Style.STROKE);
        key_paint.setStrokeWidth(5);
        line_paint.setARGB(255, 255, 0, 0);
        line_paint.setStyle(Paint.Style.STROKE);
        line_paint.setStrokeWidth(3);
        line_point_paint.setARGB(255, 0, 255, 0);
        line_point_paint.setStyle(Paint.Style.STROKE);
        line_point_paint.setStrokeWidth(10);
        text_paint.setARGB(255, 255, 0, 0);
        text_paint.setStyle(Paint.Style.STROKE);
        text_paint.setTextAlign(Paint.Align.CENTER);
        text_paint.setTextSize(30);
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

    public void addTextRect(ObjectInfo[] objectstatus)
    {
        point_lines_list.clear();
        labels.clear();
        if (objectstatus != null && objectstatus.length!=0)
        {
            for (int i=0; i<objectstatus.length; i++)
            {
                float[] point_lines = new float[4 * 4];
                point_lines[0] = objectstatus[i].key_points[0][0];
                point_lines[1] = objectstatus[i].key_points[0][1];
                point_lines[2] = objectstatus[i].key_points[1][0];
                point_lines[3] = objectstatus[i].key_points[1][1];
                point_lines[4] = objectstatus[i].key_points[1][0];
                point_lines[5] = objectstatus[i].key_points[1][1];
                point_lines[6] = objectstatus[i].key_points[2][0];
                point_lines[7] = objectstatus[i].key_points[2][1];
                point_lines[8] = objectstatus[i].key_points[2][0];
                point_lines[9] = objectstatus[i].key_points[2][1];
                point_lines[10] = objectstatus[i].key_points[3][0];
                point_lines[11] = objectstatus[i].key_points[3][1];
                point_lines[12] = objectstatus[i].key_points[3][0];
                point_lines[13] = objectstatus[i].key_points[3][1];
                point_lines[14] = objectstatus[i].key_points[0][0];
                point_lines[15] = objectstatus[i].key_points[0][1];

                point_lines_list.add(point_lines);
                labels.add(String.format("%s", objectstatus[i].label));
            }
        }

        postInvalidate();
    }

    public void addObjectRect(ObjectInfo[] objectstatus)
    {
        points_list.clear();
        point_lines_list.clear();
        if (objectstatus != null && objectstatus.length != 0)
        {
            for (int i = 0; i < objectstatus.length; i++) {
                float[][] key_points = objectstatus[i].key_points;
                if (key_points != null && key_points.length != 0) {
                    float[] points = new float[key_points.length * 2];
                    for (int j = 0; j < key_points.length; ++j) {
                        points[j * 2] = key_points[j][0];
                        points[j * 2 + 1] = key_points[j][1];
                    }
                    points_list.add(points);
                }

                int[][] lines = objectstatus[i].lines;
                if (lines != null && lines.length != 0) {
                    float[] point_lines = new float[lines.length * 4];
                    for (int j = 0; j < lines.length; ++j) {
                        point_lines[j * 4] = key_points[objectstatus[i].lines[j][0]][0];
                        point_lines[j * 4 + 1] = key_points[objectstatus[i].lines[j][0]][1];
                        point_lines[j * 4 + 2] = key_points[objectstatus[i].lines[j][1]][0];
                        point_lines[j * 4 + 3] = key_points[objectstatus[i].lines[j][1]][1];
                    }
                    point_lines_list.add(point_lines);
                }
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

        if (points_list.size() > 0) {
            for (int i = 0; i < points_list.size(); ++i) {
                float[] points = points_list.get(i);
                canvas.drawPoints(points, point_lines_list.isEmpty() ? key_paint : line_point_paint);
            }
        }

        if (point_lines_list.size() > 0) {
            for (int i = 0; i < point_lines_list.size(); ++i) {
                float[] point_lines = point_lines_list.get(i);
                canvas.drawLines(point_lines, line_paint);
                if(labels.size() > 0) {
                    canvas.drawText(labels.get(i), point_lines[0], point_lines[1], text_paint);
                }
            }
        }
    }
}
