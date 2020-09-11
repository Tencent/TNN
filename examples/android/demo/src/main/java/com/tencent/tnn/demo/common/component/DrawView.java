package com.tencent.tnn.demo.common.component;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceView;

import com.tencent.tnn.demo.FaceDetector;
import com.tencent.tnn.demo.ObjectDetector;
import com.tencent.tnn.demo.ObjectDetectorSSD;
import com.tencent.tnn.demo.ObjectInfo;


import java.util.ArrayList;


public class DrawView extends SurfaceView
{
    private static String TAG = DrawView.class.getSimpleName();
    private Paint paint = new Paint();
    private ArrayList<String> labels = new ArrayList<String>();
    private ArrayList<Rect> rects = new ArrayList<Rect>();

    public DrawView(Context context, AttributeSet attrs)
    {
        super(context, attrs);
        paint.setARGB(255, 0, 255, 0);
        paint.setStyle(Paint.Style.STROKE);
        setWillNotDraw(false);
    }

    public void addFaceRect(FaceDetector.FaceInfo[] facestatus, int w, int h)
    {
        rects.clear();
        Log.d(TAG, "canvas " + getWidth() + "x" + getHeight() + " wh " + w + "x" +h);
        float scalew = getWidth() / (float)w;
        float scaleh = getHeight() / (float)h;
        if (facestatus != null && facestatus.length!=0)
        {
            for (int i=0; i<facestatus.length; i++)
            {
                rects.add(new Rect((int)(facestatus[i].x1 * scalew), (int)(facestatus[i].y1 * scaleh), (int)(facestatus[i].x2 * scalew), (int)(facestatus[i].y2 * scaleh)));
            }
        }

        postInvalidate();
    }

    public void addObjectRect(ObjectInfo[] objectstatus, String[]  label_list, int w, int h)
    {
        rects.clear();
        labels.clear();
        Log.d(TAG, "canvas " + getWidth() + "x" + getHeight() + " wh " + w + "x" +h);
        float scalew = getWidth() / (float)w;
        float scaleh = getHeight() / (float)h;
        if (objectstatus != null && objectstatus.length!=0)
        {
            for (int i=0; i<objectstatus.length; i++)
            {
                rects.add(new Rect((int)(objectstatus[i].x1 * scalew), (int)(objectstatus[i].y1 * scaleh), (int)(objectstatus[i].x2 * scalew), (int)(objectstatus[i].y2 * scaleh)));
                labels.add(String.format("%s : %f", label_list[objectstatus[i].class_id], objectstatus[i].score));
            }
        }

        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas)
    {
        Log.d(TAG, "draw face count " + rects.size());
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
    }
}
