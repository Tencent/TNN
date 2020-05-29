package com.tencent.tnn.demo.common.component;

import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.Parameters;
import android.hardware.Camera.Size;
import android.media.CamcorderProfile;
import android.text.TextUtils;
import android.util.Log;
import android.view.Surface;
import android.view.WindowManager;
import java.util.Iterator;
import java.util.List;

public class CameraSetting {
    private static final String TAG = CameraSetting.class.getSimpleName();
    public static int mCameraFacing = 1;
    public static int mRotate = 0;
    static int mDesiredPreviewWidth = 640;
    static int mDesiredPreviewHeight = 480;

    public static void setCameraFacing(int cameraFacing) {
        mCameraFacing = cameraFacing;
    }

    public static void setCameraRotate(int cameraRotate) {
        mRotate = cameraRotate;
    }

    public static int getDesiredPreviewWidth() {
        return mDesiredPreviewWidth;
    }

    public static int getDesiredPreviewHeight() {
        return mDesiredPreviewHeight;
    }

    public static int getRotate(Context context, int cameraId, int cameraFacing) {
        int cameraRotateAngle = getVideoRotate(context, cameraId);
        int rotateTag = getRotateTag(cameraRotateAngle, cameraFacing);
        return rotateTag;
    }

    public static int transBackFacingCameraRatateTag(int backRotate) {
        if (backRotate == 1) {
            return 2;
        } else if (backRotate == 2) {
            return 1;
        } else if (backRotate == 3) {
            return 4;
        } else if (backRotate == 4) {
            return 3;
        } else if (backRotate == 5) {
            return 8;
        } else if (backRotate == 6) {
            return 7;
        } else if (backRotate == 7) {
            return 6;
        } else if (backRotate == 8) {
            return 5;
        } else {
            Log.w(TAG, "[YtCameraSetting.transBackFacingCameraRatateTag] unsurported rotateTag: " + backRotate);
            return 0;
        }
    }

    public static int getRotateTag(int cameraRotate, int cameraFacing) {
        int rotate = 1;
        if (cameraRotate == 90) {
            rotate = 7;
        } else if (cameraRotate == 180) {
            rotate = 3;
        } else if (cameraRotate == 270) {
            rotate = 5;
        } else {
            Log.i(TAG, "camera rotate not 90degree or 180degree, input: " + cameraRotate);
        }

        return cameraFacing == 1 ? rotate : transBackFacingCameraRatateTag(rotate);
    }

    public static int getVideoRotate(Context context, int mCameraID) {
        CameraInfo cameraInfo = new CameraInfo();
        Camera.getCameraInfo(mCameraID, cameraInfo);
        int rotation = ((WindowManager)context.getSystemService(Context.WINDOW_SERVICE)).getDefaultDisplay().getRotation();
        int degrees = 0;
        switch(rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
        }

        int videoOrientation;
        if (cameraInfo.facing == 1) {
            int hintOrientation = (cameraInfo.orientation + degrees) % 360;
            videoOrientation = (360 - hintOrientation) % 360;
        } else {
            videoOrientation = (cameraInfo.orientation - degrees + 360) % 360;
        }

        Log.i(TAG, "debug camera orientation is " + cameraInfo.orientation + " ui degrees is " + degrees);
        return videoOrientation;
    }

    public static int initCamera(Context context, Camera camera, int mCameraID) {
        Parameters parameters;
        try {
            parameters = camera.getParameters();
        } catch (Exception var20) {
            Log.e(TAG, "get camera parameters failed. 1. Check Camera.getParameters() interface. 2. Get logs for more detail.");
            return 1;
        }

        List<String> suporrtedFocusModes = parameters.getSupportedFocusModes();

        int videoOrientation;
        for(videoOrientation = 0; videoOrientation < suporrtedFocusModes.size(); ++videoOrientation) {
            Log.v(TAG, "suporrtedFocusModes " + videoOrientation + " :" + (String)suporrtedFocusModes.get(videoOrientation));
        }

        if (suporrtedFocusModes != null && suporrtedFocusModes.indexOf("continuous-video") >= 0) {
            parameters.setFocusMode("continuous-video");
            Log.d(TAG, "set camera focus mode continuous video");
        } else if (suporrtedFocusModes != null && suporrtedFocusModes.indexOf("auto") >= 0) {
            parameters.setFocusMode("auto");
            Log.d(TAG, "set camera focus mode auto");
        } else {
            Log.d(TAG, "NOT set camera focus mode");
        }

        try {
            camera.setParameters(parameters);
        } catch (Exception var18) {
            Log.e(TAG, "Camera.setParameters.setPreviewSize failed!!: " + var18.getLocalizedMessage());
        } finally {
            parameters = camera.getParameters();
        }

        videoOrientation = getVideoRotate(context, mCameraID);
        camera.setDisplayOrientation(videoOrientation);
        Log.d(TAG, "videoOrietation is" + videoOrientation);
        CamcorderProfile camcorderProfile;
        if (CamcorderProfile.hasProfile(mCameraID, 4)) {
            camcorderProfile = CamcorderProfile.get(mCameraID, 4);
            Log.d(TAG, "480P camcorderProfile:" + camcorderProfile.videoFrameWidth + "x" + camcorderProfile.videoFrameHeight);
        } else if (CamcorderProfile.hasProfile(mCameraID, 5)) {
            camcorderProfile = CamcorderProfile.get(mCameraID, 5);
            Log.d(TAG, "720P camcorderProfile:" + camcorderProfile.videoFrameWidth + "x" + camcorderProfile.videoFrameHeight);
        } else {
            camcorderProfile = CamcorderProfile.get(mCameraID, 1);
            Log.d(TAG, "High camcorderProfile:" + camcorderProfile.videoFrameWidth + "x" + camcorderProfile.videoFrameHeight);
        }

        setVideoSize(parameters, camcorderProfile);
        parameters.setPreviewSize(mDesiredPreviewWidth, mDesiredPreviewHeight);
        parameters.setPreviewFormat(17);

        try {
            camera.setParameters(parameters);
        } catch (Exception var17) {
            Log.e(TAG, "Camera.setParameters.setPreviewSize failed!!: " + var17.getLocalizedMessage());
        }

        parameters = camera.getParameters();
        int fps = chooseFixedPreviewFps(parameters, 30000);
        Log.d(TAG, "choose camera fps is : " + fps);

        try {
            camera.setParameters(parameters);
        } catch (Exception var16) {
            Log.e(TAG, "Camera.setParameters.preview fps failed!!: " + var16.getLocalizedMessage());
        }

        parameters = camera.getParameters();
        int[] newFpsRange = new int[2];
        parameters.getPreviewFpsRange(newFpsRange);
        int newFps = parameters.getPreviewFrameRate();
        Log.d(TAG, "after set parameters getPreviewFpsRange=" + newFpsRange[0] + "-" + newFpsRange[1] + " ;after set parameter fps=" + newFps);
        Size previewSize = parameters.getPreviewSize();
        Log.d(TAG, "camera preview size is " + previewSize.width + " " + previewSize.height);
        return 0;
    }

    public static void setVideoSize(Parameters parameters, CamcorderProfile camcorderProfile) {
        List<Size> sizes = parameters.getSupportedPreviewSizes();
        if (parameters.getSupportedVideoSizes() == null) {
            Size optimalSize = getOptimalPreviewSize(sizes, camcorderProfile.videoFrameWidth, camcorderProfile.videoFrameHeight);
            if (null == optimalSize) {
                Log.d(TAG, "do not find proper preview size, use default");
                camcorderProfile.videoFrameWidth = 640;
                camcorderProfile.videoFrameHeight = 480;
            }
        }

        boolean isVideoSizeOptimal = false;
        List<Size> videoSizes = parameters.getSupportedVideoSizes();
        if (videoSizes != null) {
            for(int i = 0; i < videoSizes.size(); ++i) {
                Size temp = (Size)videoSizes.get(i);
                if (temp.width == camcorderProfile.videoFrameWidth && temp.height == camcorderProfile.videoFrameHeight) {
                    isVideoSizeOptimal = true;
                }
            }

            if (!isVideoSizeOptimal) {
                camcorderProfile.videoFrameWidth = 640;
                camcorderProfile.videoFrameHeight = 480;
            }
        }

        Log.d(TAG, "select video size camcorderProfile:" + camcorderProfile.videoFrameWidth + "x" + camcorderProfile.videoFrameHeight);
    }

    private static Size getOptimalPreviewSize(List<Size> sizes, int width, int height) {
        double ASPECT_TOLERANCE = 0.001D;
        if (sizes == null) {
            return null;
        } else {
            Size optimalSize = null;
            double minDiff = 1.7976931348623157E308D;
            int targetWidth = Math.max(width, height);
            int targetHeight = Math.min(width, height);
            double targetRatio = (double)targetWidth / (double)targetHeight;
            Log.d(TAG, "sizes size=" + sizes.size());
            Iterator var12 = sizes.iterator();

            Size size;
            while(var12.hasNext()) {
                size = (Size)var12.next();
                double ratio = (double)size.width / (double)size.height;
                if (Math.abs(ratio - targetRatio) <= 0.001D && (double)Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = (double)Math.abs(size.height - targetHeight);
                }
            }

            if (optimalSize == null) {
                Log.d(TAG, "No preview size match the aspect ratio");
                minDiff = 1.7976931348623157E308D;
                var12 = sizes.iterator();

                while(var12.hasNext()) {
                    size = (Size)var12.next();
                    if ((double)Math.abs(size.height - targetHeight) < minDiff) {
                        optimalSize = size;
                        minDiff = (double)Math.abs(size.height - targetHeight);
                    }
                }
            }

            return optimalSize;
        }
    }

    private static int chooseFixedPreviewFps(Parameters parms, int desiredThousandFps) {
        List<int[]> supported = parms.getSupportedPreviewFpsRange();
        Iterator var3 = supported.iterator();

        while(var3.hasNext()) {
            int[] entry = (int[])var3.next();
            Log.d(TAG, "entry: " + entry[0] + " - " + entry[1]);
            if (entry[0] == entry[1] && entry[0] == desiredThousandFps) {
                parms.setPreviewFpsRange(entry[0], entry[1]);
                Log.d(TAG, "use preview fps range: " + entry[0] + " " + entry[1]);
                return entry[0];
            }
        }

        int[] tmp = new int[2];
        parms.getPreviewFpsRange(tmp);
        int guess;
        if (tmp[0] == tmp[1]) {
            guess = tmp[0];
        } else {
            guess = desiredThousandFps;
            if (desiredThousandFps > tmp[1]) {
                guess = tmp[1];
            }

            if (guess < tmp[0]) {
                guess = tmp[0];
            }
        }

        String preview_frame_rate_values = parms.get("preview-frame-rate-values");
        if (!TextUtils.isEmpty(preview_frame_rate_values) && !preview_frame_rate_values.contains("" + guess / 1000)) {
            String[] values = preview_frame_rate_values.split(",");
            String[] var7 = values;
            int var8 = values.length;

            for(int var9 = 0; var9 < var8; ++var9) {
                String string = var7[var9];
                int fps = Integer.parseInt(string) * 1000;
                if (guess < fps) {
                    parms.setPreviewFrameRate(fps / 1000);
                    return fps;
                }
            }

            if (values.length > 0) {
                int fps = Integer.parseInt(values[values.length - 1]) * 1000;
                if (guess > fps) {
                    guess = fps;
                }
            }
        }

        parms.setPreviewFrameRate(guess / 1000);
        return guess;
    }
}

