#!/bin/bash

OPENCV_INSTALL_DIR=`dirname $0`/../third_party/opencv

download_opencv() { #platfotm:iOS/Android, URL
    PLATFORM=$1
    SAVEDIR=${OPENCV_INSTALL_DIR}/${PLATFORM}
    if [ ! -d ${SAVEDIR} ]; then
        mkdir -p ${SAVEDIR}
    fi
    FILENAME=`basename $2`
    SAVEPATH=${SAVEDIR}/${FILENAME}
    if [ ! -e ${SAVEPATH} ]; then
        echo "downloading ${FILENAME}"
        status=`curl -L $2 -w %{http_code} -o ${SAVEPATH}`
        if [ $status -ne 200 ]; then
            echo "download ${FILENAME} failed!"
            if [ -d ${SAVEDIR} ]; then
                rm -r ${SAVEDIR}
            fi
            return -1
        fi
    fi
    echo "unzip ${FILENAME} into ${SAVEDIR}"
    #unzip
    unzip -q ${SAVEPATH} -d ${SAVEDIR}
}
os="all"
if [ "$#" -ge 1 ]; then
    os=`echo $1 | tr 'A-Z' 'a-z'`
fi

if [ "ios" == $os ]; then
    download_opencv iOS https://sourceforge.net/projects/opencvlibrary/files/3.4.13/opencv-3.4.13-ios-framework.zip
elif [ "android" == $os ]; then
    download_opencv Android https://sourceforge.net/projects/opencvlibrary/files/3.4.13/opencv-3.4.13-android-sdk.zip
else
    download_opencv iOS https://sourceforge.net/projects/opencvlibrary/files/3.4.13/opencv-3.4.13-ios-framework.zip
    download_opencv Android https://sourceforge.net/projects/opencvlibrary/files/3.4.13/opencv-3.4.13-android-sdk.zip
fi
