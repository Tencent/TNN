#!/bin/bash

HIAI_DDK_VERSION=hwhiai-ddk-100.500.010.011
#URL, local path
download_file() { #URL, path
  if [ -e $2 ]; then return 0; fi

  name=`basename $2`
  echo "downloading $name ..."
  status=`curl $1 -s -w %{http_code} -o $2`
  if (( status == 200 )); then
    return 0
  else
    echo "download $name failed" 1>&2
    return -1
  fi
}

download_ddk() {
  directory="./"
  if [ ! -e ${directory} ]; then
    mkdir -p ${directory}
  fi

  ddk_name=`basename $1`
  ddk_path_local="${directory}/${ddk_name}"
  if [ ! -f ${ddk_path_local} ]; then
    download_file $1 $ddk_path_local
  fi

}
download_ddk\
    "https://raw.githubusercontent.com/darrenyao87/tnn-models/master/ddk/$HIAI_DDK_VERSION.tar"\
  "$HIAI_DDK_VERSION.tar"
tar -xvf $HIAI_DDK_VERSION.tar
rm hiai_ddk_latest
ln -s $HIAI_DDK_VERSION hiai_ddk_latest
