#!/bin/bash

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
	"https://raw.githubusercontent.com/darrenyao87/tnn-models/master/ddk/hwhiai-ddk-100.320.030.010.tar"\
  "hwhiai-ddk-100.320.030.010.tar"
tar -xvf hwhiai-ddk-100.320.030.010.tar 
mv hwhiai-ddk-100.320.030.010 ./hiai_ddk_latest/
