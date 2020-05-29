import os
import argparse
import sys
import time
# sys.path.append('./onnx-optimizer')
# from onnx_optimizer import onnx_optimizer

import onnx2tnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Input file path')
    parser.add_argument('-version', help='version to set')
    args = parser.parse_args()
    file_path = args.file_path
    file_version = args.version
    # print(file_path)
    # print(file_version)

    if file_version == None:
        print('0.----get file version:')
        status = onnx2tnn.version(file_path)
        print('1.----get file time:')
        status = onnx2tnn.time(file_path)
        print("2.----file version status: "+str(status))
    else:
        print('0.----set file version:')
        status = onnx2tnn.set_version(file_path, file_version)
        print('1.----set file time:')
        file_time = time.strftime("%Y%m%d %H:%M:%S", time.localtime())
        status = onnx2tnn.set_time(file_path, file_time)
        print("2.----file version status: "+str(status))

if __name__ == '__main__':
    main()
