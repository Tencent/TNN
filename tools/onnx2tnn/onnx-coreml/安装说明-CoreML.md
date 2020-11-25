//参考 https://stringpiggy.hpd.io/mac-osx-python3-dual-install/
//遇到资源无法下载，可切换到GestWiFi，或尝试用代理
<!--
export http_proxy=http://web-proxy.oa.com:8080/
export all_proxy=https://web-proxy.oa.com:8080/
export ftp_proxy=ftp://web-proxy.oa.com:8080/
export socks_proxy=socks://web-proxy.oa.com:8080/
export ALL_PROXY=https://web-proxy.oa.com:8080/
export https_proxy=https://web-proxy.oa.com:8080/
-->

步骤：
<!-- 版本不一致会导致protobuf库找不到，最好与转化库so所用的版本保持一致，否则都用最新版本，so重新编译
参考：https://blog.csdn.net/qq_21383435/article/details/81035852
-->
<!-- 编译+运行 -->
1. 安装coremltools //https://pypi.org/project/coremltools/
pip3 install -U coremltools
pip3 install -U onnx-coreml
