# ARM LINUX demo

## Compile: 
 * 1, cd TNN/examples/armlinux
 * 2, set CC & CXX & TNN_LIB_PATH in build_aarch64_linux.sh or build_armhf_linux.sh
 * 3, run build_aarch64.sh or build_armhf.sh and you will get binary build/demo_arm_linux

## Function：
    Create a tnn instance，and run inference. 
    input is random bgr data, output is nchw float blob
    your need to git the demo your proto and model
