#[=======================================================================[.rst:
#FindDDK
#---------
#You Must Export DDK_HOME At First!!!
#it may be /home/youruser/tools/che/ddk/ddk

#DDK headers and libraries

#    DDK_INCLUDE_DIRS        - 
#    DDK_HOST_LIBRARIES      -
#    DDK_DEVICE_LIBRARIES    -
#    DDK_FOUND               -

#DDK Third_party

#    cereal
#        DDK_CEREAL_INCLUDE_DIRS     - 

#    protobuf
#        DDK_PROTOBUF_INCLUDE_DIRS   -  
#        DDK_PROTOBUF_LIBRARYS       -

#    glog
#        DDK_GLOG_INCLUDE_DIRS       - 
#        DDK_GLOG_LIBRARYS           -

#    gflags
#        DDK_GFLAGS_INCLUDE_DIRS     -
#        DDK_GFLAGS_LIBRARYS         -

#    opencv
#        DDK_OPENCV_INCLUDE_DIRS     -
#        DDK_OPENCV_LIBRARYS         -
    
#]=======================================================================]

if(NOT DEFINED ENV{DDK_HOME})
    message(FATAL_ERROR "please define environment variable:DDK_HOME")  
endif()

set(_DDK_ROOT_PATHS PATHS $ENV{DDK_HOME} NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_FIND_ROOT_PATH)

#
set(_DDK_HOST_LIB_PATHS ${_DDK_ROOT_PATHS})
if(${CMAKE_BUILD_TARGET} MATCHES "A500")
    set(_DDK_HOST_PATH_SUFFIXES "lib64")
else()
    set(_DDK_HOST_PATH_SUFFIXES "host/lib")
endif()
set(_DDK_DEVICE_PATH_SUFFIXES "device/lib")


#
function(find_ddk_host_lib _var _names)
    find_library(${_var} NAMES ${_names} NAMES_PER_DIR ${_DDK_HOST_LIB_PATHS} PATH_SUFFIXES ${_DDK_HOST_PATH_SUFFIXES} NO_CMAKE_SYSTEM_PATH)
endfunction()

function(find_ddk_device_lib _var _names)
    find_library(${_var} NAMES ${_names} NAMES_PER_DIR ${_DDK_ROOT_PATHS} PATH_SUFFIXES ${_DDK_DEVICE_PATH_SUFFIXES} NO_CMAKE_SYSTEM_PATH)
endfunction()

function(add_engine_library engine engine_dir side)
    aux_source_directory(${engine_dir} ${engine}_src)
    add_library(${engine} SHARED ${${engine}_src})
    target_include_directories(${engine} PUBLIC ${engine_dir})
    target_link_libraries(${engine} ${DDK_${side}_LIBRARIES})
endfunction()

if(NOT DDK_FOUND)

    #ddk include path
    set(DDK_INCLUDE_DIRS ${DDK_INCLUDE_DIRS} $ENV{DDK_HOME}/include)
    set(DDK_INCLUDE_DIRS ${DDK_INCLUDE_DIRS} $ENV{DDK_HOME}/include/inc)
    set(DDK_INCLUDE_DIRS ${DDK_INCLUDE_DIRS} $ENV{DDK_HOME}/include/libc_sec/include)

    #ddk lib
    #c_sec
    find_ddk_host_lib(DDK_CSEC_LIBRARY c_sec)

    #common
    find_ddk_host_lib(DDK_DRVDEVDRV_LIBRARY drvdevdrv)
    find_ddk_host_lib(DDK_DRVHDC_HOST_LIBRARY drvhdc_host)
    find_ddk_host_lib(DDK_MMPA_LIBRARY mmpa)
    find_ddk_host_lib(DDK_MEMORY_LIBRARY memory)
    find_ddk_host_lib(DDK_MATRIX_LIBRARY matrix)
    find_ddk_host_lib(DDK_PROFILERCLIENT_LIBRARY profilerclient)
    find_ddk_host_lib(DDK_SLOG_LIBRARY slog)

    set(DDK_HOST_LIBRARIES 
        ${DDK_DRVDEVDRV_LIBRARY} 
        ${DDK_DRVHDC_HOST_LIBRARY}
        ${DDK_MMPA_LIBRARY} 
        ${DDK_MEMORY_LIBRARY}
        ${DDK_MATRIX_LIBRARY} 
        ${DDK_PROFILERCLIENT_LIBRARY} 
        ${DDK_SLOG_LIBRARY} 
        ${DDK_CSEC_LIBRARY})

    #device
    find_ddk_device_lib(DDK_IDEDAEMON_LIBRARY idedaemon)

    #dvpp lib
    find_ddk_device_lib(DDK_DVPP_API_LIBRARY Dvpp_api)
    find_ddk_device_lib(DDK_DVPP_JPEG_D_LIBRARY Dvpp_jpeg_decoder)
    find_ddk_device_lib(DDK_DVPP_JEPG_E_LIBRARY Dvpp_jpeg_encoder)
    find_ddk_device_lib(DDK_DVPP_PNG_D_LIBRARY Dvpp_png_decoder)
    find_ddk_device_lib(DDK_DVPP_VPC_LIBRARY Dvpp_vpc)
    set(DDK_DVPP_LIBRARYS 
        ${DDK_DVPP_API_LIBRARY} 
        ${DDK_DVPP_JPEG_D_LIBRARY} 
        ${DDK_DVPP_JEPG_E_LIBRARY} 
        ${DDK_DVPP_PNG_D_LIBRARY} 
        ${DDK_DVPP_VPC_LIBRARY})

    set(DDK_DEVICE_LIBRARIES ${DDK_IDEDAEMON_LIBRARY} ${DDK_DVPP_LIBRARYS})

    #third_party
    set(_DDK_THIRD_INC_DIR $ENV{DDK_HOME}/include/third_party)

    #cereal - only header file(hpp)
    set(DDK_CEREAL_INCLUDE_DIRS ${_DDK_THIRD_INC_DIR}/cereal/include)

    #glog
    set(DDK_GLOG_INCLUDE_DIR ${_DDK_THIRD_INC_DIR}/glog/include)
    find_ddk_host_lib(DDK_GLOG_LIBRARYS glog)

    #gflags
    set(DDK_GFLAGS_INCLUDE_DIRS ${_DDK_THIRD_INC_DIR}/gflags/include)
    find_ddk_host_lib(DDK_GFLAGS_LIBRARYS gflags)

    #openssl
    find_ddk_host_lib(DDK_OPENSSL_SSL_LIBRARY ssl)
    find_ddk_host_lib(DDK_OPENSSL_CRYPTO_LIBRARY crypto)
    set(DDK_OPENSSL_LIBRARYS ${DDK_OPENSSL_SSL_LIBRARY} ${DDK_OPENSSL_CRYPTO_LIBRARY})

    #protobuf
    set(DDK_PROTOBUF_INCLUDE_DIRS ${_DDK_THIRD_INC_DIR}/protobuf/include)
    find_ddk_host_lib(DDK_PROTOBUF_LIBRARYS protobuf)

    #opencv
    set(DDK_OPENCV_INCLUDE_DIRS ${_DDK_THIRD_INC_DIR}/opencv/include)
    find_ddk_host_lib(DDK_HOST_OPENCV_LIBRARYS opencv_world)
    # set(DDK_HOST_OPENCV_LIBRARYS ${DDK_OPENCV_LIBRARYS})
    find_ddk_device_lib(DDK_DEVICE_OPENCV_LIBRARYS opencv_world)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
        set(OpenCV_LIBS ${DDK_DEVICE_OPENCV_LIBRARYS})
        set(DDK_OPENCV_LIBRARYS ${DDK_DEVICE_OPENCV_LIBRARYS})
        set(DDK_HOST_LIBRARIES ${DDK_HOST_LIBRARIES} ${DDK_PROTOBUF_LIBRARYS} ${DDK_OPENSSL_LIBRARYS})  
    else()
        set(OpenCV_LIBS ${DDK_HOST_OPENCV_LIBRARYS})
        set(DDK_OPENCV_LIBRARYS ${DDK_HOST_OPENCV_LIBRARYS})
    endif()
    #
    set(DDK_THIRD_LIBRARYS 
        ${DDK_GLOG_LIBRARYS} 
        ${DDK_GFLAGS_LIBRARYS}
        ${DDK_PROTOBUF_LIBRARYS} 
        ${DDK_OPENCV_LIBRARYS})
    
    #
    set(DDK_FOUND TRUE)
endif(NOT DDK_FOUND)
