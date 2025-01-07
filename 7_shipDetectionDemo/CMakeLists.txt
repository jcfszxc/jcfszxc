cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(ShipDetectionTG465)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加 OpenCV 的路径
set(OpenCV_DIR "/home/firefly/opencv_build/opencv-4.5.4/build")
find_package(OpenCV REQUIRED)

# 如果找不到 OpenCV，输出错误信息
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please specify OpenCV_DIR.")
endif()

# 设置可能的 RKNN 库路径
set(RKNN_LIB_PATHS
    "/usr/lib"
)

# 添加 RK RGA 库的路径
set(RK_RGA_INCLUDE_DIR "/home/firefly/local/rkrga/include")
set(RK_RGA_LIB_DIR "/home/firefly/local/rkrga/lib")

# 查找 RKNN 库
find_library(RKNN_LIBRARY
    NAMES rknn_api
    PATHS ${RKNN_LIB_PATHS}
)

if(NOT RKNN_LIBRARY)
    message(FATAL_ERROR "RKNN library not found. Please check the library paths.")
endif()

# 获取 RKNN 库的目录
get_filename_component(RKNN_LIB_DIR ${RKNN_LIBRARY} DIRECTORY)

# 设置 RKNN 头文件目录
set(RKNN_INCLUDE_DIR "${RKNN_LIB_DIR}/../include")

# 设置 Rockchip MPP 库路径和头文件路径
set(RK_MPP_INCLUDE_DIR "/home/firefly/local/rkmpp/include/rockchip")
set(RK_MPP_LIBRARY_DIR "/home/firefly/local/rkmpp/lib")

# 查找 Rockchip MPP 库
find_library(ROCKCHIP_MPP_LIBRARY
    NAMES rockchip_mpp
    PATHS ${RK_MPP_LIBRARY_DIR}
)

if(NOT ROCKCHIP_MPP_LIBRARY)
    message(FATAL_ERROR "Rockchip MPP library not found. Please install Rockchip MPP library.")
else()
    message(STATUS "Found Rockchip MPP library: ${ROCKCHIP_MPP_LIBRARY}")
endif()

# 添加 FFmpeg 自定义路径
set(FFMPEG_ROOT "/home/firefly/local/ffmpeg")

# 添加额外库的链接
set(FFMPEG_LIBRARIES
    ${FFMPEG_ROOT}/lib/libavcodec.a
    ${FFMPEG_ROOT}/lib/libavformat.a
    ${FFMPEG_ROOT}/lib/libavutil.a
    ${FFMPEG_ROOT}/lib/libswscale.a
    ${FFMPEG_ROOT}/lib/libswresample.a
    /usr/lib/aarch64-linux-gnu/libx264.so  # 显式链接 x264
    /usr/lib/aarch64-linux-gnu/libx265.so  # 显式链接 x265
    /home/firefly/local/rkmpp/lib/librockchip_mpp.so  # 显式链接 rkmpp
    m  # Math library
    pthread  # POSIX threads library
    dl  # Dynamic loading library
    z  # zlib compression library (libz)
    lzma  # LZMA compression library (liblzma)
    drm  # Direct Rendering Manager library (libdrm)
)


# 查找 librtmp 库
set(LIBRTMP_INCLUDE_DIR "/usr/include/librtmp")
set(LIBRTMP_LIBRARY_PATHS
    "/usr/lib/aarch64-linux-gnu"
    "/home/firefly/libs/lib"
)

find_library(LIBRTMP_LIBRARY
    NAMES rtmp
    PATHS ${LIBRTMP_LIBRARY_PATHS}
)

if(NOT LIBRTMP_LIBRARY)
    message(FATAL_ERROR "librtmp library not found. Please check the library paths.")
else()
    message(STATUS "Found librtmp library: ${LIBRTMP_LIBRARY}")
endif()

# 查找所有的源文件（包括 .cpp 和 .cc 文件）
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cc")

# 查找所有的头文件
file(GLOB_RECURSE HEADERS "include/*.h")

# 添加可执行文件
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 添加头文件目录
target_include_directories(${PROJECT_NAME} PRIVATE 
    include 
    ${OpenCV_INCLUDE_DIRS}
    ${RK_MPP_INCLUDE_DIR}
    ${RKNN_INCLUDE_DIR}
    ${RK_RGA_INCLUDE_DIR}
    ${FFMPEG_ROOT}/include  # 添加 FFmpeg 头文件目录
    ${LIBRTMP_INCLUDE_DIR}
)

# 链接库列表
set(LINK_LIBRARIES
    ${OpenCV_LIBS}
    ${ROCKCHIP_MPP_LIBRARY}
    ${RKNN_LIBRARY}
    ${RK_RGA_LIB_DIR}/librga.so
    ${FFMPEG_LIBRARIES}
    ${LIBRTMP_LIBRARY}
)

# 设置编译选项
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS)

# 链接所有库
target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_LIBRARIES})

# 输出库信息（可选，用于调试）
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "Rockchip MPP library status:")
message(STATUS "    include path: ${RK_MPP_INCLUDE_DIR}")
message(STATUS "    library: ${ROCKCHIP_MPP_LIBRARY}")

message(STATUS "FFmpeg library status:")
message(STATUS "    Using static FFmpeg libraries.")

message(STATUS "RKNN library status:")
message(STATUS "    library: ${RKNN_LIBRARY}")
message(STATUS "    include path: ${RKNN_INCLUDE_DIR}")

message(STATUS "librtmp library status:")
message(STATUS "    library: ${LIBRTMP_LIBRARY}")
