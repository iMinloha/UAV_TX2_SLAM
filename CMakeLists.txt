cmake_minimum_required(VERSION 3.26)
project(DeepVision)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR E:/opencv/install/x64/mingw/lib)
# 寻找必要文件
find_package(OpenCV REQUIRED)
# 导入OpenCV的头
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加所有项目文件夹
file(GLOB_RECURSE DeepVision_SRC "DeepGraph/*.*")
file(GLOB_RECURSE BuildMap_SRC "BuildMap/*.*")
file(GLOB_RECURSE DataStruct_SRC "DataStruct/*.*")

add_executable(DeepVision main.cpp ${DeepVision_SRC} ${BuildMap_SRC} ${DataStruct_SRC}
        DeepGraph/DeepGraphCalculate.hpp
        DeepGraph/DeepGraph.h)

target_link_libraries(DeepVision ${OpenCV_LIBS})
