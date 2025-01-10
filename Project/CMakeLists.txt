cmake_minimum_required(VERSION 3.10)

project(MyProject CXX)
set(CMAKE_CXX_STANDARD 17)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含OpenCV的头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# 添加可执行文件，并链接OpenCV库
add_executable(MyExecutable ${SOURCES})
target_link_libraries(MyExecutable ${OpenCV_LIBS})

# 设置编译器优化等级为O3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
