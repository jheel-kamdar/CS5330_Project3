cmake_minimum_required(VERSION 3.10)
project(NewMethod)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(NewMethod task9.cpp)
target_link_libraries(NewMethod ${OpenCV_LIBS})