cmake_minimum_required(VERSION 3.10)
project(ObjectRecognition)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(ObjectRecognition src/main.cpp)
target_link_libraries(ObjectRecognition ${OpenCV_LIBS})