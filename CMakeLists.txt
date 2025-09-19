cmake_minimum_required(VERSION 3.10)
project(PanoramaStitching)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(PanoramaStitching main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})