cmake_minimum_required(VERSION 4.0)
project(frame_labeler)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(framelabeler ./FrameLabeler.cpp)
target_link_libraries(framelabeler ${OpenCV_LIBS} ${Boost_LIBRARIES})
