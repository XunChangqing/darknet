cmake_minimum_required (VERSION 2.8.11)
project (object_detection_srzn_test)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(CUDA)
#find_package(Boost REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})
#set(FaceRecSrzn_DIR "/home/mythxcq/work/face_recognition_srzn/face_rec_lib/install/lib/CMake/FaceRecSrzn")
#find_package(FaceRecSrzn REQUIRED)
include_directories("../include")
link_directories("../lib")
set(ObjectDetectionSrzn_LIBRARIES "object_detection_srzn")

#find_package(boost REQUIRED)
#include_directories(${boost_INCLUDE_DIRS})

add_executable(test test.cpp)
target_link_libraries(test ${OpenCV_LIBRARIES} ${ObjectDetectionSrzn_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})

