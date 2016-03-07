cmake_minimum_required (VERSION 2.8.11)
project (object_detection_library)

set (object_detection_library_VERSION_MAJOR 1)
set (object_detection_library_VERSION_MINOR 0)

OPTION(USE_GPU "Use GPU" ON) # Enabled by default
IF(USE_GPU)
ADD_DEFINITIONS(-DGPU=1)
ENDIF(USE_GPU)

find_package(CUDA)
SET(CUDA_NVCC_FLAGS "--gpu-architecture=compute_20 --gpu-code=compute_20" CACHE STRING "nvcc flags" FORCE)
#CFLAGS+= -DGPU
#cuda_compile(../src/convolutional_kernels.cu ../src/deconvolutional_kernels.cu ../src/activation_kernels.cu ../src/im2col_kernels.cu ../src/col2im_kernels.cu ../src/blas_kernels.cu ../src/crop_layer_kernels.cu ../src/dropout_layer_kernels.cu ../src/maxpool_layer_kernels.cu ../src/softmax_layer_kernels.cu ../src/network_kernels.cu ../src/avgpool_layer_kernels.cu ../src/gemm.c ../src/utils.c ../src/cuda.c ../src/deconvolutional_layer.c ../src/convolutional_layer.c)
cuda_add_library(object_detection SHARED ../src/convolutional_kernels.cu ../src/deconvolutional_kernels.cu ../src/activation_kernels.cu ../src/im2col_kernels.cu ../src/col2im_kernels.cu ../src/blas_kernels.cu ../src/crop_layer_kernels.cu ../src/dropout_layer_kernels.cu ../src/maxpool_layer_kernels.cu ../src/softmax_layer_kernels.cu ../src/network_kernels.cu ../src/avgpool_layer_kernels.cu ../src/gemm.c ../src/utils.c ../src/cuda.c ../src/deconvolutional_layer.c ../src/convolutional_layer.c ../src/list.c ../src/image.c ../src/activations.c ../src/im2col.c ../src/col2im.c ../src/blas.c ../src/crop_layer.c ../src/dropout_layer.c ../src/maxpool_layer.c ../src/softmax_layer.c ../src/matrix.c ../src/network.c ../src/connected_layer.c ../src/cost_layer.c ../src/parser.c ../src/detection_layer.c ../src/route_layer.c ../src/box.c ../src/normalization_layer.c ../src/avgpool_layer.c ../src/layer.c ../src/local_layer.c ../src/shortcut_layer.c ../src/activation_layer.c ../src/rnn_layer.c ../src/option_list.c ../src/data.c object_detection.c)
# Create a library called "Hello" which includes the source file "hello.cxx".
# The extension is already found. Any number of sources could be listed here.
#add_library (object_detection SHARED ../src/convolutional_kernels.cu ../src/list.c ../src/image.c ../src/activations.c ../src/im2col.c ../src/col2im.c ../src/blas.c ../src/crop_layer.c ../src/dropout_layer.c ../src/maxpool_layer.c ../src/softmax_layer.c ../src/matrix.c ../src/network.c ../src/connected_layer.c ../src/cost_layer.c ../src/parser.c ../src/detection_layer.c ../src/route_layer.c ../src/box.c ../src/normalization_layer.c ../src/avgpool_layer.c ../src/layer.c ../src/local_layer.c ../src/shortcut_layer.c ../src/activation_layer.c ../src/rnn_layer.c ../src/option_list.c ../src/data.c object_detection.c)

#message(STATUS ${CUDA_LIBRARIES})
# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
target_include_directories (object_detection PRIVATE ../src ${CUDA_INCLUDE_DIRS})
#CUDA_ADD_CUBLAS_TO_TARGET(object_detection)
#target_link_libraries(object_detection m pthread object_detection_cuda)
target_link_libraries(object_detection m pthread ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})

#add_subdirectory(test)
