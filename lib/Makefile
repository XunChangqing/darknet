GPU=0
OPENCV=0
DEBUG=0


ARCH= --gpu-architecture=compute_20 --gpu-code=compute_20

VPATH=../src/
LIBPATH=./
#LIB_STATIC=object_detection
LIB_DYNAMIC=object_detection.so.1.0
OBJDIR=./obj/

CC=gcc
NVCC=nvcc
OPTS=-Ofast
#LDFLAGS= -lm -pthread -lstdc++ 
LDFLAGS= -lm -lstdc++ 
COMMON= 
CFLAGS=-Wall -Wfatal-errors -fPIC -I../src

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

#OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o imagenet.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o rnn.o
#ifeq ($(GPU), 1) 
#OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o yolo_kernels.o coco_kernels.o
#endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o matrix.o network.o connected_layer.o cost_layer.o parser.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o object_detection.o
#rnn.o imagenet.o captcha.o writing.o nightmare.o data.o option_list.o darknet.o compare.o classifier.o swag.o coco.o dice.o yolo.o 
ifeq ($(GPU), 1) 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o 
#yolo_kernels.o coco_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj $(LIB_DYNAMIC)

#$(EXEC): $(OBJS)
	#$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(LIB_DYNAMIC): $(OBJS) $(LIB_OBJS)
	$(CC) -shared $(COMMON) $(CFLAGS) -Wl,-soname,object_detection.so.1 -o $@ $^ $(LDFLAGS)

#$(LIB_STATIC): $(OBJS)
	#ar -cvq .a ctest1.o ctest2.o

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

#mv libctest.so.1.0 /opt/lib
#ln -sf /opt/lib/libctest.so.1.0 /opt/lib/libctest.so.1
#ln -sf /opt/lib/libctest.so.1.0 /opt/lib/libctest.so

obj:
	mkdir -p obj
#results:
	#mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(LIB_DYNAMIC)

