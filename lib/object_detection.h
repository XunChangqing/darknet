#ifndef HEADER_OBJECT_DETECTION
#define HEADER_OBJECT_DETECTION

#ifdef __cplusplus
extern "C" {
#endif

//初始化检测器，检测器以void指针类型返回，该指针后续的调用都需要
void* detector_init(char *cfg_file, char *weight_file);
//最多返回10个物体，每个物体的x，y（左上角坐标），w，h（宽和高），class（类型，0为人员，1为车辆）
//objects: x,y,w,h,class(0-person, 1-car)
#define MAX_OBJECTS (10)
//image_buffer为指向图像数据内容的指针，w和h分别为宽和高，必须为3通道，objects为返回的物体信息
int detector_process_buffer(void *detector, float *image_buffer, int w, int h, int *objects);
int detector_process_file(void *detector, char *filename, int *objects);
void detector_release(void *detector);

#ifdef __cplusplus
}
#endif

#endif
