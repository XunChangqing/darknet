#ifndef HEADER_OBJECT_DETECTION
#define HEADER_OBJECT_DETECTION

#ifdef __cplusplus
extern "C" {
#endif

void* detector_init(char *cfg_file, char *weight_file);
//最多返回10个物体，每个物体的x，y（左上角坐标），w，h（宽和高），class（类型，0为人员，1为车辆）
//objects: x,y,w,h,class(0-person, 1-car)
#define MAX_OBJECTS (10)
int detector_process_buffer(void *detector, float *image_buffer, int w, int h, int *objects);
int detector_process_file(void *detector, char *filename, int *objects);
void detector_release(void *detector);

#ifdef __cplusplus
}
#endif

#endif
