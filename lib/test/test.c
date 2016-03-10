#include <time.h>
#include <stdio.h>

#include "opencv2/highgui/highgui_c.h"
#include "object_detection.h"

int main()
{
  void *detector = detector_init("model/448_version_1.cfg", "model/448_version_1.weights");
  clock_t time;
  int objects[5*MAX_OBJECTS] = {0};
  int obj_num;
  {
    //给定文件名，sdk自行加载并检测
    time=clock();
    obj_num = detector_process_file(detector, "test_images/person.jpg", objects);
    printf("Detected in %f seconds.\n", (double)(clock()-time)/CLOCKS_PER_SEC);
    int *objs = objects;
    int j;
    for(j=0;j<obj_num;j++)
    {
      printf("%d, %d, %d, %d, %d\n", objs[0], objs[1], objs[2], objs[3], objs[4]); 
      objs += 5;
    }
  }

  {
    //给定图像所存储的缓存，sdk负责检测
    //使用CV加载图像以后，需要进行处理，算法接受float类型的图像数据
    IplImage *src = cvLoadImage("test_images/person.jpg", 1);
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    float *img_buf = calloc(h*w*c, sizeof(float));
    int i, j, k, count=0;;

    for(k= 0; k < c; ++k){
      for(i = 0; i < h; ++i){
        for(j = 0; j < w; ++j){
          img_buf[count++] = data[i*step + j*c + k]/255.;
        }
      }
    }

    cvReleaseImage(&src);
    time=clock();
    obj_num = detector_process_buffer(detector, img_buf, w, h, objects);
    printf("Detected in %f seconds.\n", (double)(clock()-time)/CLOCKS_PER_SEC);
    int *objs = objects;
    for(j=0;j<obj_num;j++)
    {
      printf("%d, %d, %d, %d, %d\n", objs[0], objs[1], objs[2], objs[3], objs[4]); 
      objs += 5;
    }
  }

  /*cvWaitKey(0);*/
  /*cvDestroyAllWindows();*/
  return 0;
}
