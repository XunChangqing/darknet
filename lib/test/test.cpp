#include <time.h>
#include <stdio.h>
#include <iostream>
#include <sstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "object_detection.h"

#define SSTR(x) static_cast< std::ostringstream &>(\
    (std::ostringstream() << std::dec<<x)).str()

using namespace cv;
using namespace std;
int main() {
  void *detector =
      detector_init(string("../../../weights_models_ours/yolo_7c.cfg").c_str(),
                    string("../../../weights_models_ours/yolo_7c_28000.weights").c_str());
  clock_t time;
  int objects[5 * MAX_OBJECTS] = {0};
  int obj_num;
  VideoCapture vcap("../../../haoyun_atm_videos/atm_env_1.avi");
  Mat src;
  int i=0;
  cout<<"start!"<<endl;
  float *img_buf = new float[448*448*sizeof(float)*3];
  while (vcap.read(src)) {
    if(i++%(25*60) != 0)
      continue;
    imwrite("atm_env_1/"+SSTR(i)+".jpg", src);
    //Mat src = imread("../../../screenshot/3.jpeg");
    //给定图像所存储的缓存，sdk负责检测
    //使用CV加载图像以后，需要进行处理，算法接受float类型的图像数据
    // Mat src = imread("test_images/person.jpg");
    // IplImage *src = cvLoadImage("test_images/person.jpg", 1);
    time = clock();

    Mat dst;
    resize(src, dst, Size(448,448));

    unsigned char *data = (unsigned char *)dst.data;
    int h = dst.rows;
    int w = dst.cols;
    int c = dst.channels();
    // float *img_buf = calloc(h*w*c, sizeof(float));
    //float *img_buf = new float[h * w * c];

    dst.convertTo(dst, CV_32F, 1.f/255);
    Mat channels[3];
    split(dst, channels);
    //cout<<"Continuous: "<<channels[0].isContinuous()<<" "
      //<<channels[1].isContinuous()<<" "
      //<<channels[2].isContinuous()<<endl;
    int width = 448*448*sizeof(float);
    memcpy(img_buf, channels[0].data, width);
    memcpy(img_buf+448*448, channels[1].data, width);
    memcpy(img_buf+2*448*448, channels[2].data, width); 
    
    //int step = dst.step1();
    //int i, j, k, count = 0;

    //for (k = 0; k < c; ++k) {
      //for (i = 0; i < h; ++i) {
        //for (j = 0; j < w; ++j) {
          //img_buf[count++] = data[i * step + j * c + k] / 255.;
        //}
      //}
    //}
    //std::cout<<"after transform!"<<std::endl;

    obj_num = detector_process_buffer(detector, img_buf, src.cols, src.rows, objects);
    printf("Detected in %f seconds.\n",
           (double)(clock() - time) / CLOCKS_PER_SEC);
    int *objs = objects;
    for (int j = 0; j < obj_num; j++) {
      // printf("%d, %d, %d, %d, %d\n", objs[0], objs[1], objs[2], objs[3],
      // objs[4]);
      rectangle(src, Point(objs[0], objs[1]),
                Point(objs[0] + objs[2], objs[1] + objs[3]), Scalar(200, 0, 0),
                2);
      objs += 5;
    }
    resize(src, src, Size(0,0), 0.5f, 0.5f);
    imshow("src", src);
    waitKey(1);
    //if (waitKey(30) > 0)
      //break;
  }

  /*cvWaitKey(0);*/
  /*cvDestroyAllWindows();*/
  return 0;
}
