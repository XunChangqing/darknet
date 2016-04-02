#include <time.h>
#include <stdio.h>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "object_detection.h"

using namespace cv;
int main() {
  void *detector =
      detector_init(string("../../../weights_models_ours/yolo_7c.cfg").c_str(),
                    string("../../../weights_models_ours/yolo_7c_28000.weights").c_str());
  clock_t time;
  int objects[5 * MAX_OBJECTS] = {0};
  int obj_num;
  //VideoCapture vcap(0);
  //Mat src;
  //while (vcap.read(src)) {
  {
    Mat src = imread("../../../screenshot/3.jpeg");
    //给定图像所存储的缓存，sdk负责检测
    //使用CV加载图像以后，需要进行处理，算法接受float类型的图像数据
    // Mat src = imread("test_images/person.jpg");
    // IplImage *src = cvLoadImage("test_images/person.jpg", 1);
    unsigned char *data = (unsigned char *)src.data;
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    int step = src.step1();
    // float *img_buf = calloc(h*w*c, sizeof(float));
    float *img_buf = new float[h * w * c];
    int i, j, k, count = 0;
    ;

    for (k = 0; k < c; ++k) {
      for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
          img_buf[count++] = data[i * step + j * c + k] / 255.;
        }
      }
    }
    std::cout<<"after transform!"<<std::endl;

    time = clock();
    obj_num = detector_process_buffer(detector, img_buf, w, h, objects);
    printf("Detected in %f seconds.\n",
           (double)(clock() - time) / CLOCKS_PER_SEC);
    int *objs = objects;
    for (j = 0; j < obj_num; j++) {
      // printf("%d, %d, %d, %d, %d\n", objs[0], objs[1], objs[2], objs[3],
      // objs[4]);
      rectangle(src, Point(objs[0], objs[1]),
                Point(objs[0] + objs[2], objs[1] + objs[3]), Scalar(200, 0, 0),
                2);
      objs += 5;
    }
    delete[] img_buf;
    imshow("src", src);
    waitKey(0);
    //if (waitKey(30) > 0)
      //break;
  }

  /*cvWaitKey(0);*/
  /*cvDestroyAllWindows();*/
  return 0;
}
