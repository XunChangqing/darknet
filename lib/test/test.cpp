#include <time.h>
#include <stdio.h>
#include <iostream>
#include <sstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "object_detection.hpp"

#define SSTR(x)                                                                \
  static_cast<std::ostringstream &>((std::ostringstream() << std::dec << x))   \
      .str()

using namespace cv;
using namespace std;
using namespace srzn_object_detection;
int main() {
  ObjectDetector object_detector(
      "../../../weights_models_ours/yolo_7c.cfg",
      "../../../weights_models_ours/yolo_7c_28000.weights");
  VideoCapture vcap("../../../haoyun_atm_videos/atm_env_1.avi");
  Mat src;
  int i = 0;
  clock_t time;
  // while (vcap.read(src)) {
  {
    // if(i++%(25*60) != 0)
    // continue;
    // imwrite("atm_env_1/"+SSTR(i)+".jpg", src);
    Mat src = imread("../../../screenshot/1.jpeg");
    time = clock();

    vector<ObjectDetector::Object> objs = object_detector.Process(src);
    // printf("Detected in %f seconds.\n",
    //(double)(clock() - time) / CLOCKS_PER_SEC);
    // int *objs = objects;
    for (int j = 0; j < objs.size(); j++) {
      rectangle(src, objs[j].bounding_box, Scalar(200, 0, 0), 2);
      putText(src, SSTR(objs[j].probs),
              Point(objs[j].bounding_box.x, objs[j].bounding_box.y+50),
              CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 250));
    }
    resize(src, src, Size(0, 0), 0.5f, 0.5f);
    imshow("src", src);
    waitKey(0);
    // if (waitKey(30) > 0)
    // break;
  }

  /*cvWaitKey(0);*/
  /*cvDestroyAllWindows();*/
  return 0;
}
