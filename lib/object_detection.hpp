#ifndef HEADER_OBJECT_DETECTION
#define HEADER_OBJECT_DETECTION
// object detection library
// author: xun changqing
// 2016.3.10
// email: xunchangqing AT qq.com

#include "opencv2/opencv.hpp"

struct network;
namespace srzn_object_detection {
using namespace cv;
class ObjectDetector {
public:
  struct Object {
    Rect bounding_box;
    float probs;
    int type;
  };
  ObjectDetector(const string &cfg_file, const string &weight_file);
  vector<Object> Process(const Mat &img);

private:
  network *conv_net;
};
}

#endif
