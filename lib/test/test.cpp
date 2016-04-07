#include <time.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include "boost/filesystem.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "object_detection.hpp"

#define SSTR(x)                                                                \
  static_cast<std::ostringstream &>((std::ostringstream() << std::dec << x))   \
      .str()

using namespace cv;
using namespace std;
using namespace srzn_object_detection;
namespace fs = ::boost::filesystem;

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path &root, const string &ext, vector<fs::path> &ret) {
  if (!fs::exists(root) || !fs::is_directory(root))
    return;

  fs::recursive_directory_iterator it(root);
  fs::recursive_directory_iterator endit;

  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ext)
      ret.push_back(it->path().filename());
    ++it;
  }
}

int main() {
  // VideoCapture vcap("../../../haoyun_atm_videos/atm_env_1.avi");
  vector<fs::path> jpgs;
  get_all("../../../haoyun_atm_videos/haoyun_bank_zhaoqing/JPEGImages", ".jpg",
          jpgs);
  //cout << jpgs[0].string() << endl;
  Mat src;
  int i = 0;
  clock_t time;
  ObjectDetector object_detector(
      "../../../models_object_detection/7c.cfg",
      "../../../models_object_detection/7c_28000.weights");
  // while (vcap.read(src)) {
  // if (i++ % (25) != 0)
  // continue;
  for (int i = 0; i < jpgs.size(); ++i) {
    cout << jpgs[i].string() << endl;
    src = imread("../../../haoyun_atm_videos/haoyun_bank_zhaoqing/JPEGImages/"+
        jpgs[i].string());
    // imwrite("atm_env_1/"+SSTR(i)+".jpg", src);
    // Mat src = imread("../../../screenshot/1.jpeg");
    time = clock();

    vector<ObjectDetector::Object> objs = object_detector.Process(src);
    // printf("Detected in %f seconds.\n",
    //(double)(clock() - time) / CLOCKS_PER_SEC);
    // int *objs = objects;
    for (int j = 0; j < objs.size(); j++) {
      rectangle(src, objs[j].bounding_box, Scalar(200, 0, 0), 2);
      putText(src, SSTR(objs[j].probs),
              Point(objs[j].bounding_box.x, objs[j].bounding_box.y + 50),
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
