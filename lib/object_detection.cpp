#ifdef __cplusplus
extern "C" {
#endif

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "cuda.h"

#ifdef __cplusplus
}
#endif

#include "opencv2/opencv.hpp"

#include "object_detection.hpp"

// char *voc_names[] = {"aeroplane", "bicycle",   "bird",        "boat",
//"bottle",    "bus",       "car",         "cat",
//"chair",     "cow",       "diningtable", "dog",
//"horse",     "motorbike", "person",      "pottedplant",
//"sheep",     "sofa",      "train",       "tvmonitor"};

namespace srzn_object_detection {
ObjectDetector::ObjectDetector(const string &cfg_file,
                               const string &weight_file) {
  // gpu_index = 0;
  /*cudaError_t status = cudaSetDevice(0);*/
  // cudaSetDevice(0);
  char *cfg_str = new char[cfg_file.length() + 1];
  strcpy(cfg_str, cfg_file.c_str());
  char *weight_str = new char[weight_file.length() + 1];
  strcpy(weight_str, weight_file.c_str());
  network net = parse_network_cfg(cfg_str);
  if (!weight_file.empty()) {
    load_weights(&net, weight_str);
  } else
    return;
  delete[] cfg_str;
  delete[] weight_str;

  set_batch_network(&net, 1);
  srand(2222222);
  // network *ret_net = calloc(1, sizeof(network));
  network *ret_net = new network;
  *ret_net = net;
  conv_net = ret_net;
}

static void convert_yolo_detections(float *predictions, int classes, int num,
                                    int square, int side, int w, int h,
                                    float thresh, float **probs, box *boxes,
                                    int only_objectness) {
  int i, j, n;
  // int per_cell = 5*num+classes;
  for (i = 0; i < side * side; ++i) {
    int row = i / side;
    int col = i % side;
    for (n = 0; n < num; ++n) {
      int index = i * num + n;
      int p_index = side * side * classes + i * num + n;
      float scale = predictions[p_index];
      int box_index = side * side * (classes + num) + (i * num + n) * 4;
      boxes[index].x = (predictions[box_index + 0] + col) / side * w;
      boxes[index].y = (predictions[box_index + 1] + row) / side * h;
      boxes[index].w = pow(predictions[box_index + 2], (square ? 2 : 1)) * w;
      boxes[index].h = pow(predictions[box_index + 3], (square ? 2 : 1)) * h;
      for (j = 0; j < classes; ++j) {
        int class_index = i * classes;
        float prob = scale * predictions[class_index + j];
        probs[index][j] = (prob > thresh) ? prob : 0;
      }
      if (only_objectness) {
        probs[index][0] = scale;
      }
    }
  }
}

static vector<ObjectDetector::Object>
write_out_detections(int num, float thresh, box *boxes, float **probs,
                     int classes, int w, int h) {
  vector<ObjectDetector::Object> ret_objs;
  int i;
  int obj_idx = 0;
  for (i = 0; i < num; ++i) {
    int obj_cls = max_index(probs[i], classes);
    float prob = probs[i][obj_cls];
    if (prob > thresh)
      printf("probs: %f %f", probs[i][0], probs[i][6]);
    // if (prob > thresh && obj_idx < MAX_OBJECTS && (class == 6 || class ==
    // 14))
    //{
    /*if (prob > thresh && obj_idx < MAX_OBJECTS && class == 0) {*/
    if (probs[i][0] > thresh) {
      box b = boxes[i];

      int left = (b.x - b.w / 2.) * w;
      int right = (b.x + b.w / 2.) * w;
      int top = (b.y - b.h / 2.) * h;
      int bot = (b.y + b.h / 2.) * h;

      if (left < 0)
        left = 0;
      if (right > w - 1)
        right = w - 1;
      if (top < 0)
        top = 0;
      if (bot > h - 1)
        bot = h - 1;

      /*draw_box_width(im, left, top, right, bot, width, red, green, blue);*/
      ObjectDetector::Object new_obj;
      new_obj.bounding_box = Rect(left, top, right - left, bot - top);
      new_obj.probs = probs[i][0];
      new_obj.type = 0;
      ret_objs.push_back(new_obj);
    }
  }
  printf("\n");
  return ret_objs;
}

vector<ObjectDetector::Object> ObjectDetector::Process(const Mat &img) {
  Mat dst;
  resize(img, dst, Size(conv_net->w, conv_net->h));

  dst.convertTo(dst, CV_32F, 1.f / 255);
  Mat channels[3];
  split(dst, channels);
  int width = conv_net->w * conv_net->h * sizeof(float);
  unsigned char *img_buf = new unsigned char[width * 3];
  memcpy(img_buf, channels[0].data, width);
  memcpy(img_buf + width, channels[1].data, width);
  memcpy(img_buf + 2 * width, channels[2].data, width);

  detection_layer l = conv_net->layers[conv_net->n - 1];
  int j, k;
  float nms = .5;
  //float thresh = .2;
  float thresh = .3;
  box *boxes = (box *)calloc(l.side * l.side * l.n, sizeof(box));
  float **probs = (float **)calloc(l.side * l.side * l.n, sizeof(float *));
  for (j = 0; j < l.side * l.side * l.n; ++j)
    probs[j] = (float *)calloc(l.classes, sizeof(float *));

  clock_t time;
  float *X = (float *)img_buf;
  /*time = clock();*/
  float *predictions = network_predict(*conv_net, X);
  /*printf("Predict in %f seconds.\n",*/
  /*(double)(clock() - time) / CLOCKS_PER_SEC);*/
  convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1,
                          thresh, probs, boxes, 0);

  if (l.classes == 7) {
    for (k = 0; k < l.side * l.side * l.n; ++k) {
      int obj_cls = max_index(probs[k], l.classes - 1);
      float max_prob = probs[k][obj_cls];
      probs[k][obj_cls] = 0.0f;
      probs[k][0] = max_prob;
    }
  }

  if (nms)
    do_nms_sort(boxes, probs, l.side * l.side * l.n, l.classes, nms);

  vector<ObjectDetector::Object> ret_objs =
      write_out_detections(l.side * l.side * l.n, thresh, boxes, probs,
                           l.classes, img.cols, img.rows);

  free(boxes);
  for (j = 0; j < l.side * l.side * l.n; ++j)
    free(probs[j]);
  free(probs);
  delete[] img_buf;
  // return obj_num;
  return ret_objs;
}
}
