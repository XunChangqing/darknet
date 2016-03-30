#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#include "object_detection.h"

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
image voc_labels[20];

void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
  int i,j,n;
  //int per_cell = 5*num+classes;
  for (i = 0; i < side*side; ++i){
    int row = i / side;
    int col = i % side;
    for(n = 0; n < num; ++n){
      int index = i*num + n;
      int p_index = side*side*classes + i*num + n;
      float scale = predictions[p_index];
      int box_index = side*side*(classes + num) + (i*num + n)*4;
      boxes[index].x = (predictions[box_index + 0] + col) / side * w;
      boxes[index].y = (predictions[box_index + 1] + row) / side * h;
      boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
      boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
      for(j = 0; j < classes; ++j){
        int class_index = i*classes;
        float prob = scale*predictions[class_index+j];
        probs[index][j] = (prob > thresh) ? prob : 0;
      }
      if(only_objectness){
        probs[index][0] = scale;
      }
    }
  }
}

int write_out_detections(int num, float thresh, box *boxes, float **probs, int classes, int w, int h, int *objects)
{
  int i;
  int obj_idx = 0;
  for(i = 0; i < num; ++i){
    int class = max_index(probs[i], classes);
    float prob = probs[i][class];
    if(prob > thresh 
        && obj_idx<MAX_OBJECTS 
        && (class == 6 || class == 14)){
      box b = boxes[i];

      int left  = (b.x-b.w/2.)*w;
      int right = (b.x+b.w/2.)*w;
      int top   = (b.y-b.h/2.)*h;
      int bot   = (b.y+b.h/2.)*h;

      if(left < 0) left = 0;
      if(right > w-1) right = w-1;
      if(top < 0) top = 0;
      if(bot > h-1) bot = h-1;

      /*draw_box_width(im, left, top, right, bot, width, red, green, blue);*/
      objects[0] = left;
      objects[1] = top;
      objects[2] = right-left;
      objects[3] = bot-top;
      objects[4] = class==6?1:0;
      obj_idx++;
      objects = objects+5;
    }
  }
  return obj_idx;
}

void* detector_init(const char *cfg_file, const char *weight_file)
{
  gpu_index = 0;
  network net = parse_network_cfg(cfg_file);
  if(weight_file){
    load_weights(&net, weight_file);
  }
  else
    return NULL;
  set_batch_network(&net, 1);
  srand(2222222);
  network *ret_net = calloc(1, sizeof(network));
  *ret_net = net;
  return (void*)ret_net;
}
int detector_process_image(void *pnetwork, image im, int *objects)
{
  network net = *(network*)pnetwork;
  detection_layer l = net.layers[net.n-1];
  int j;
  float nms=.5;
  float thresh = .2;
  box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
  float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
  for(j = 0; j < l.side*l.side*l.n; ++j) 
    probs[j] = calloc(l.classes, sizeof(float *));

  image sized = resize_image(im, net.w, net.h);
  float *X = sized.data;
  float *predictions = network_predict(net, X);
  convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
  if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
  int obj_num = write_out_detections(l.side*l.side*l.n, thresh, boxes, probs, 20, im.w, im.h, objects);

  free(boxes);
  free_image(sized);
  for(j = 0; j < l.side*l.side*l.n; ++j) 
    free(probs[j]);
  free(probs);
  return obj_num;
}

int detector_process_buffer(void *detector, float *image_buffer, int w, int h, int *objects)
{
  image im = float_to_image(w, h, 3, (float*)image_buffer);
  return detector_process_image(detector, im, objects);
}

int detector_process_file(void *detector, char *filename, int *objects)
{
  image im = load_image_color(filename,0,0);
  int ret = detector_process_image(detector, im, objects);
  free_image(im);
  return ret;
}

void detector_release(void *detector)
{
}
