import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
bank_name = "bankcs"
sets=[('day1', 'train'), ('day1', 'val'), ('day1', 'unannotated'),
      ('night1', 'train'), ('night1', 'val'), ('night1', 'unannotated')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["person", "person_atm_face", "person_atm_card", "person_night",
           "person_atm_face_night", "person_atm_card_night"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(time, image_id):
    try:
        in_file = open('%s/%s/Annotations/%s.xml'%(bank_name, time, image_id))
        out_file = open('%s/%s/labels/%s.txt'%(bank_name, time, image_id), 'w')
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    except:
        pass

wd = getcwd()

# for year, image_set in sets:
for time, image_set in sets:
    if not os.path.exists('%s/%s/labels/'%(bank_name, time)):
        os.makedirs('%s/%s/labels/'%(bank_name, time))
    image_ids = open('%s/%s/ImageSets/Main/%s.txt'%(bank_name, time, image_set)).read().strip().split()
    list_file = open('%s/%s/%s.txt'%(bank_name, time, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/%s/%s/JPEGImages/%s.jpg\n'%(wd, bank_name, time, image_id))
        convert_annotation(time, image_id)
    list_file.close()

