import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import isfile, join, basename
import glob
import random

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
bank_name = "bankcs"
sets=['day1', 'night1']
train_ratio = 0.7

# for year, image_set in sets:
for time in sets:
    if not os.path.exists('%s/%s/ImageSets/Main'%(bank_name, time)):
        os.makedirs('%s/%s/ImageSets/Main'%(bank_name, time))
    train_image_ids = open('%s/%s/ImageSets/Main/train.txt'%(bank_name, time), 'w')
    val_image_ids = open('%s/%s/ImageSets/Main/val.txt'%(bank_name, time), 'w')

    annotation_path = '%s/%s/Annotations'%(bank_name, time)
    xmlfiles = glob.glob('%s/*.xml'%annotation_path)
    #xmlfiles = [f for f in listdir(annotation_path)
                #if isfile(join(annotation_path, f)) and f.endswith('.xml')]
    train_num = int(train_ratio*len(xmlfiles))
    #train_files = random.sample(xmlfiles, train_num)
    random.shuffle(xmlfiles)
    train_files = xmlfiles[:train_num]
    val_files = xmlfiles[train_num:]

    for f in train_files:
        train_image_ids.write(basename(os.path.splitext(f)[0])+'\n')

    for f in val_files:
        val_image_ids.write(basename(os.path.splitext(f)[0])+'\n')

    train_image_ids.close()
    val_image_ids.close()

