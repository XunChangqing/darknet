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

# for year, image_set in sets:
for time in sets:
    if not os.path.exists('%s/%s/ImageSets/Main'%(bank_name, time)):
        os.makedirs('%s/%s/ImageSets/Main'%(bank_name, time))
    unannotated_image_file = open('%s/%s/ImageSets/Main/unannotated.txt'%(bank_name, time), 'w')

    annotation_path = '%s/%s/Annotations'%(bank_name, time)
    image_path = '%s/%s/JPEGImages'%(bank_name, time)
    xmlfiles = glob.glob('%s/*.xml'%annotation_path)
    xmlfiles = [basename(os.path.splitext(f)[0]) for f in xmlfiles ]
    imgfiles = glob.glob('%s/*.jpg'%image_path)
    imgfiles = [basename(os.path.splitext(f)[0]) for f in imgfiles ]


    for f in imgfiles:
        if f not in xmlfiles:
            unannotated_image_file.write(f+'\n')
            print f
    unannotated_image_file.close()
