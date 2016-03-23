import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
#bank_name = "bankcs"
#sets=[('day1', 'train'), ('day1', 'val'), ('day2', 'train'), ('day2', 'val'),
      #('night1', 'train'), ('night1', 'val')]
bank_name = "bankcs"
sets=['night1']

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#name_dict = {'person':'person_night', 'person_atm_face':'person_atm_face_night',
             #'person_atm_card':'person_atm_card_night'}
name_dict = {'person_night':'person', 'person_atm_face':'person_atm_face_night',
             'person_atm_card':'person_atm_card_night'}

#import pdb; pdb.set_trace()  # XXX BREAKPOINT

for time in sets:
    annotation_path = '%s/%s/Annotations'%(bank_name, time)
    xmlfiles = glob.glob('%s/*.xml'%annotation_path)
    for anno_file in xmlfiles:
        tree = ET.parse(anno_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            name_ele = obj.find('name')
            if name_dict.has_key(name_ele.text):
                name_ele.text = name_dict[name_ele.text]
            #for name in obj.iter('name'):
                #print name.text
        tree.write(anno_file)
        print anno_file

