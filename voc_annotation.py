import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
DIR = 'F:/label/dataset/data_normalization_no_aug_lowresolution_fix/'

wd = getcwd()
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["glomerular_normal_0","glomerular_mild_1","glomerular_severe_2","glomerular_fibrosis_3"]
classes = ["glomerular"]

def convert_annotation(year, image_id, list_file):
    # in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    in_file = open(DIR + 'VOC%s/Annotations/%s.xml' % (year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    # list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
    list_file.write(DIR + 'VOC%s/JPEGImages/%s.jpg' % (year, image_id))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

for year, image_set in sets:
    # image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    image_ids = open(DIR + 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()
