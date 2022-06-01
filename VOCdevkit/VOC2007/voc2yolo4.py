import os
import random 

filepath = 'F:/label/dataset/data_normalization_no_aug_lowresolution_fix/VOC2007/'
xmlfilepath = filepath + 'Annotations'
saveBasePath = filepath + 'ImageSets/Main/'
# xmlfilepath = 'E:/code/yolov4-pytorch-master-1-bf/VOCdevkit/VOC2007/Annotations'#r'./VOCdevkit/VOC2007/Annotations'
# saveBasePath = 'E:/code/yolov4-pytorch-master-1-bf/VOCdevkit/VOC2007/ImageSets/Main/'#r"./VOCdevkit/VOC2007/ImageSets/Main/"

trainval_percent=0.9
train_percent=1

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
