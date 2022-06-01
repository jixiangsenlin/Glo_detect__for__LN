#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss,Generator
from nets.yolo4 import YoloBody


#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

def classes2twoclasses(targets):
    for b in range(len(targets)):
        for n_box in range(targets[b].shape[0]):
            if targets[b][n_box][4] > 0.0:
                targets[b][n_box][4] = 0.0

def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_size:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        # classes2twoclasses(targets)
        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        total_loss += loss
        waste_time = time.time() - start_time
        print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (total_loss/(iteration+1),waste_time))
        start_time = time.time()

    print('Start Validation')
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_size_val:
            break
        images_val, targets_val = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            else:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            # classes2twoclasses(targets_val)
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
            loss = sum(losses)
            val_loss += loss
    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    torch.save(model.state_dict(), 'model_data/yolo4_new_weights.pth')


#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    #-------------------------------#
    # input_shape = (512,1536)#(256,768)#(768,256)#(416,608)#(416,416) w h
    input_shape = (256,768)
    #-------------------------------#
    #   tricks的使用设置
    #-------------------------------#
    Cosine_lr = False
    mosaic = False#True
    # 用于设定是否使用cuda
    Cuda = True
    smoooth_label = 0
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    num_workers = 4 #8

    annotation_path = '2007_train.txt'
    #-------------------------------#
    #   获得先验框和类
    #-------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/voc_classes.txt'   
    class_names = get_classes(classes_path)
    # anchors = (get_anchors(anchors_path)/2).astype(int)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    # num_classes = 1# f b
    
    # 创建模型
    model = YoloBody(len(anchors[0]),num_classes)
    #-------------------------------------------#
    #   权值文件的下载请看README
    #-------------------------------------------#
    # model_path = "model_data/yolo4_weights.pth"
    # model_path = "model_data/yolo4_new_weights.pth"
    model_path = "model_data/yolo4_voc_weights.pth"
    # 加快模型训练的效率
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    # for k, v in pretrained_dict.items():
    #     if np.shape(model_dict[k]) == np.shape(v):#only save backbone.conv1 backbone.stages.0
    #         print("same",k,v.shape)# k:parameter_name  v:parameter_value
    #     else:print("different",k)

    # for k, v in pretrained_dict.items():
    #     if k in model_dict:#only save backbone.conv1 backbone.stages.0
    #         print("same",k,v.shape)# k:parameter_name  v:parameter_value
    #     else:print("different",k)

    # for k, v in pretrained_dict.items():
    #     if k in model_dict and np.shape(model_dict[k]) == np.shape(v):#only save backbone.conv1 backbone.stages.0
    #         print("same",k,v.shape)# k:parameter_name  v:parameter_value
    #     else:print("different",k)

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}#if k in model_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (np.shape(model_dict[k]) ==  np.shape(v))}  # if k in model_dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors,[-1,2]),num_classes, \
                                (input_shape[1], input_shape[0]), smoooth_label, Cuda))

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    freeze_epoch = 5#25

    # if not True:
    if True:
        lr = 1e-3
        Batch_size = 4#4
        Init_Epoch = 0
        Freeze_Epoch = freeze_epoch
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers,pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(mosaic = mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate(mosaic = False)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   冻结一定部分训练(冻结backbone:CSPDarkNet)
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 2
        Freeze_Epoch = freeze_epoch
        Unfreeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers,pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(mosaic = mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate(mosaic = False)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step()
