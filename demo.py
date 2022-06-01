#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image

yolo = YOLO()

img_path = 'img/120126-1.jpg'
try:
    image = Image.open(img_path)
except:
    print('Open Error! Try again!')
else:
    r_image = yolo.detect_image(image)
    r_image.show()
# while True:
#     img_path = input('Input image filename:')
#     # img_path = 'img/street.jpg'
#     try:
#         image = Image.open(img_path)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = yolo.detect_image(image)
#         r_image.show()


















# import argparse
# import torch
# import time
# import numpy as np
# import math
# import cv2
# from nets.yolo4 import YoloBody
# import matplotlib.pyplot as plt
#
# # """hyper parameters"""
# use_cuda = True
#
# #---------------------------------------------------#
# #   获得类和先验框
# #---------------------------------------------------#
# def get_classes(classes_path):
#     '''loads the classes'''
#     with open(classes_path) as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#     return class_names
#
# def get_anchors(anchors_path):
#     '''loads the anchors from a file'''
#     with open(anchors_path) as f:
#         anchors = f.readline()
#     anchors = [float(x) for x in anchors.split(',')]
#     return np.array(anchors).reshape([-1,3,2])[::-1,:,:]
#
#
#
# def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
#     model.eval()
#     t0 = time.time()
#
#     if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
#         img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
#     elif type(img) == np.ndarray and len(img.shape) == 4:
#         img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
#     else:
#         print("unknow image type")
#         exit(-1)
#
#     if use_cuda:
#         img = img.cuda()
#     img = torch.autograd.Variable(img)
#
#     t1 = time.time()
#
#     output = model(img)
#
#     t2 = time.time()
#
#     print('-----------------------------------')
#     print('           Preprocess : %f' % (t1 - t0))
#     print('      Model Inference : %f' % (t2 - t1))
#     print('-----------------------------------')
#
#     return post_processing(img, conf_thresh, nms_thresh, output)
#
# def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
#     # print(boxes.shape)
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#
#     areas = (x2 - x1) * (y2 - y1)
#     order = confs.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         idx_self = order[0]
#         idx_other = order[1:]
#
#         keep.append(idx_self)
#
#         xx1 = np.maximum(x1[idx_self], x1[idx_other])
#         yy1 = np.maximum(y1[idx_self], y1[idx_other])
#         xx2 = np.minimum(x2[idx_self], x2[idx_other])
#         yy2 = np.minimum(y2[idx_self], y2[idx_other])
#
#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         inter = w * h
#
#         if min_mode:
#             over = inter / np.minimum(areas[order[0]], areas[order[1:]])
#         else:
#             over = inter / (areas[order[0]] + areas[order[1:]] - inter)
#
#         inds = np.where(over <= nms_thresh)[0]
#         order = order[inds + 1]
#
#     return np.array(keep)
#
# def post_processing(img, conf_thresh, nms_thresh, output):
#     # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
#     # num_anchors = 9
#     # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#     # strides = [8, 16, 32]
#     # anchor_step = len(anchors) // num_anchors
#
#     # [batch, num, 1, 4]
#     box_array = output[0]
#     # [batch, num, num_classes]
#     confs = output[1]
#
#     t1 = time.time()
#
#     if type(box_array).__name__ != 'ndarray':
#         box_array = box_array.cpu().detach().numpy()
#         confs = confs.cpu().detach().numpy()
#
#     num_classes = confs.shape[2]
#
#     # [batch, num, 4]
#     box_array = box_array[:, :, 0]
#
#     # [batch, num, num_classes] --> [batch, num]
#     max_conf = np.max(confs, axis=2)
#     max_id = np.argmax(confs, axis=2)
#
#     t2 = time.time()
#
#     bboxes_batch = []
#     for i in range(box_array.shape[0]):
#
#         argwhere = max_conf[i] > conf_thresh
#         l_box_array = box_array[i, argwhere, :]
#         l_max_conf = max_conf[i, argwhere]
#         l_max_id = max_id[i, argwhere]
#
#         bboxes = []
#         # nms for each class
#         for j in range(num_classes):
#
#             cls_argwhere = l_max_id == j
#             ll_box_array = l_box_array[cls_argwhere, :]
#             ll_max_conf = l_max_conf[cls_argwhere]
#             ll_max_id = l_max_id[cls_argwhere]
#
#             keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
#
#             if (keep.size > 0):
#                 ll_box_array = ll_box_array[keep, :]
#                 ll_max_conf = ll_max_conf[keep]
#                 ll_max_id = ll_max_id[keep]
#
#                 for k in range(ll_box_array.shape[0]):
#                     bboxes.append(
#                         [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k],
#                          ll_max_conf[k], ll_max_id[k]])
#
#         bboxes_batch.append(bboxes)
#
#     t3 = time.time()
#
#     print('-----------------------------------')
#     print('       max and argmax : %f' % (t2 - t1))
#     print('                  nms : %f' % (t3 - t2))
#     print('Post processing total : %f' % (t3 - t1))
#     print('-----------------------------------')
#
#     return bboxes_batch
#
# def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
#     img = np.copy(img)
#     colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
#
#     def get_color(c, x, max_val):
#         ratio = float(x) / max_val * 5
#         i = int(math.floor(ratio))
#         j = int(math.ceil(ratio))
#         ratio = ratio - i
#         r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
#         return int(r * 255)
#
#     width = img.shape[1]
#     height = img.shape[0]
#     for i in range(len(boxes)):
#         box = boxes[i]
#         x1 = int(box[0] * width)
#         y1 = int(box[1] * height)
#         x2 = int(box[2] * width)
#         y2 = int(box[3] * height)
#
#         if color:
#             rgb = color
#         else:
#             rgb = (255, 0, 0)
#         if len(box) >= 7 and class_names:
#             cls_conf = box[5]
#             cls_id = box[6]
#             print('%s: %f' % (class_names[cls_id], cls_conf))
#             classes = len(class_names)
#             offset = cls_id * 123457 % classes
#             red = get_color(2, offset, classes)
#             green = get_color(1, offset, classes)
#             blue = get_color(0, offset, classes)
#             if color is None:
#                 rgb = (red, green, blue)
#             img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
#     if savename:
#         print("save plot results to %s" % savename)
#         cv2.imwrite(savename, img)
#     plt.plot(img)
#     plt.show()
#     return img
#
# def detect_cv2(weightfile, imgfile):
#     anchors_path = 'model_data/yolo_anchors.txt'
#     classes_path = 'model_data/voc_classes.txt'
#     class_names = get_classes(classes_path)
#     anchors = get_anchors(anchors_path)
#     num_classes = len(class_names)
#     # 创建模型
#     m = YoloBody(len(anchors[0]), num_classes).cuda()
#     model_path = "model_data/yolo4_voc_weights.pth"
#     # 加快模型训练的效率
#     print('Loading weights into state dict...')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model_dict = m.state_dict()
#     pretrained_dict = torch.load(model_path, map_location=device)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
#     model_dict.update(pretrained_dict)
#     m.load_state_dict(model_dict)
#     print('Finished!')
#
#     # if use_cuda:
#     #     m.cuda()
#
# #dataloader
#     img = cv2.imread(imgfile)
#     sized = cv2.resize(img, (416, 416))
#     sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
#
#     for i in range(2):
#         start = time.time()
#         boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
#         finish = time.time()
#         if i == 1:
#             print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
#
#     plot_boxes_cv2(img, boxes[0], savename='data/predictions.jpg', class_names=class_names)
#
# def get_args():
#     parser = argparse.ArgumentParser('Test your image or video by trained model.')
#     parser.add_argument('-weightfile', type=str,
#                         default='./model_data/yolo4_voc_weights.pth',# yolo4_new_weights  yolo4_voc_weights     yolov4.weights
#                         help='path of trained model.', dest='weightfile')
#     parser.add_argument('-imgfile', type=str,
#                         default='./img/street.jpg',
#                         help='path of your image file.', dest='imgfile')
#     args = parser.parse_args()
#
#     return args
#
#
# if __name__ == '__main__':
#     args = get_args()
#     if args.imgfile:
#         detect_cv2(args.weightfile, args.imgfile)