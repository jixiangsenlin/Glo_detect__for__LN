# -------------------------------------#
#       mAP所需文件计算代码 预测部分
# -------------------------------------#
import cv2
import numpy as np
import time
import colorsys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from yolo import YOLO
from nets.yolo4 import YoloBody
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as transforms
from utils.utils import non_max_suppression, bbox_iou, DecodeBox, letterbox_image, yolo_correct_boxes, letterbox_image_r
import matplotlib.pyplot as plt
from color_normalization import normalization

DIR = 'I:/code/color/original_data_6_not_mild_severe_fibrosis_incomplete_uncertain__without2_data2/'
# DIR = 'I:/code/color/data15/'
# DIR = '/data2/zxs/original_data_6_not_mild_severe_fibrosis_incomplete_uncertain__without2_data2/'
# DIR = 'F:/label/dataset/no_aug_data_change_size_img_xml/'
glo_label = ["glomerular_normal_0", "glomerular_mild_1", "glomerular_severe_2", "glomerular_fibrosis_3", "glomerular_incomplete_4"]



class mAP_Yolo(YOLO):
    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_id, image):
        self.confidence = 0.01# 0.05 0.001

        Rn = np.zeros([1, len(glo_label)],dtype=np.int64)  # 'Slightly diseased glomerulus', 'Severe diseased glomerulus', 'Fibrosis glomerulus', 'Incomplete glomerulus',

        # print(self.confidence)
        f = open("./input/detection-results/" + image_id + ".txt", "w")
        image_shape = np.array(np.shape(image)[0:2])#0h   1w
        # crop_img = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.ANTIALIAS)
        in_h = image_shape[0]
        in_w = image_shape[1]
        if in_w >= in_h * 3:
            w = self.model_image_size[1]
            h = int(in_h / in_w * w)
        else:
            h = self.model_image_size[0]
            w = int(in_w / in_h * h)
        crop_img = image.resize((w, h), Image.ANTIALIAS)
        HE2 = cv2.imread("131143-2.JPG")
        HE2 = cv2.cvtColor(HE2, cv2.COLOR_BGR2RGB)
        HE2 = cv2.resize(HE2, (w, h))
        crop_img = normalization(crop_img, HE2)
        crop_img = np.array(letterbox_image(Image.fromarray(np.uint8(crop_img*255)).convert('RGB'), (self.model_image_size[1], self.model_image_size[0])))


        # plt.imshow(crop_img)
        # plt.show()

        # # crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))

        # crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))

        # crop_img0 = np.array(letterbox_image(crop_img0, (self.model_image_size[1], self.model_image_size[0])))
        # HE2 = cv2.imread("131143-2.JPG")
        # crop_img = normalization(crop_img, HE2)
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres=self.confidence,
                                               nms_thres=0.3)

        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)


###############################
        norm_mean = [0.485, 0.456, 0.406]  #
        norm_std = [0.229, 0.224, 0.225]  #

        valid_transform = transforms.Compose([  #
            transforms.Resize((512, 512)),  #
            transforms.ToTensor(),  #
            transforms.Normalize(norm_mean, norm_std),  #
        ])
#############################





        predicted_classes = []
        predicteds = []
        imagen = np.array(image)

        for i, c in enumerate(top_label):
            # predicted_class = self.class_names[c]#!!!!!!!!!!!!!!!!!
            score = str(top_conf[i])
            # np.where(boxes[i] > 0, boxes[i], 0)
            # np.maximum(boxes[i], 0)
            top, left, bottom, right = boxes[i]
            if int(top) < 0:
                top = 0
            if int(left) < 0:
                left = 0
            if int(bottom) < 0:
                bottom = 0
            if int(right) < 0:
                right = 0
            if int(top) >= int(bottom) or int(left) >= int(right):
                continue


####################

            img_glo = imagen[int(top):int(bottom), int(left):int(right)]
            if img_glo.shape[1]==0 or img_glo.shape[0]==0 :
                continue
            # plt.imshow(img_glo)
            # plt.show()
            HE2_GLO = cv2.imread("tt.JPG")
            HE2_GLO = cv2.cvtColor(HE2_GLO, cv2.COLOR_BGR2RGB)
            # img_glo = cv2.resize(img_glo,(HE2_GLO.shape[1],HE2_GLO.shape[0]))
            # img_glo = normalization(img_glo,HE2=HE2_GLO)
            # print(img_glo.shape)
            HE2_GLO = cv2.resize(HE2_GLO, (img_glo.shape[1], img_glo.shape[0]))
            img_glo = normalization(img_glo, HE2=HE2_GLO)
            # # assert img_glo.size != 0
            # plt.imshow(img_glo)
            # plt.show()


            # crop_img_glo = np.array(img_glo)
            # img_glo = Image.fromarray(img_glo)
            # img_glo = Image.open('myplot.jpg')
            # plt.imshow(img_glo)
            # plt.show()
            img_glo = Image.fromarray(np.uint8(img_glo*255)).convert('RGB')
            # img_glo_s = img_glo

            # plt.imshow(img_glo)
            # plt.show()
            img_glo = valid_transform(img_glo)  # [C, H, W]
            img_glo = torch.unsqueeze(img_glo, dim=0)

            # crop_img_glo = np.resize(img_glo, (512, 512, 3))
            # photo_img_glo = np.array(crop_img_glo, dtype=np.float32)
            # photo_img_glo /= 255.0
            # photo_img_glo = np.transpose(photo_img_glo, (2, 0, 1))
            # photo_img_glo = photo_img_glo.astype(np.float32)
            # images_glo = []
            # images_glo.append(photo_img_glo)
            # images_glo = np.asarray(images_glo)

            # img_glo /= 255.0
            images_glo = img_glo

            self.classnet.eval()
            with torch.no_grad():
                # images_glo = torch.from_numpy(images_glo)
                if self.cuda:
                    images_glo = images_glo.cuda()
                outputs_glo = self.classnet(images_glo)
            _, predicted = torch.max(outputs_glo.data, 1)

            # if predicted == 0:
            #     continue

            # img_glo_s.save("glo_" + image_id + "__" + str(i) + "_" + str(predicted.numpy()[0]) + ".JPG")

            # # 0 4 ----- meaningless
            # if predicted == 4:
            #     predicted = predicted - 4

            predicted_class = glo_label[predicted]

            # print("predicted_class",predicted_class)
            predicted_classes.append(predicted_class)
            predicteds.append(predicted)
##################

            Rn[0][predicted] = Rn[0][predicted] + 1

            if predicted == 0:
                continue


            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()

        # print(Rn)
        # print(sum(Rn[0]))

        # font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(2.5e-2 * np.shape(image)[0] + 0.5).astype('int32'))

        # thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[1]

        for i, c in enumerate(top_label):
            # predicted_class = self.class_names[c]
            predicted_class = predicted_classes[i]
            predicted = predicteds[i]
            score = top_conf[i]

            if predicted == 0:
                continue

            top, left, bottom, right = boxes[i]

            if top < 0:
                top = 0
            if left < 0:
                left = 0
            if bottom < 0:
                bottom = 0
            if right < 0:
                right = 0
            if top >= bottom or left >= right:
                continue

            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            # label = '{} {:.2f}'.format('glo ', score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                # draw.rectangle(
                #     [left + i, top + i, right - i, bottom - i],
                #     outline=self.colors[self.class_names.index(predicted_class)])# glo_label
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[predicted])  # glo_label
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[self.class_names.index(predicted_class)])# glo_label
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[predicted])  # glo_label
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        # # image = letterbox_image_r(image, (in_w + 3000, in_h))
        # image = letterbox_image_r(image, (int(in_w + 0.5 * in_h), in_h))
        #
        # draw = ImageDraw.Draw(image)
        # draw.text(np.array([10, 50]), glo_label[1] + ": " + str(Rn[0][1]), fill=(0, 0, 0), font=font)
        # draw.text(np.array([10, 50 + int(in_h / 10)]), glo_label[2] + ": " + str(Rn[0][2]), fill=(0, 0, 0), font=font)
        # print(glo_label[2] + str(Rn[0][2]))
        # draw.text(np.array([10, 50 + int(in_h / 10) * 2]), glo_label[3] + ": " + str(Rn[0][3]), fill=(0, 0, 0),
        #           font=font)
        # draw.text(np.array([10, 50 + int(in_h / 10) * 3]), glo_label[4] + ": " + str(Rn[0][4]), fill=(0, 0, 0),
        #           font=font)
        # # draw.text(np.array([10, 50 + int(in_h / 10) * 4]), glo_label[5] + ": " + str(Rn[0][5]), fill=(0, 0, 0),
        # #           font=font)
        # draw.text(np.array([10, 50 + int(in_h / 10) * 5]), "glo num: " + str(sum(Rn[0])), fill=(0, 0, 0), font=font)
        # del draw

        image = letterbox_image_r(image, (int(in_w + 0.5 * in_h), in_h))

        draw = ImageDraw.Draw(image)
        draw.text(np.array([10, 50]), glo_label[1] + ": " + str(Rn[0][1]), fill=(0, 0, 0), font=font)
        draw.text(np.array([10, 50 + int(in_h / 10)]), glo_label[2] + ": " + str(Rn[0][2]), fill=(0, 0, 0), font=font)
        # print(glo_label[2] + str(Rn[0][2]))
        draw.text(np.array([10, 50 + int(in_h / 10) * 2]), glo_label[3] + ": " + str(Rn[0][3]), fill=(0, 0, 0),
                  font=font)
        draw.text(np.array([10, 50 + int(in_h / 10) * 3]), glo_label[4] + ": " + str(Rn[0][4]), fill=(0, 0, 0),
                  font=font)
        draw.text(np.array([10, 50 + int(in_h / 10) * 5]), "glo num: " + str(sum(Rn[0]) - Rn[0][0]),
                  fill=(0, 0, 0), font=font)
        del draw

        return image


yolo = mAP_Yolo()
# image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
# image_ids = open(DIR + 'VOC2007/ImageSets/Main/test.txt').read().strip().split()
image_ids = os.listdir(DIR + "VOC2007/JPEGImages/")
print(len(image_ids))
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")
time_start = time.time()
for image_id in image_ids:

    # print(DIR,"VOC2007/JPEGImages/",image_id)
    # if not os.path.exists(DIR + "VOC2007/JPEGImages/" + image_id + ".jpg"):
    #     print(image_id,"no image")
    #     continue
    # if os.path.exists("./input/images-optional/" + image_id + ".jpg"):
    #     print(image_id, "has been detected")
    #     continue


    # if not os.path.exists(DIR + "VOC2007/JPEGImages/" + image_id + ".JPG"):
    #     print(image_id,"no image")
    #     continue
    # if os.path.exists("./input/images-optional/" + image_id + ".JPG"):
    #     print(image_id, "has been detected")
    #     continue

    print(image_id, " start!")
    image_path = DIR + "VOC2007/JPEGImages/" + image_id
    image = Image.open(image_path)
    # i_H = image.width
    # i_W = image.height

    # 开启后在之后计算mAP可以可视化
    # image.save("./input/images-optional/"+image_id+".jpg")
    img = yolo.detect_image(image_id, image)
    # plt.imshow(img)
    # plt.show()
    # print(image_id, " done!")
    # img.save("./input/images-optional/" + image_id + ".JPG")
    # print('save!')
time_end = time.time()
time_c= time_end - time_start
print('time cost', time_c, 's')
print(len(image_ids))
print("time/image",time_c/len(image_ids))
print("CPU   Conversion completed!")
