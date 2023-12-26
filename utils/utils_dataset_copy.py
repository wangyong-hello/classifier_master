import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2,random
import sys
import math
import warnings
warnings.filterwarnings('ignore')

class Dataset(data.Dataset):
    def __init__(self,  data_list_file, input_shape,transform=None):
       
        self.input_shape = input_shape
        with open(data_list_file, 'r') as fd:
            lines = fd.readlines()
        image_label_list = [line.strip() for line in lines]
        self.image_label_list = np.random.permutation(image_label_list)
        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                          std=[0.5, 0.5, 0.5])
        self.transform = T.Compose(
            [
        T.ToTensor()])

        # Transformation for converting original image array to an image, rotate it randomly between -45 degrees and 45 degrees, and then convert it to a tensor
        self.transform1 = T.Compose([
            T.ToPILImage(),                                          
            T.RandomRotation(45),
            T.ToTensor()                                 
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -90 degrees and 90 degrees, and then convert it to a tensor
        self.transform2 = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(90),
            T.ToTensor()                                  
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -120 degrees and 120 degrees, and then convert it to a tensor
        self.transform3 = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(120),
            T.ToTensor()                                  
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -180 degrees and 180 degrees, and then convert it to a tensor
        self.transform4 = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(180),
            T.ToTensor()                                
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -270 degrees and 270 degrees, and then convert it to a tensor
        self.transform5 = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(270),
            T.ToTensor()                                
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -300 degrees and 300 degrees, and then convert it to a tensor
        self.transform6 = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(300),
            T.ToTensor()                               
        ])

        # Transformation for converting original image array to an image, rotate it randomly between -330 degrees and 330 degrees, and then convert it to a tensor
        self.transform7 = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(330),
            T.ToTensor()                                 
        ])
            # if self.phase == 'train':
            #     self.T = transform                   
            
            # else:
            #     self.T = T.Compose([
            #         T.ToPILImage(),
            #         T.ToTensor(),
            #         normalize 
            #     ])
    def __getitem__(self, index):
   
        sample = self.image_label_list[index]
        split = sample.split(' ')
        img_path = split[0]
        # print("===> img_path: ",img_path)  
        image = cv2.imread(img_path)    #tag:默认读取的图片是rgb三通道的
        image =cv2.resize(image,dsize=(self.input_shape,self.input_shape))
        # image = Image.fromarray(image)

        # image = Image.open(img_path)   #tag:读取的图片是单通道的就是单通道的
        # image = image.convert("RGB")
        # image=image.resize((self.input_shape,self.input_shape))
        
        image = self.transform(image)
        # img = self.transform(data)
        # Augmented image at 45 degrees as a tensor
        image45 = self.transform1(image)

        # Augmented image at 90 degrees as a tensor
        image90 = self.transform2(image)

        # Augmented image at 120 degrees as a tensor
        image120 = self.transform3(image)

        # Augmented image at 180 degrees as a tensor
        image180 = self.transform4(image)

        # Augmented image at 270 degrees as a tensor
        image270 = self.transform5(image)

        # Augmented image at 300 degrees as a tensor
        image300 = self.transform6(image)

        # imagemented image at 330 degrees as a tensor
        image330 = self.transform7(image)      
        
        # store the transformed images in a list
        new_batch = [image, image45, image90, image120, image180, image270, image300, image330]
       

        label = split[1:]
        label = np.int32(label)   ###不能返回label是one hot类型 ，不支持这样操作
        label = torch.from_numpy(label)   ###不能返回label是one hot类型 ，不支持这样操作
        label_inex=np.argmax(label)
        # label = torch.zeros(4, dtype=torch.float32)
        # label[int(label_inex)] = 1.0
        # new_labels = [label, label, label, label, label, label, label, label]
        # new_labels = [label_inex, label_inex, label_inex, label_inex, label_inex, label_inex, label_inex, label_inex]

        
        # return  (torch.stack(new_batch),torch.stack(new_labels))
        # return new_batch,new_labels
        return image,label_inex
    def __len__(self):
        return len(self.image_label_list)

if __name__ == '__main__':

    dataset = Dataset(root='/home/wlzhang/Downloads/DVR_recongize/DVR_recognize/Datasets/datasets_DVR',
                      data_list_file='/home/wlzhang/Downloads/DVR_recongize/DVR_recognize/Datasets/labels_DVR.CSV',
                      phase='train',
                      input_shape=(1, 128, 128))