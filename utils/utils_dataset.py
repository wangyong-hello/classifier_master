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
    def __init__(self,  data_list_file, input_shape,transform):
       
        self.input_shape = input_shape
        with open(data_list_file, 'r') as fd:
            lines = fd.readlines()
        image_list = [line.strip() for line in lines]
        self.image_list = np.random.permutation(image_list)
        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                          std=[0.5, 0.5, 0.5])
        self.transform = transform 
        # if self.phase == 'train':
        #     self.transforms = transform                   
           
        # else:
        #     self.transforms = T.Compose([
        #         T.ToPILImage(),
        #         T.ToTensor(),
        #         normalize 
        #     ])
    def __getitem__(self, index):
   
        sample = self.image_list[index]
        split = sample.split(' ')
        img_path = split[0]
        # print("===> img_path: ",img_path)  
        image = cv2.imread(img_path)    #tag:默认读取的图片是rgb三通道的
        # image =cv2.resize(image,dsize=(256,256))
        # image = Image.fromarray(image)

        # image = Image.open(img_path)     
        image = Image.fromarray(image)
        image=image.resize((self.input_shape,self.input_shape))
        image = self.transform(image)
        # img = self.transform(data)

        label = split[1:]
        label = np.int32(label)   ###不能返回label是one hot类型 ，不支持这样操作
        label = torch.from_numpy(label)   ###不能返回label是one hot类型 ，不支持这样操作
        label_inex=np.argmax(label)
       

        return image,label_inex
        # return image,label
    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':

    dataset = Dataset(root='/home/wlzhang/Downloads/DVR_recongize/DVR_recognize/Datasets/datasets_DVR',
                      data_list_file='/home/wlzhang/Downloads/DVR_recongize/DVR_recognize/Datasets/labels_DVR.CSV',
                      phase='train',
                      input_shape=(1, 128, 128))