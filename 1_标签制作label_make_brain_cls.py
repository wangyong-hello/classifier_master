#将分类后的图片做成标签
import os
import numpy as np
from tqdm import tqdm

def collect_file(root,dirs):
    full_path = []
    for dir in dirs:
        file_path=os.path.join(root,dir)
        for image in tqdm(os.listdir(file_path)):
            image_path=os.path.join(file_path,image)
            full_path.append(image_path)
    print(len(full_path))
    img_name = [file_path.split('\\')[-1] for file_path in full_path]
    # print(len(img_name))
    img_name_list = list(set(img_name))
    print(len(img_name_list))
    full_path_new = []
    for file in img_name_list:
        # index = np.where(img_name == file)
        # print(file)
        index = img_name.index(file)
        # print(img_name[index])

        img_name_new = full_path[index]
        # print(img_name_new)
        # print('++++++')
        full_path_new.append(img_name_new)
    print(len(full_path_new))
    return full_path_new


def make_label(root,dirs,full_path_new,save_file):

    with open(save_file,'w') as wf:
        glioma_tumor_file_path = os.listdir(os.path.join(root,dirs[0]))
        glioma_tumor_file_path_name = [file_path.split('\\')[-1] for file_path in glioma_tumor_file_path]

        meningioma_tumor_file_path = os.listdir(os.path.join(root,dirs[1]))
        meningioma_tumor_file_name = [file_path.split('\\')[-1] for file_path in meningioma_tumor_file_path]

        # no_tumor_file_path = os.listdir(os.path.join(root,'notumor'))
        # no_tumor_file_name = [file_path.split('\\')[-1] for file_path in no_tumor_file_path]

        pituitary_tumor_file_path = os.listdir(os.path.join(root,dirs[2]))
        pituitary_tumor_file_name = [file_path.split('\\')[-1] for file_path in pituitary_tumor_file_path]


        for file_path in full_path_new:
            glioma_tumor=0
            meningioma_tumor=0
            no_tumor=0
            pituitary_tumor=0
            file_name = file_path.split('\\')[-1]
            if file_name in glioma_tumor_file_path_name:
                glioma_tumor = 1
            if file_name in meningioma_tumor_file_name:
                meningioma_tumor=1      
            # if file_name in no_tumor_file_path:
            #     no_tumor = 1  
            if file_name in pituitary_tumor_file_name:
                pituitary_tumor = 1 

            # wf.write(file_path + ' ' + str(glioma_tumor)+ ' ' + str(meningioma_tumor)+ ' ' + str(no_tumor)+ ' ' + str(pituitary_tumor) + '\n')
            wf.write(file_path + ' ' + str(glioma_tumor)+ ' ' + str(meningioma_tumor)+ ' '  + str(pituitary_tumor) + '\n')
            
            # with open('D:\\test\\label.txt','a') as f:
            #     f.write(image_path,'\t',+str("label_eye"),'\t','+slabel_open','\t',+str('label_definition),'\t',+'label_no_light')
    

if __name__ == '__main__':
    root = r'C:\Users\14833\Desktop\process_dataset\split\val'
    # dirs = ['glioma','meningioma','notumor','pituitary']
    dirs = ['glioma_tumor','meningioma_tumor','pituitary_tumor']
    save_file = './Datasets/label_3064_aug_val.txt'
    #save_root = r'D:\workspace\datasets\\Eye_status\\open_eye_video_set'
    #label_image(root,dirs)
    full_path_new = collect_file(root,dirs)  
    make_label(root,dirs,full_path_new,save_file)

