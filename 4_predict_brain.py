import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, torch, argparse, datetime, tqdm, random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from utils import utils_aug
from utils.utils import predict_single_image, cam_visual, dict_to_PrettyTable, select_device, model_fuse
from utils.utils_model import select_model

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from torchvision import transforms, models
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=r'D:\Desktop\process_dataset\split\test', help='source data path(file, folder)')
    parser.add_argument('--label_path', type=str, default=r"D:\Desktop\pytorch-classifier-master\pytorch_classifier_master\Datasets\label_3064aug.txt", help='label path')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--test_tta', action="store_true", help='using TTA Tricks')
    parser.add_argument('--cam_visual', default=True,action="store_true", help='visual cam')
    parser.add_argument('--cam_type', type=str, choices=['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad'], default='FullGrad', help='cam type')
    parser.add_argument('--half', default=False, action="store_true", help='use FP16 half-precision inference')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--model', type=str, default='resnet50', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--model_dirname', type=str, default='resnet50_3064增强大数据集(laeblsoooth0.1)', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #tag：修改存放模型的文件夹名字，即model_dirname
    opt = parser.parse_known_args()[0]
    
    train_model_save_path=opt.save_path+'/train/'+ opt.model_dirname
    if not os.path.exists(os.path.join(train_model_save_path, 'best.pt')):
        raise Exception('best.pt not found. please check your --save_path folder')
    ckpt = torch.load(os.path.join(train_model_save_path, 'best.pt'))
    
    DEVICE = select_device(opt.device)
    if opt.half and DEVICE.type == 'cpu':
        raise Exception('half inference only supported GPU.')
    if opt.half and opt.cam_visual:
        raise Exception('cam visual only supported cpu. please set device=cpu.')
    if (opt.device != 'cpu') and opt.cam_visual:
        raise Exception('cam visual only supported FP32.')
    with open(opt.label_path) as f:
        CLASS_NUM = len(f.readlines())

  
    model = select_model(opt.model, 3)
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=True)
    model_fuse(model)

    # # model = select_model(ckpt['model'].name, CLASS_NUM)
    # model.load_state_dict(ckpt['model'], strict=False)
    model.to(DEVICE)


    # model = (model.half() if opt.half else model)

    # model = models.resnet50()
    # model.fc =torch. nn.Sequential(
    #         torch.nn.Dropout(0.2),
    #         torch.nn.Linear(model.fc.in_features, 4)
    #         )
    # ckpt = torch.load(os.path.join(train_model_save_path, 'best.pt'))
    # model.load_state_dict(ckpt['model'].state_dict(), strict=True)
    train_opt = ckpt['opt']
    set_seed(train_opt.random_seed)

    print('found checkpoint from {}, model type:{}\n{}'.format(train_model_save_path, opt.model_dirname, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))
    test_transform = utils_aug.get_dataprocessing_teststage(train_opt, opt, torch.load(os.path.join(train_model_save_path, 'preprocess.transforms')))

    try:
        with open(opt.label_path, encoding='utf-8') as f:
            label = list(map(lambda x: x.strip(), f.readlines()))
    except Exception as e:
        with open(opt.label_path, encoding='gbk') as f:
            label = list(map(lambda x: x.strip(), f.readlines()))

    return opt, DEVICE, model, test_transform, label

if __name__ == '__main__':
    opt, DEVICE, model, test_transform, label = parse_opt()

    if opt.cam_visual:
        cam_model = cam_visual(model, test_transform, DEVICE, opt)

    if os.path.isdir(opt.source):
        save_path = os.path.join(opt.save_path, 'predict', datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S'))
        os.makedirs(os.path.join(save_path))
        result = []
        predict=[]
        predict_file_wrong=[]
        for dir in tqdm.tqdm(os.listdir(opt.source)[0:]):
            for file in tqdm.tqdm(os.listdir(os.path.join(opt.source,dir))[0:]):
                
                # pred, pred_result = predict_single_image(os.path.join(opt.source+'\\'+dir, file), model, test_transform, DEVICE, half=opt.half)
                # result.append('{},{},{}'.format(os.path.join(opt.source, file), label[pred], pred_result[pred]))
                image = Image.open(os.path.join(opt.source+'\\'+dir, file))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image=image.resize((256,256))
                # x = test_transform(image).unsqueeze(0).to(DEVICE)
                transform=transforms.ToTensor()
                x =transform( image).unsqueeze(0).to(DEVICE)

                pred = model(x)
                pred_result=torch.softmax(pred, 1)

                print(os.path.join(opt.source+'\\'+dir, file))
                
                predict.append(int(pred))
                
                # max_value, max_index = torch.max(pred_result, dim=0)
                # max_index=max_index.cpu().tolist()    
                # if pred!=max_index:
                #     predict_file_wrong.append(os.path.join(opt.source+'\\'+dir, file))
                label= 0  
                if 'glioma' in dir:
                    label=0
                elif 'meningioma' in dir:
                    label=1
                # elif 'notumor' in dir:
                #     result.append(int('2'))
                elif 'pituitary' in dir:
                    label=2
                result.append(label)

                if pred!=label:
                    predict_file_wrong.append(os.path.join(opt.source+'\\'+dir, file))
                
                if opt.cam_visual:
                    cam_output = cam_model(os.path.join(opt.source, file))
                    plt.imshow(cam_output)

        # LABELS = [ 'Glioma','Meningioma', 'notumor','Pitutary']
        LABELS = [ 'Glioma','Meningioma','Pitutary']
        print(len(predict),len(result))
        arr = confusion_matrix(result,predict )
        df_cm = pd.DataFrame(arr, LABELS, LABELS)
        plt.figure(figsize = (9,9))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
        plt.xlabel("Prediction")
        plt.ylabel("Target")
        # print(predict_file_wrong)
        plt.show()
                # #     re
                # meningioma
                # notumor
                # pituitary
            #     plt.figure(figsize=(6, 6))
            # if opt.cam_visual:
            #     cam_output = cam_model(os.path.join(opt.source, file))
            #     plt.imshow(cam_output)
            # else:
            #     plt.imshow(plt.imread(os.path.join(opt.source, file)))
            # plt.axis('off')
            # plt.title('predict label:{}\npredict probability:{:.4f}'.format(label[pred], float(pred_result[pred])),
            #         fontdict={
            #                     'family': 'serif',
            #                     'color': 'darkblue',
            #                     'weight': 'bold',
            #                     'size': 16
            #                 }
            #         )
            # plt.tight_layout()
            # plt.savefig(os.path.join(save_path, file),dpi=500)
            # plt.clf()
            # plt.close()
                
            #     # with open(os.path.join(save_path, 'result.csv'), 'w+') as f:
            #     #     f.write('img_path,pred_class,pred_class_probability\n')
            #     #     f.write('\n'.join(result))
    