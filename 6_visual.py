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
    #meningioma_tumor   glioma_tumor  pituitary_tumor

    parser.add_argument('--source', type=str, default=r'D:\Desktop\3_oringin3064_croped_seleted\split\test\pituitary_tumor', help='source data path(file, folder)')
    parser.add_argument('--label_path', type=str, default=r"D:\Desktop\pytorch-classifier-master\pytorch_classifier_master\Datasets\label_3064aug.txt", help='label path')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--test_tta', action="store_true", help='using TTA Tricks')
    parser.add_argument('--cam_visual', default=True,action="store_true", help='visual cam')
    parser.add_argument('--cam_type', type=str, choices=['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad'], default='GradCAM', help='cam type')
    parser.add_argument('--half', default=False, action="store_true", help='use FP16 half-precision inference')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--model', type=str, default='efficientnet_b0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--model_dirname', type=str, default='efficientnet_b0_3064增强大数据集(laeblsoooth0.1)1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #tag：修改存放模型的文件夹名字，即model_dirname
    opt = parser.parse_known_args()[0]
    
    train_model_save_path=opt.save_path+'/train/'+ opt.model_dirname
    if not os.path.exists(os.path.join(train_model_save_path, '96.pt')):
        raise Exception('best.pt not found. please check your --save_path folder')
    ckpt = torch.load(os.path.join(train_model_save_path, '96.pt'))
    
    
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
    # model_fuse(model)

    # # model = select_model(ckpt['model'].name, CLASS_NUM)
    # model.load_state_dict(ckpt['model'], strict=False)
    model.to(DEVICE)


    # model = (model.half() if opt.half else model)

    # model = models.efficientnet_b0()
    # model.fc =torch. nn.Sequential(
    #         torch.nn.Dropout(0.2),
    #         torch.nn.Linear(model.fc.in_features, 4)
    #         )
    # ckpt = torch.load(os.path.join(train_model_save_path, 'best.pt'))
    # model.load_state_dict(ckpt['model'].state_dict(), strict=True)
    train_opt = ckpt['opt']
    set_seed(train_opt.random_seed)

    
    # print('found checkpoint from {}, model type:{}\n{}'.format(opt.train_model_save_path, ckpt['model'].name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))
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
        tumor_type='pituitary_tumor'+'efficientnet_b0'
        save_path = os.path.join(opt.save_path, 'predict', datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')+tumor_type)
        os.makedirs(os.path.join(save_path))
        result = []
        for file in tqdm.tqdm(os.listdir(opt.source)[111:131]):
            pred, pred_result = predict_single_image(os.path.join(opt.source, file), model, test_transform, DEVICE, half=opt.half)
            result.append('{},{},{}'.format(os.path.join(opt.source, file), label[pred], pred_result[pred]))
            
            plt.figure(figsize=(6, 6))
            if opt.cam_visual:
                cam_output = cam_model(os.path.join(opt.source, file))
                print(os.path.join(opt.source, file))
                cam_output=cam_output.rotate(270)
                plt.imshow(cam_output)
                # plt.show()    #加它延时停留
            else:
                plt.imshow(plt.imread(os.path.join(opt.source, file)))
            plt.axis('off')
            plt.title('predict label: {}\npredict probability: {:.4f}'.format(label[pred], float(pred_result[pred])),fontdict={
                                # 'family': 'serif',
                                'family': 'Times New Roman',
                                'color': 'black',
                                'weight': 'bold',
                                'size': 16     
                            })
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, file))
            plt.clf()
            plt.close()
        
        with open(os.path.join(save_path, 'result.csv'), 'w+') as f:
            f.write('img_path,pred_class,pred_class_probability\n')
            f.write('\n'.join(result))
    # elif os.path.isfile(opt.source):
    #     pred, pred_result = predict_single_image(opt.source, model, test_transform, DEVICE, half=opt.half)
        
    #     plt.figure(figsize=(6, 6))
    #     if opt.cam_visual:
    #         cam_output = cam_model(opt.source, pred)
    #         plt.imshow(cam_output)
    #     else:
    #         plt.imshow(plt.imread(opt.source))
    #     plt.axis('off')
    #     plt.title('predict label:{}\npredict probability:{:.4f}'.format(label[pred], float(pred_result[pred])),fontdict={
    #                             'family': 'serif',
    #                             # 'family': 'simsun',
    #                             'color': 'darkblue',
    #                             'weight': 'bold',
    #                             'size': 16     
    #                         })
    #     plt.tight_layout()
    #     plt.show()