import warnings, sys, datetime, random
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, torch, argparse, time, torchvision, tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from utils import utils_aug
from utils.utils import classification_metrice, Metrice_Dataset, visual_predictions, visual_tsne, dict_to_PrettyTable, Model_Inference, select_device, model_fuse
from torchvision import transforms, models
from utils.utils_model import select_model


from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt


torch.backends.cudnn.deterministic = True
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_path', type=str, default=r'dataset/train', help='train data path')
    # parser.add_argument('--val_path', type=str, default=r'dataset/val', help='val data path')
    parser.add_argument('--test_path', type=str, default=r'D:\Desktop\3_oringin3064_croped_seleted\process_dataset\split\test', help='test data path')
    parser.add_argument('--label_path', type=str, default=r".\Datasets\label_3064aug.txt", help='label path')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--task', type=str, choices=['train', 'val', 'test', 'fps'], default='test', help='train, val, test, fps')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save result')
    parser.add_argument('--test_tta', action="store_true", help='using TTA Tricks')
    parser.add_argument('--visual',default=True, action="store_true", help='visual dataset identification')
    parser.add_argument('--tsne',default=False, action="store_true", help='visual tsne')
    parser.add_argument('--half', action="store_true", help='use FP16 half-precision inference')
    parser.add_argument('--model_type', type=str, choices=['torch', 'torchscript', 'onnx', 'tensorrt'], default='torch', help='model type(default: torch)')
    parser.add_argument('--model_name', type=str, default='resnet50', help='model_name')
    opt = parser.parse_known_args()[0]
    


    parser.add_argument('--model', type=str, default='efficientnet_b0_own', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--model_dirname', type=str, default='efficientnet_b0_fc(pre)_ECA_GBAM_GAM_3064增强大数据集(laeblsoooth0.1)', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #tag：修改存放模型的文件夹名字，即model_dirname
    opt = parser.parse_known_args()[0]
    
    train_model_save_path=opt.save_path+'/train/'+ opt.model_dirname
    model_path=opt.save_path+'\\train\\'+ opt.model_dirname

    if not os.path.exists(os.path.join(train_model_save_path, 'best.pt')):
        raise Exception('best.pt not found. please check your --save_path folder')
    ckpt = torch.load(os.path.join(train_model_save_path, 'best.pt'))
    
    
    train_opt = ckpt['opt']

    model = select_model(opt.model, 3)
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=True)

    model_fuse(model)
    # model = (model.half() if opt.half else model)
    DEVICE = select_device(opt.device, opt.batch_size)

    model.to(DEVICE)
    model.eval()
    set_seed(train_opt.random_seed)
    # model = Model_Inference(DEVICE, opt)

    # print('found checkpoint from {}, model type:{}\n{}'.format(model_path, ckpt['model'].name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))
    print('found checkpoint from {}, model type:{}\n{}'.format(model_path, opt.model_name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))

    test_transform = utils_aug.get_dataprocessing_teststage(train_opt, opt, torch.load(os.path.join(model_path, 'preprocess.transforms')))


    save_test_path=model_path.replace('train','test')
    # save_path = os.path.join(save_test_path, datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S'))
    save_path=save_test_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    CLASS_NUM = len(os.listdir(eval('opt.{}_path'.format(opt.task))))
    print(eval('opt.{}_path'.format(opt.task)))
    test_dataset = Metrice_Dataset(torchvision.datasets.ImageFolder(eval('opt.{}_path'.format(opt.task)), transform=test_transform))
    test_dataset = torch.utils.data.DataLoader(test_dataset, opt.batch_size, shuffle=False,
                                                num_workers=(0 if opt.test_tta else opt.workers))

    try:
        with open(opt.label_path, encoding='utf-8') as f:
            label = list(map(lambda x: x.strip(), f.readlines()))
    except Exception as e:
        with open(opt.label_path, encoding='gbk') as f:
            label = list(map(lambda x: x.strip(), f.readlines()))

    return opt, model, test_dataset, DEVICE, CLASS_NUM, label, save_path


if __name__ == '__main__':
    opt, model, test_dataset, DEVICE, CLASS_NUM, label, save_path = parse_opt()
    y_true, y_pred, y_score, y_feature, img_path = [], [], [], [], []
    wrong_img_path=[]
    with torch.inference_mode():
        for x, y, path in tqdm.tqdm(test_dataset, desc='Test Stage'):
            x = (x.half().to(DEVICE) if opt.half else x.to(DEVICE))
            if opt.test_tta:
                bs, ncrops, c, h, w = x.size()
                pred = model(x.view(-1, c, h, w))
                pred = pred.view(bs, ncrops, -1).mean(1)

                if opt.tsne:
                    pred_feature = model.forward_features(x.view(-1, c, h, w))
                    pred_feature = pred_feature.view(bs, ncrops, -1).mean(1)
            else:
                pred = model(x)

                if opt.tsne:
                    pred_feature = model.forward_features(x)
            try:
                pred = torch.softmax(pred, 1)
            except:
                pred = torch.softmax(torch.from_numpy(pred), 1) # using torch.softmax will faster than numpy

            y_true.extend(list(y.cpu().detach().numpy()))
            y_pred.extend(list(pred.argmax(-1).cpu().detach().numpy()))
            y_score.extend(list(pred.max(-1)[0].cpu().detach().numpy()))
            img_path.extend(list(path))
            if int(y.cpu().detach().numpy())!= int(pred.argmax(-1).cpu().detach().numpy()):
                wrong_img_path.append(path)
            if opt.tsne:
                y_feature.extend(list(pred_feature.cpu().detach().numpy()))
    
    ####混淆矩阵
    LABELS = [ 'Glioma','Meningioma','Pitutary']
    predict=y_pred
    result=y_true
    print(len(predict),len(result))
    arr = confusion_matrix(result,predict )
    df_cm = pd.DataFrame(arr, LABELS, LABELS)
    plt.figure(figsize = (9,9))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    # print(predict_file_wrong)
    plt.show()

    print(wrong_img_path)
    classification_metrice(np.array(y_true), np.array(y_pred), CLASS_NUM, label, save_path)
    
    if opt.visual:
        visual_predictions(np.array(y_true), np.array(y_pred), np.array(y_score), np.array(img_path), label, save_path)
    if opt.tsne:
        visual_tsne(np.array(y_feature), np.array(y_pred), np.array(img_path), label, save_path)