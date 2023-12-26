import warnings,sys
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, argparse, shutil, random, imp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torchvision, time, datetime, copy
from sklearn.utils.class_weight import compute_class_weight
from copy import deepcopy
from utils.utils_fit import fitting
from utils.utils_model import select_model
from utils import utils_aug
from utils.utils import save_model, plot_train_batch, WarmUpLR, show_config, setting_optimizer, check_batch_size, \
    plot_log, update_opt, load_weights, get_channels, dict_to_PrettyTable, select_device
# from utils.utils_distill import *
from utils.utils_loss import *
from utils.utils_dataset import Dataset
from torch.utils import data
from torchvision import transforms, models


torch.backends.cudnn.deterministic = True
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default='efficientnet_b0_own', help='model name,用来创建模型')
    parser.add_argument('--model_name', type=str, default='resnet50', help='model name,用来创建模型')
    # parser.add_argument('--model_ater_name', type=str, default='test', help='用来做保存的文件夹名')
    parser.add_argument('--model_ater_name', type=str, default='resnet50_3064增强大数据集(laeblsoooth0.1)1', help='用来做保存的文件夹名')
    parser.add_argument('--resume',default=False, action="store_true", help='resume from save_path traning')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (-1 for autobatch)')
    

    parser.add_argument('--train_list', type=str, default=r'D:\Desktop\pytorch-classifier-master\pytorch_classifier_master\Datasets\label_3064_aug_train.txt', help='train data path')
    parser.add_argument('--val_list', type=str, default=r'D:\Desktop\pytorch-classifier-master\pytorch_classifier_master\Datasets\label_3064_aug_val.txt', help='val data root')
    parser.add_argument('--train_path', type=str, default=r'D:\Desktop\3_oringin3064_croped_seleted\split\aug_train', help='train data path，用来计算mean，std')   
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--save_path', type=str, default=r'./runs/exp', help='save path for model and log')


    parser.add_argument('--class_num', type=int, default=3, help='label path')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--self_model',default=False, action="store_true", help='self design model or not ')#tag:
    parser.add_argument('--image_channel', type=int, default=3, help='image channel')
    parser.add_argument('--pretrained', default=True ,action="store_true", help='using pretrain weight')
    parser.add_argument('--weight', type=str, default='', help='loading weight path')
    parser.add_argument('--config', type=str, default=r'.\config\config.py', help='config path')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # optimizer parameters
    parser.add_argument('--loss', type=str,choices=['PolyLoss', 'CrossEntropyLoss', 'FocalLoss'],
                        default='CrossEntropyLoss', help='loss function')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW', 'RMSProp'], default='SGD', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--class_balance', action="store_true", help='using class balance in loss')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum in optimizer')
    parser.add_argument('--amp', action="store_true", help='using AMP(Automatic Mixed Precision)')
    parser.add_argument('--warmup', action="store_true", help='using WarmUp LR')
    parser.add_argument('--warmup_ratios', type=float, default=0.05,
                        help='warmup_epochs = int(warmup_ratios * epoch) if warmup=True')
    parser.add_argument('--warmup_minlr', type=float, default=1e-6,
                        help='minimum lr in warmup(also as minimum lr in training)')
    parser.add_argument('--metrice', type=str, choices=['loss', 'acc', 'mean_acc'], default='mean_acc', help='best.pt save relu')
    parser.add_argument('--patience', type=int, default=20, help='EarlyStopping patience (--metrice without improvement)')

    # Data Processing parameters
    parser.add_argument('--imagenet_meanstd', action="store_true", help='using ImageNet Mean and Std')
    parser.add_argument('--mixup', type=str, choices=['mixup', 'cutmix', 'none'], default='none', help='MixUp Methods')
    parser.add_argument('--Augment', type=str,
                        choices=['RandAugment', 'AutoAugment', 'TrivialAugmentWide', 'AugMix', 'none'], default='none',
                        help='Data Augment')
    parser.add_argument('--test_tta', action="store_true", help='model_ater_nameusing TTA')

    # Tricks parameters
    parser.add_argument('--rdrop', action="store_true", help='using R-Drop')

    opt = parser.parse_known_args()[0]

    opt.save_path=opt.save_path+'/train/'+  str(opt.model_ater_name)
    if opt.resume:
        opt.resume = True
        if not os.path.exists(os.path.join(opt.save_path, 'last.pt')):
            raise Exception('last.pt not found. please check your --save_path folder and --resume parameters')
        ckpt = torch.load(os.path.join(opt.save_path, 'last.pt'))
        opt = ckpt['opt']
        opt.resume = True
        print('found checkpoint from {}, model type:{}\n{}'.format(opt.save_path,opt.model_name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))
    else:
        if os.path.exists(opt.save_path):
            # shutil.rmtree(opt.save_path)
            print('{} already exist!!!!'.format(opt.save_path))
            sys.exit()
        os.makedirs(opt.save_path)
        config = imp.load_source('config', opt.config).Config()
        shutil.copy(__file__, os.path.join(opt.save_path, 'main.py'))
        shutil.copy(opt.config, os.path.join(opt.save_path, 'config.py'))
        # shutil.copy('model\efficientnetb0.py' , os.path.join(opt.save_path, 'efficientnetb0.py'))  #tag:
        
        opt = update_opt(opt, config._get_opt())

    set_seed(opt.random_seed)
    show_config(deepcopy(opt))

    # CLASS_NUM = len(os.listdir(opt.train_path))
    CLASS_NUM = opt.class_num
    DEVICE = select_device(opt.device, opt.batch_size)

    train_transform, test_transform = utils_aug.get_dataprocessing(torchvision.datasets.ImageFolder(opt.train_path),
                                                                   opt)
    train_dataset = Dataset( opt.train_list, input_shape=opt.image_size,transform=train_transform)
    test_dataset = Dataset( opt.val_list, input_shape=opt.image_size,transform=test_transform)

    if opt.resume:
        model = select_model(opt.model_name, CLASS_NUM, (opt.image_size, opt.image_size), opt.image_channel,
                                opt.save_path)
        model.load_state_dict( ckpt['model'])
        model =model.to(DEVICE).float()
    else:
        if opt.self_model:
            model = select_model(opt.model_name, CLASS_NUM, (opt.image_size, opt.image_size), opt.image_channel,
                                opt.save_path)   ###自己的模型没有用opt.pretrained该设置
        else :
            model = select_model(opt.model_name, CLASS_NUM, (opt.image_size, opt.image_size), opt.image_channel,opt.save_path,
                                opt.pretrained)
            # plot_train_batch(copy.deepcopy(train_dataset), opt)   #检查拼接的16张图片
            model = load_weights(model, opt).to(DEVICE)
            
            # state_dict=torch.load(r'D:\Desktop\pytorch-classifier-master\pytorch_classifier_master\vit_small_p16_224-15ec54c9.pth')
            # model_dict=model.state_dict()
            # state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # # 更新模型的权重
            # model_dict.update(state_dict)
            # model.load_state_dict(model_dict)

           # sub_model = torch.nn.Sequential(*list(model.children())[0:-5])  ###导入第0层到倒数第二层？
            # sub_model.load_state_dict(state_dict)

            # model = models.densenet121( pretrained=True)   #tag:模型的定义
            # model.fc =torch. nn.Sequential(
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(model.fc.in_features, CLASS_NUM)
            # )

            # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # pretrained_dict = model.state_dict()
            # # 删除不匹配的权重项
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}

            # # 更新自定义模型的权重
            # model_dict = model.state_dict()
            # model_dict.update(pretrained_dict)
            # model.load_state_dict(model_dict)
            # # set all paramters as trainable
            # for param in model.parameters():
            #     param.requires_grad = True

            # model.fc =torch. nn.Sequential(
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(model.fc.in_features, CLASS_NUM)
            # )
            # set all paramters of the model as trainable
            # for name, child in model.named_children():
            #     for name2, params in child.named_parameters():
            #         params.requires_grad = True
            # model.to(device='cuda')
    batch_size = opt.batch_size if opt.batch_size != -1 else check_batch_size(model, opt.image_size, amp=opt.amp)

    # if opt.class_balance:
    #     class_weight = np.sqrt(compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets))
    # else:
    #     class_weight = np.ones_like(np.unique(train_dataset.targets))
    # print('class weight: {}'.format(class_weight))


    trainloader = data.DataLoader(dataset=train_dataset,      
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers = opt.workers ,pin_memory=True)
    testloader = data.DataLoader(dataset=test_dataset,
                                  batch_size=max(batch_size // (10 if opt.test_tta else 1), 1),
                                  shuffle=False,
                                  num_workers=(0 if opt.test_tta else opt.workers),pin_memory=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(opt.amp if torch.cuda.is_available() else False))

    # ema = ModelEMA(model) if opt.ema else None
    optimizer = setting_optimizer(opt, model)
    lr_scheduler = WarmUpLR(optimizer, opt)
    if opt.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        loss = ckpt['loss'].to(DEVICE)
        scaler.load_state_dict(ckpt['scaler'])
        # if opt.ema:
        #     ema.ema = ckpt['ema'].to(DEVICE).float()
        #     ema.updates = ckpt['updates']
    else:
        # loss = eval(opt.loss)(label_smoothing=opt.label_smoothing,
        #                       weight=torch.from_numpy(class_weight).to(DEVICE).float())
        # loss = eval(opt.loss)(label_smoothing=opt.label_smoothing)
        loss = torch.nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
        if opt.rdrop:
            loss = RDropLoss(loss)
    return opt, model, trainloader, testloader, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, (
        ckpt['epoch'] if opt.resume else 0), (ckpt['best_metrice'] if opt.resume else None)


if __name__ == '__main__':
    opt, model, train_dataset, test_dataset, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, begin_epoch, best_metrice = parse_opt()
    
    if not opt.resume:
        save_epoch = 0
        with open(os.path.join(opt.save_path, 'train.txt'), 'w+') as f:
            f.write('time,epoch,lr,loss,acc,mean_acc,test_loss,test_acc,test_mean_acc')
    else:
        # opt.save_path=opt.save_path+'/train'
        save_epoch = torch.load(os.path.join(opt.save_path, 'last.pt'))['best_epoch']

    print('{} begin train!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    for epoch in range(begin_epoch, opt.epoch):

        if epoch > (save_epoch + opt.patience) and opt.patience != 0:
            print('No Improve from {} to {}, EarlyStopping.'.format(save_epoch + 1, epoch))
            break

        begin = time.time()
       
        metrice = fitting(model, loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler,'{}/{}'.format(epoch + 1, opt.epoch), opt)
        '''
        metrice是一个元组，第一个元素是一个字典，第二个元素是字符串
        {'train_loss': 0.5613811125757394, 'test_loss': 1.198867320186562, 'train_acc': 0.7897155360877525, 
         'test_acc': 0.645359019207937, 'train_mean_acc': 0.7828407332491699, 'test_mean_acc': 0.6296069615784327}
        '0.561381,0.789716,0.782841,1.198867,0.645359,0.629607'
        '''

        with open(os.path.join(opt.save_path, 'train.txt'), 'a+') as f:
            f.write(
                '\n{},{},{:.10f},{}'.format(datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S'),
                        epoch + 1, optimizer.param_groups[2]['lr'], metrice[1]))

        n_lr = optimizer.param_groups[2]['lr']
        lr_scheduler.step()

        if best_metrice is None:
            best_metrice = metrice[0]
        else:
            # if eval('{} {} {}'.format(metrice[0]['test_{}'.format(opt.metrice)], '<' if opt.metrice == 'loss' else '>', best_metrice['test_{}'.format(opt.metrice)])):
            if 1:    
                best_metrice = metrice[0]
                save_model(
                    # os.path.join(opt.save_path, 'best.pt'),
                    os.path.join(opt.save_path, str(epoch)+'.pt'),
                    **{
                    #  'model': deepcopy(model).to('cpu').half(),   #tag: troch<2.0
                     'model': deepcopy(model),   #tag: troch<2.0
                    #  'model': model.state_dict(),    #tag: torch2.0
                    'opt': opt,
                    'best_metrice': best_metrice,
                    }
                )
                save_epoch = epoch
        
        save_model(
            os.path.join(opt.save_path, 'last.pt'),
            **{
              # 'model': deepcopy(model).to('cpu').half(),   #tag: troch<2.0
              'model': deepcopy(model),   #tag: troch<2.0
            #    'model': model.state_dict(),    #tag: torch2.0
            #    'ema': (deepcopy(ema.ema).to('cpu').half() if opt.ema else None),
            #    'updates': (ema.updates if opt.ema else None),
               'opt': opt,
               'epoch': epoch + 1,
               'optimizer' : optimizer.state_dict(),
               'lr_scheduler': lr_scheduler.state_dict(),
               'best_metrice': best_metrice,
               'loss': deepcopy(loss).to('cpu'),
            #    'kd_loss': (deepcopy(kd_loss).to('cpu') if opt.kd else None),
               'scaler': scaler.state_dict(),
               'best_epoch': save_epoch,
            }
        )

        print(dict_to_PrettyTable(metrice[0], '{} epoch:{}/{}, best_epoch:{}, time:{:.2f}s, lr:{:.8f}'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    epoch + 1, opt.epoch, save_epoch + 1, time.time() - begin, n_lr,
                )))
    
    plot_log(opt)
