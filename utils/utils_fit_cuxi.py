import torch, tqdm
import numpy as np
from copy import deepcopy
from .utils_aug import mixup_data, mixup_criterion
from .utils import Train_Metrice

def fitting(model ,coarse_classifier_loss, fine_classifier_loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler, show_thing, opt):
    model.train()
    metrice = Train_Metrice(CLASS_NUM)
    # coarse_classifier_loss =torch.nn.BCELoss()
    # fine_classifier_loss =  torch.nn.CrossEntropyLoss()
    for x, y in tqdm.tqdm(train_dataset, desc='{} Train Stage'.format(show_thing)):
        x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
                    
        coarse_classifier_pred ,fine_classifier_pred = model(x)

        y_coarse=y[:,3:]
        
        predict_y=torch.cat((fine_classifier_pred,coarse_classifier_pred),dim=1)

        loss1=coarse_classifier_loss(coarse_classifier_pred,y_coarse)
        # fine_classifier_loss(fine_classifier_pred,y_fine)
        # # l = loss(pred, y)
        if torch.any(coarse_classifier_pred > 0.5):
            
            predict_coarse_mask =predict_y[:, -1] > 0.5   # n*4的bool
            fine_classifier_= predict_y[predict_coarse_mask]
            fine_classifier_pred_ = fine_classifier_[:,0:3]
            
            # label_coarse_mask= y_coarse[coarse_classifier_pred>0.5]
         
            # y_fine=y_fine[label_coarse_mask]
            label_fine=y[predict_coarse_mask]
            label_fine = label_fine[:,0:3]
            y_fine=torch.argmax(label_fine[:,0:3],1)

            loss2=fine_classifier_loss(fine_classifier_pred_, y_fine)

            l= loss1 + loss2

        else :
            l = loss1
        metrice.update_loss(float(l.data))

        # pred = torch.zeros(size=(16,3),dtype=torch.float)
        # pred[:,3:]=coarse_classifier_pred
        # pred[:,0:3]=fine_classifier_pred

        y= torch.argmax( y , 1)
        metrice.update_y(y, predict_y)
        
        scaler.scale(l).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
   
    model_eval = model.eval()
    with torch.inference_mode():
        for x, y in tqdm.tqdm(test_dataset, desc='{} Test Stage'.format(show_thing)):
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()

            # with torch.cuda.amp.autocast(opt.amp):
            #     if opt.test_tta:
            #         bs, ncrops, c, h, w = x.size()
            #         pred = model_eval(x.view(-1, c, h, w))
            #         pred = pred.view(bs, ncrops, -1).mean(1)
            #         l = loss(pred, y)
            #     else:
            coarse_classifier_pred ,fine_classifier_pred = model_eval(x)
            
            y_coarse=y[:,3:]
            loss1=coarse_classifier_loss(coarse_classifier_pred,y_coarse)

            predict_y=torch.cat((fine_classifier_pred,coarse_classifier_pred),dim=1)

            if torch.any(coarse_classifier_pred > 0.5):
                
                predict_coarse_mask =predict_y[:, -1] > 0.5   # n*4的bool
                fine_classifier_= predict_y[predict_coarse_mask]
                fine_classifier_pred = fine_classifier_[:,0:3]
                
                label_fine=y[predict_coarse_mask]
                label_fine = label_fine[:,0:3]
                y_fine=torch.argmax(label_fine[:,0:3],1)

                loss2=fine_classifier_loss(fine_classifier_pred, y_fine)
                l= loss1 + loss2

            else :
                l = loss1

            # pred = torch.zeros(size=(16,3),dtype=torch.float)
            # pred[:,3:]=coarse_classifier_pred
            # pred[:,0:3]=fine_classifier_pred
        
            metrice.update_loss(float(l.data), isTest=True)
            y= torch.argmax( y , 1)
            metrice.update_y(y, predict_y, isTest=True)

    return metrice.get()


