import torch
import torch.nn as nn
# from transformers import BertModel, GPT2Model
from .resnet import *
from .efficientnetv2 import *
from .densenet import *
# from torchvision.models import resnet50,efficientnet_b0


class HybridModel(nn.Module):
    def __init__(self, model1, model2):
        super(HybridModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        # self.fc = nn.Linear(model1.fc.in_features + model2.classifier[-1].in_features, 4)
        # self.fc1 = nn.Linear(2000, 1000)
        # self.fc2 = nn.Linear(1000, 4)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2000, 4),
        )
    def forward(self, inputs):
        model1_outputs = self.model1(inputs)
        model2_outputs = self.model2(inputs)

        # 可以根据需要自定义如何组合两个模型的输出
        # combined_outputs = torch.cat((model1_outputs.last_hidden_state, model2_outputs.last_hidden_state), dim=1)
        combined_outputs = torch.cat((model1_outputs, model2_outputs), dim=1)
        
        # logits = self.fc1(combined_outputs)
        # logits = self.fc2(logits)
        logits = self.fc(combined_outputs)

        return logits


def hybridModel():

    # model1 = resnet50(pretrained=True)
    model1 = densenet121(pretrained=True)
    model2 = efficientnet_b0(pretrained=True)
    # model2 = efficientnet_b0()
    # model2.classifier[-1] = nn.Linear(in_features=model2.classifier[-1].in_features, out_features=1)
    hybridModel=HybridModel(model1,model2)
    # out=hybridModel(inputs)
    # print(out.shape)
    return hybridModel