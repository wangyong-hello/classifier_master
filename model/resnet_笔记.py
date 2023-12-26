import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
# from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
# import sys
# sys.path.append('./utils')
# from utils.utils import load_weights_from_state_dict, fuse_conv_bn

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

'''
解读：BasicBlock是为resnet18、34设计的，由于较浅层的结构可以不使用Bottleneck。
这个结构就是由两个3*3的结构为主加上bn和一次relu激活组成。
其中有个downsample是由于有x+out的操作，要保证这两个可以加起来所以对原始输入的x进行downsample。'''
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x   
# 在深度学习中，"identity"（恒等映射）通常指代一个网络层或模块，它将输入直接传递到输出，没有进行任何变换或操作。
# 这种恒等映射通常用于跳跃连接（skip connection）或残差连接（residual connection）等结构中。
        out = self.conv1(x)
        if hasattr(self, 'bn1'):
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if hasattr(self, 'bn2'):
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    #tag:
    '''
    fuse_conv_bn(conv, bn)是一个函数，用于将卷积层（Convolutional Layer）和批归一化层（Batch Normalization Layer）
    融合（fuse）为一个更高效的操作。在卷积神经网络中，批归一化层用于在卷积操作之后对特征图进行归一化处理，
    以加速训练和提高模型的性能。然而，卷积层和批归一化层的操作在计算上是可以合并的，从而减少计算量和内存占用。
    fuse_conv_bn函数接受卷积层conv和批归一化层bn作为输入，并返回一个融合后的新卷积层。
    融合后的卷积层将卷积操作和批归一化操作合并在一起，以提高计算效率。
    具体的融合操作可能因深度学习框架而异，但通常包括以下步骤：
    提取卷积层的权重参数（卷积核权重和偏置项）和批归一化层的参数（缩放因子、偏置项、均值和方差）。
    将批归一化层的缩放因子和权重参数相乘，得到新的卷积核权重。
    将批归一化层的偏置项和权重参数相乘，得到新的偏置项。
    将新的卷积核权重和偏置项应用于卷积操作。
    融合卷积层和批归一化层可以减少内存访问和计算量，并提高模型的推理速度。
    这种优化通常在模型训练之后进行，以便在推理阶段获得性能提升。
    请注意，具体的融合操作和实现方式可能因深度学习框架和版本而异，因此建议查阅相关框架的文档或源代码以了解更多细节。'''    
    # def switch_to_deploy(self):
    #     self.conv1 = fuse_conv_bn(self.conv1, self.bn1)
    #     del self.bn1
    #     self.conv2 = fuse_conv_bn(self.conv2, self.bn2)
    #     del self.bn2

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4   #tag: 隐藏层中通道数的变换系数 (模块内的通道变换系数)

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if hasattr(self, 'bn1'):
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if hasattr(self, 'bn2'):
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if hasattr(self, 'bn3'):
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    # def switch_to_deploy(self):
    #     self.conv1 = fuse_conv_bn(self.conv1, self.bn1)
    #     del self.bn1
    #     self.conv2 = fuse_conv_bn(self.conv2, self.bn2)
    #     del self.bn2
    #     self.conv3 = fuse_conv_bn(self.conv3, self.bn3)
    #     del self.bn3

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,   #tag:控制每组卷积块的输出通道数都是64的倍数
        replace_stride_with_dilation: Optional[List[bool]] = None,
# replace_stride_with_dilation函数或选项允许在ResNet中将步幅为2的卷积层替换为具有特定扩张率的空洞卷积。
# 通过增加扩张率，可以保持特征图的尺寸，并扩大卷积层的感受野，以更好地捕捉图像中的上下文信息。
# 这种替换通常在深度较大的ResNet模型中使用，例如ResNet-101或ResNet-152，以增加感受野并提升模型的性能。
# 总而言之，replace_stride_with_dilation是ResNet中的一个函数或选项，用于将步幅为2的卷积层替换为具有特定扩张率的空洞卷积。
# 这种替换可以保留特征图的尺寸，并增加卷积层的感受野，以提高模型的性能和特征提取能力。
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64   # tag:inplanes 表示输入特征图的通道数
        self.dilation = 1  # tag:dilation 是卷积操作中卷积核的采样间隔，用于调整卷积操作的感受野大小，影响特征提取的范围和精度
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)  #tag:修改输入的通道数目
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])   #tag：resnet50的layers=[3, 4, 6, 3]，该模块搭建的次数
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes) 
        #tag:block.expansion是一个表示残差块（residual block）中的通道扩展系数（channel expansion factor）的属性或变量。

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    # def switch_to_deploy(self):
    #     self.conv1 = fuse_conv_bn(self.conv1, self.bn1)   
    #     del self.bn1

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer     ###tag:_make_layer以单下划线开头定义的，impport *   时，就不会被导入
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):   #tag:blocks表示这个模块搭建重复多少次
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor, need_fea=False) -> Tensor:
        return self._forward_impl(x, need_fea)
    
    def _forward_impl(self, x: Tensor, need_fea=False) -> Tensor:  
        # See note [TorchScript super()]
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.fc(features_fc)
            return features, features_fc, x   #tag:最后模型的输出在这
        else:
            x = self.forward_features(x)
            x = self.fc(x)
            return x

    def forward_features(self, x, need_fea=False):  #tag：输出网络最后一张特征图的展平
        x = self.conv1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if need_fea:      
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

            x = self.avgpool(x4)
            x = torch.flatten(x, 1)
            return [x1, x2, x3, x4], x  
        #tag：模型最后的输出，以resnet50为例
        # feature 1 shape:torch.Size([1, 256, 56, 56])
        # feature 2 shape:torch.Size([1, 512, 28, 28])
        # feature 3 shape:torch.Size([1, 1024, 14, 14])
        # feature 4 shape:torch.Size([1, 2048, 7, 7])
        # fc shape:torch.Size([1, 2048])
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    def cam_layer(self):
        return self.layer4[-1]

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)ss
    #     load_weights_from_state_dict(model, state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

if __name__ == '__main__':
    inputs = torch.rand((1,3, 256, 256))
    model = resnet50(pretrained=True)
    model.eval()
    out = model(inputs)     #tag:这种方式输出仅最后一个拉平的向量
    print('out shape:{}'.format(out.size()))


    feas, fea_fc, out = model(inputs, need_fea=True)    #tag:返回一个feas是不同层的一个特征图列表，fea_fc是特征图拉平了，out是需要分类
    for idx, fea in enumerate(feas):
        print('feature {} shape:{}'.format(idx + 1, fea.size()))
    print('fc shape:{}'.format(fea_fc.size()))
    print('out shape:{}'.format(out.size()))




    print('=====>',model.fc)   #tag:  =====> Linear(in_features=2048, out_features=1000, bias=True)
    print('=====>',model.fc.in_features)  
    #tag:在这个模型中，model.fc 表示ResNet模型的最后一个全连接层。
    # 我们可以对这个全连接层进行进一步的操作，例如修改其输出特征数量、替换为不同的全连接层或添加其他层。