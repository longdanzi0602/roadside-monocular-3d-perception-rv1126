#!/usr/bin/env python3
"""
RTM3D Fast + ResNet-18 backbone.
Same optimized neck+heads as model_rtm3d_fast.py but with ResNet-18.

Architecture:
  Input 512x1696
  -> ResNet-18 (512ch @ H/32xW/32)
  -> deconv x3 (128ch, stride 2) -> H/4xW/4
  -> Shared stem (128->64, 3x3)
  -> 9x Head (64->Cout, 1x1)

Backbone weights transfer from V5 ResNet-18 checkpoint.
"""
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

# ================================================================
# ResNet-18 backbone
# ================================================================

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.out_channels = 512

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion, momentum=BN_MOMENTUM),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x


# ================================================================
# RTM3D Fast + ResNet-18
# ================================================================

class PoseFastR18(nn.Module):
    def __init__(self, heads):
        self.heads = heads
        super().__init__()

        # Backbone
        self.backbone = ResNet18Backbone()  # 512ch output

        # Neck: 512->128->128->128
        self.neck = self._make_neck(512, [256, 256, 256], [4, 4, 4])

        # Shared stem: 256->64
        self.stem = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # 1x1 heads
        for head in sorted(self.heads):
            self.__setattr__(head, nn.Conv2d(64, self.heads[head], 1))

    def _make_neck(self, in_ch, num_filters, num_kernels):
        layers = []
        self.inplanes = in_ch
        for i in range(len(num_filters)):
            k, planes = num_kernels[i], num_filters[i]
            layers.append(nn.ConvTranspose2d(self.inplanes, planes, k, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.stem(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, pretrained_backbone=True, v5_weight_path=None):
        if v5_weight_path and pretrained_backbone:
            self._load_v5_backbone(v5_weight_path)
        elif pretrained_backbone:
            self._load_imagenet_backbone()
        self._init_neck_and_heads()

    def _load_imagenet_backbone(self):
        try:
            url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
            state = model_zoo.load_url(url, progress=True)
            bb_state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            missing, _ = self.backbone.load_state_dict(bb_state, strict=False)
            print(f'=> Backbone: ImageNet ResNet-18 ({len(bb_state)-len(missing)} layers)')
        except Exception as e:
            print(f'=> Warning: ImageNet load failed ({e})')

    def _load_v5_backbone(self, v5_path):
        import os
        if not os.path.exists(v5_path):
            print('=> V5 not found, using ImageNet')
            self._load_imagenet_backbone()
            return
        ckpt = torch.load(v5_path, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        # V5 keys: conv1.weight, layer1.0.conv1.weight, ...
        # Our keys: backbone.conv1.weight, backbone.layer1.0.conv1.weight, ...
        our_state = self.state_dict()
        loaded = 0
        for our_key in our_state:
            if our_key.startswith('backbone.'):
                v5_key = our_key[9:]  # strip 'backbone.' prefix
                if v5_key in sd and our_state[our_key].shape == sd[v5_key].shape:
                    our_state[our_key] = sd[v5_key]; loaded += 1
        self.load_state_dict(our_state, strict=False)
        print(f'=> Backbone from V5: {loaded} layers loaded')

    def _init_neck_and_heads(self):
        for m in self.neck.modules():
            if isinstance(m, nn.ConvTranspose2d): nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv2d) and m.bias is not None: nn.init.constant_(m.bias, 0)
        for head in self.heads:
            m = self.__getattr__(head)
            if 'hm' in head: nn.init.constant_(m.bias, -2.19)
            elif hasattr(m, 'weight'): nn.init.normal_(m.weight, std=0.001)


def create_rtm3d_fast_r18(heads, pretrained_backbone=True, v5_weight_path=None):
    model = PoseFastR18(heads)
    model.init_weights(pretrained_backbone=pretrained_backbone, v5_weight_path=v5_weight_path)
    return model


if __name__ == '__main__':
    heads = {'hm':3,'wh':2,'hps':18,'rot':8,'dim':3,'prob':1,'reg':2,'hm_hp':9,'hp_offset':2}
    v5 = '/home/srtp_2025/zxh/qianrushi/rv1126b_rtm3d_proj/exp_v5_512x1696_d8/model_best.pth'
    model = create_rtm3d_fast_r18(heads, pretrained_backbone=True, v5_weight_path=v5)
    n = sum(p.numel() for p in model.parameters())
    print(f'Params: {n:,} ({n/1e6:.1f}M)')
    x = torch.randn(1, 3, 512, 1696)
    with torch.no_grad():
        out = model(x)[0]
    for k, v in out.items():
        print(f'  {k}: {v.shape}')
