import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module



class MBatt(nn.Module):
    def __init__(self, in_channels, out_channels, num_head = 2):
        super(MBatt, self).__init__()
        self.num_head = num_head
        for i in range(int(self.num_head)):
            setattr(self, "cat_head%d" % (i), AttHead(in_channels, out_channels))
        self.ela = ELA(in_channels=in_channels, phi='T')
        
    def forward(self, x):
        x = self.ela(x)
        heads = []

        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))

        y = heads[0]

        for i in range(1, self.num_head):
            y = torch.max(y, heads[i])

        y = x * y
        
        return y

    

import torch
import torch.nn as nn

class AttHead(nn.Module):
    def __init__(self, inp, oup, ca_reduction=32, se_reduction=16):
        super(AttHead, self).__init__()

        # CoordAtt in input space
        self.coord_att = CoordAtt(inp, inp, reduction=ca_reduction)

        # Optional projection if inp != oup
        if inp != oup:
            self.proj = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.proj = nn.Identity()

        # SE MLP
        self.fc = nn.Sequential(
            nn.Linear(oup, oup // se_reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(oup // se_reduction, oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CoordAtt
        z = self.coord_att(x)      # (B, inp, H, W)
        z = self.proj(z)           # (B, oup, H, W)

        # ---- deterministic global average pooling instead of AdaptiveAvgPool2d(1) ----
        b, c, _, _ = z.size()
        y = z.mean(dim=(2, 3))         # (B, C)
        y = self.fc(y).view(b, c, 1, 1)

        return z * y


    

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # ---- deterministic coordinate pooling ----
        # Original: pool_h = AdaptiveAvgPool2d((None, 1))  -> (N, C, H, 1)
        x_h = x.mean(dim=3, keepdim=True)

        # Original: pool_w = AdaptiveAvgPool2d((1, None)) -> (N, C, 1, W), then permute to (N, C, W, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)      # (N, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out



class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


class ELA(nn.Module):
    def __init__(self, in_channels, phi):
        super(ELA, self).__init__()
        '''
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        '''
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = Kernel_size // 2
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, 
                              padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        x_h = self.con1(x_h)  # [b,c,h]
        x_w = self.con1(x_w)  # [b,c,w]
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        return x_h * x_w * input
