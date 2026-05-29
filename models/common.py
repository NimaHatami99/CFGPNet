from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn

import math

from utils.general import (ROOT, check_suffix, yaml_load)

from models.repvit import RepViTBlock 
from models.densenet import BottleneckBlock, DenseBlock 
from models.MultiBranchAttention import MBatt, ELA

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, ch=2):
        super().__init__()
        self.ch = ch
        # If this conv is wired to a specific stream (0/1), its effective input channels are half (6->3)
        eff_c1 = (c1 // 2) if (ch in (0, 1) and isinstance(c1, int)) else c1
        self.conv = nn.Conv2d(eff_c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def _select_input(self, x):
        # Accept tensor or (ir, vis) tuple
        if isinstance(x, (list, tuple)):
            if self.ch in (0, 1):        # explicit stream selection
                return x[self.ch]
            return torch.cat(x, dim=1)   # no selection: treat as 6-ch by concat
        return x

    def forward(self, x):
        x = self._select_input(x)
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        x = self._select_input(x)
        return self.act(self.conv(x))



class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

############################## GELAN with RepVGG baseline ############################

class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid


    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



############################################################################################################

class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)



class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)) # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)


        q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        max_attn = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_attn)
        attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
        x = (v @ attn.transpose(-2, -1))

        x = x.permute(0, 3, 1, 2)


        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class RepViTBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, use_se, attn, shuffle, area=1, mlp_ratio=2.0, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c1 % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c1 // 32

        # self.cv1 = RepConvN(c1, c_, k[0], 1) 
        self.cv1 = RepViTBlock(c1, c_, k[0], 1, use_se)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.block = ABlock(c1, num_heads, mlp_ratio, area) if attn else Shuffle(c1, c2) if shuffle else nn.Identity()
        self.add = c1 == c2

    def forward(self, x):
        return self.block(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepViTCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, use_se, attn, shuffle, n=1, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepViTBottleneck(c_, c_, use_se, attn, shuffle, g=g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


import torch.nn.functional as F
from torch.nn.modules.utils import _pair
    
    
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    def forward(self, x):    
        # Split 6-channel input into infrared/visible streams
        return x[:, :3, :, :], x[:, 3:, :, :]  # (infrared, visible)


##### GELAN #####        
        
class SPPELAN(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4*c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
        

class RepViTCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1, use_se=False, attn=False, shuffle=False,
                 use_gamma_y23=False, use_residual=False, init_gamma=1e-2):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepViTCSP(c3 // 2, c4, use_se, attn, shuffle, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepViTCSP(c4, c4, use_se, attn, shuffle, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

        # Branch gate γ for [y2, y3]
        self.gamma_y23 = nn.Parameter(init_gamma * torch.ones(2 * c4)) if use_gamma_y23 else None

        # Optional LayerScale residual on the block output (only if channel-compatible)
        self.use_residual = use_residual and (c1 == c2)
        self.gamma_out = nn.Parameter(init_gamma * torch.ones(c2)) if self.use_residual else None

    def forward(self, x):
        y0, y1 = self.cv1(x).chunk(2, 1)   # [B, c3//2, H, W] each
        y2 = self.cv2(y1)                  # [B, c4, H, W]
        y3 = self.cv3(y2)                  # [B, c4, H, W]

        if self.gamma_y23 is not None:
            y23 = torch.cat([y2, y3], 1) * self.gamma_y23.view(1, -1, 1, 1)
            feats = torch.cat([y0, y1, y23], 1)
        else:
            feats = torch.cat([y0, y1, y2, y3], 1)

        out = self.cv4(feats)

        if self.use_residual:
            out = x + self.gamma_out.view(1, -1, 1, 1) * out

        return out
    
#################

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(c_ + 3 * c1, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

##### CBNet #####

class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs

class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


class DetectMultiBackend(nn.Module):
    # YOLO MultiBackend class for python inference on various backends
    def __init__(self, weights='yolo.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, onnx_end2end, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLO MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]


# --------------------------------------------- shuffle --------------------------

class Shuffle(nn.Module):
    def __init__(self, c1, c2, groups = 4):  # ch_in, ch_out
        super().__init__()
        self.c = c2 // 4 
        self.cv11 = Conv(c1, self.c, 1, 1, 0)
        self.cv12 = Conv(self.c, self.c, 3, 1, 1, d=1)
        self.cv21 = Conv(c1, self.c, 3, 1, 1)
        self.cv22 = Conv(self.c, self.c, 3, 1, 3, d=3)
        self.cv31 = Conv(c1, self.c, 5, 1, 2)
        self.cv32 = Conv(self.c, self.c, 3, 1, 5, d=5)
        self.cv41 = Conv(c1, self.c, 7, 1, 3)
        self.cv42 = Conv(self.c, self.c, 3, 1, 7, d=7)
        self.cv = DWConv(4 * self.c, c2, 1, 1)
        self.groups = groups

    def forward(self, x):
        x1 = self.cv12(self.cv11(x))
        x2 = self.cv22(x1 + self.cv21(x))
        x3 = self.cv32(x2 + self.cv31(x))
        x4 = self.cv42(x3 + self.cv41(x))

        # Concatenate the results along the channel dimension
        out = torch.cat([x1, x2, x3, x4], 1)  # shape (batch, channels, height, width)

        out = channel_shuffle(out, self.groups)

        return self.cv(out)
    

def channel_shuffle(x, groups: int = 4):
    b, c, h, w = x.shape
    assert c % groups == 0, "channels must be divisible by groups"
    x = x.view(b, groups, c // groups, h, w)      # (B, g, C/g, H, W)
    x = x.transpose(1, 2).contiguous()            # (B, C/g, g, H, W)
    return x.view(b, c, h, w)


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

# --------------------------------------------- CrossAttn --------------------------


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)

        # removed AdaptiveAvgPool2d layers, will use .mean() instead
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                 kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        ch_g = c // g

        # b, c, h, w -> (b * g, c // g, h, w)
        group_x = x.reshape(b * g, ch_g, h, w)

        # ---- deterministic replacements for pool_h / pool_w ----
        # pool_h: AdaptiveAvgPool2d((None, 1)) -> mean over width
        x_h = group_x.mean(dim=3, keepdim=True)                  # (B*G, Cg, H, 1)

        # pool_w: AdaptiveAvgPool2d((1, None)) -> mean over height, then permute
        x_w = group_x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B*G, Cg, W, 1)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))          # (B*G, Cg, H+W, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() *
                     x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # ---- deterministic replacement for agp (AdaptiveAvgPool2d((1,1))) ----
        # agp(x1): (B*G, Cg, 1, 1) -> mean over H,W
        x1_gap = x1.mean(dim=(2, 3), keepdim=True)               # (B*G, Cg, 1, 1)
        x11 = self.softmax(x1_gap.reshape(b * g, -1, 1).permute(0, 2, 1))

        x12 = x2.reshape(b * g, ch_g, -1)                        # (B*G, Cg, H*W)

        x2_gap = x2.mean(dim=(2, 3), keepdim=True)               # (B*G, Cg, 1, 1)
        x21 = self.softmax(x2_gap.reshape(b * g, -1, 1).permute(0, 2, 1))

        x22 = x1.reshape(b * g, ch_g, -1)                        # (B*G, Cg, H*W)

        weights = (torch.matmul(x11, x12) +
                   torch.matmul(x21, x22)).reshape(b * g, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class EnhancedEMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32, kernel_size=7):
        super(EnhancedEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        pad = kernel_size // 2 
        self.conv = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=kernel_size, padding=pad, groups=channels // self.groups, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        cg = c // g
        group_x = x.reshape(b * g, cg, h, w)  # b*g,c//g,h,w 
        # mean along width → (B*, Cg, H), mean along height → (B*, Cg, W)
        x_h = group_x.mean(dim=3, keepdim=True).view(b * g, cg, h)
        x_w = group_x.mean(dim=2, keepdim=True).view(b * g, cg, w)
        # depthwise 1D conv + GN + sigmoid on both axes (same weights reused)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b * g, cg, h, 1)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b * g, cg, 1, w)
        x1 = self.gn(group_x * x_h * x_w)  # (b*g, cg, h, w)
        x2 = self.conv3x3(group_x)                                # (B*, cg, h, w)

        # channel queries from both paths (same as before but flattened)
        q1 = self.softmax(self.agp(x1).view(b * g, -1))           # (B*, cg)
        q2 = self.softmax(self.agp(x2).view(b * g, -1))           # (B*, cg)

        # axis summaries from both paths
        H1, W1 = x1.mean(dim=3), x1.mean(dim=2)                   # (B*, cg, h), (B*, cg, w)
        H2, W2 = x2.mean(dim=3), x2.mean(dim=2)                   # (B*, cg, h), (B*, cg, w)

        # project channel queries onto axis summaries → 1D attentions
        s_h = (H1 * q2.unsqueeze(-1)).sum(1) + (H2 * q1.unsqueeze(-1)).sum(1)   # (B*, h)
        s_w = (W1 * q2.unsqueeze(-1)).sum(1) + (W2 * q1.unsqueeze(-1)).sum(1)   # (B*, w)

        # normalize and form a separable 2D map via outer product
        att_h = torch.softmax(s_h / self.tau, dim=-1).unsqueeze(-1)          # (B*, h, 1)
        att_w = torch.softmax(s_w / self.tau, dim=-1).unsqueeze(1)           # (B*, 1, w)
        weights = torch.bmm(att_h, att_w).unsqueeze(1)            # (B*, 1, h, w)

        # gate & return
        out = (group_x * torch.sigmoid(weights)).view(b, c, h, w)
        return out


class DualEnhancedEMA(nn.Module):
    """
    Apply EnhancedEMA independently to infrared and visible feature maps,
    then concatenate the results along the channel dimension.

    Args:
        channels (int): Number of input channels (must be divisible by factor).
        c2 (ignored): Kept for drop-in compatibility.
        factor (int): Number of groups used in EnhancedEMA.
        kernel_size (int): Kernel size for EnhancedEMA's 1D depthwise conv.

    Inputs:
        xs: tuple/list of two tensors [infrared, visible],
            each of shape (B, C, H, W).

    Output:
        Tensor of shape (B, 2C, H, W) = cat([enhanced_ir, enhanced_vis], dim=1).
    """
    def __init__(self, channels, c2=None, factor=32, kernel_size=7):
        super().__init__()
        self.ema = EnhancedEMA(channels, c2=c2, factor=factor, kernel_size=kernel_size)
        self.groups = self.ema.groups
        assert channels // self.groups > 0, "channels must be >= groups"
        self.channels = channels

    @torch.no_grad()
    def _check_shapes(self, x_ir, x_vis):
        assert x_ir.shape == x_vis.shape, "infrared and visible tensors must have identical shapes"
        b, c, h, w = x_ir.shape
        assert c % self.groups == 0, f"channels ({c}) must be divisible by groups ({self.groups})"
        return b, c, h, w

    def forward(self, xs):
        # xs is expected as [infrared, visible]
        x_ir, x_vis = xs[0], xs[1]
        b, c, h, w = self._check_shapes(x_ir, x_vis)

        # Each modality is enhanced by its own attention weights
        # computed inside the shared EnhancedEMA module.
        enhanced_ir  = self.ema(x_ir)   # (B, C, H, W)
        enhanced_vis = self.ema(x_vis)  # (B, C, H, W)

        # Concatenate: [infrared_enhanced, visible_enhanced]
        return torch.cat([enhanced_ir, enhanced_vis], dim=1)


class DualEMA(nn.Module):
    """
    Apply EMA independently to infrared and visible feature maps,
    then concatenate the results along the channel dimension.

    Args:
        channels (int): Number of input channels (must be divisible by factor).
        c2 (ignored): Kept for drop-in compatibility.
        factor (int): Number of groups used in EMA.

    Inputs:
        xs: tuple/list of two tensors [infrared, visible],
            each of shape (B, C, H, W).

    Output:
        Tensor of shape (B, 2C, H, W) = cat([enhanced_ir, enhanced_vis], dim=1).
    """
    def __init__(self, channels, c2=None, factor=32):
        super().__init__()
        self.ema = EMA(channels, c2=c2, factor=factor)
        self.groups = self.ema.groups
        assert channels // self.groups > 0, "channels must be >= groups"
        self.channels = channels

    @torch.no_grad()
    def _check_shapes(self, x_ir, x_vis):
        assert x_ir.shape == x_vis.shape, "infrared and visible tensors must have identical shapes"
        b, c, h, w = x_ir.shape
        assert c % self.groups == 0, f"channels ({c}) must be divisible by groups ({self.groups})"
        return b, c, h, w

    def forward(self, xs):
        # xs is expected as [infrared, visible]
        x_ir, x_vis = xs[0], xs[1]
        b, c, h, w = self._check_shapes(x_ir, x_vis)

        # Each modality is enhanced by its own attention weights
        # computed inside the shared EMA module.
        enhanced_ir  = self.ema(x_ir)   # (B, C, H, W)
        enhanced_vis = self.ema(x_vis)  # (B, C, H, W)

        # Concatenate: [infrared_enhanced, visible_enhanced]
        return torch.cat([enhanced_ir, enhanced_vis], dim=1)

class DualELA(nn.Module):
    """
    Apply ELA independently to infrared and visible feature maps,
    then concatenate the results along the channel dimension.

    Inputs:
        xs: tuple/list of two tensors [infrared, visible],
            each of shape (B, C, H, W).

    Output:
        Tensor of shape (B, 2C, H, W) = cat([enhanced_ir, enhanced_vis], dim=1).
    """
    def __init__(self, in_channels: int, c2=None, phi: str = "T"):
        super().__init__()
        self.ela = ELA(in_channels=in_channels, phi=phi)  # shared parameters, per-input attention
        self.in_channels = in_channels
        self.phi = phi

        # sanity checks for GroupNorm / grouped Conv1d constraints
        gn_groups = self.ela.GN.num_groups
        conv_groups = self.ela.con1.groups
        assert in_channels % gn_groups == 0, f"in_channels ({in_channels}) must be divisible by GN groups ({gn_groups})"
        assert in_channels % conv_groups == 0, f"in_channels ({in_channels}) must be divisible by Conv1d groups ({conv_groups})"

    @torch.no_grad()
    def _check_shapes(self, x_ir: torch.Tensor, x_vis: torch.Tensor):
        assert x_ir.shape == x_vis.shape, "infrared and visible tensors must have identical shapes"
        b, c, h, w = x_ir.shape
        assert c == self.in_channels, f"expected C={self.in_channels}, got C={c}"
        return b, c, h, w

    def forward(self, xs):
        x_ir, x_vis = xs[0], xs[1]
        self._check_shapes(x_ir, x_vis)

        enhanced_ir  = self.ela(x_ir)   # (B, C, H, W)
        enhanced_vis = self.ela(x_vis)  # (B, C, H, W)

        return torch.cat([enhanced_ir, enhanced_vis], dim=1)  # (B, 2C, H, W)

class CrossAttn(nn.Module):
    """
    Cross-modal attention using EnhancedEMA's overall weights.
    - Computes EnhancedEMA weights separately for infrared and visible.
    - Cross-applies: visible is gated by infrared's weights, infrared by visible's weights.
    - Returns channel-wise concatenation [enhanced_infrared, enhanced_visible].

    Args:
        channels (int): input channels (must be divisible by factor).
        c2 (ignored): kept for drop-in compatibility.
        factor (int): number of groups used in EnhancedEMA.
        kernel_size (int): kernel size for EnhancedEMA's 1D depthwise conv.
    """
    def __init__(self, channels, c2 = None, factor=32, kernel_size=7):
        super().__init__()
        self.ema = EnhancedEMA(channels, c2 = c2, factor=factor, kernel_size=kernel_size)
        self.groups = self.ema.groups
        assert channels // self.groups > 0, "channels must be >= groups"
        self.channels = channels

    @torch.no_grad()
    def _check_shapes(self, x_ir, x_vis):
        assert x_ir.shape == x_vis.shape, "infrared and visible tensors must have identical shapes"
        b, c, h, w = x_vis.shape
        assert c % self.groups == 0, f"channels ({c}) must be divisible by groups ({self.groups})"
        return b, c, h, w

    def _compute_group_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reproduces EnhancedEMA's *weight map* (pre-gating feature multiply),
        using the same submodules so parameters are shared.
        Returns weights of shape (B*G, 1, H, W) just like inside EnhancedEMA.
        """
        b, c, h, w = x.shape
        g = self.groups
        cg = c // g
        ema = self.ema

        # group view
        group_x = x.reshape(b * g, cg, h, w)  # (B*, cg, H, W)

        # axis pooling (means)
        x_h = group_x.mean(dim=3, keepdim=True).reshape(b * g, cg, h)  # (B*, cg, H)
        x_w = group_x.mean(dim=2, keepdim=True).reshape(b * g, cg, w)  # (B*, cg, W)

        # shared depthwise conv + GN + sigmoid for both axes
        x_h = ema.sigmoid(ema.gn(ema.conv(x_h))).reshape(b * g, cg, h, 1)  # (B*, cg, H, 1)
        x_w = ema.sigmoid(ema.gn(ema.conv(x_w))).reshape(b * g, cg, 1, w)  # (B*, cg, 1, W)

        # two paths
        x1 = ema.gn(group_x * x_h * x_w)           # (B*, cg, H, W)
        x2 = ema.conv3x3(group_x)                  # (B*, cg, H, W)

        # channel queries
        q1 = ema.softmax(ema.agp(x1).reshape(b * g, -1))  # (B*, cg)
        q2 = ema.softmax(ema.agp(x2).reshape(b * g, -1))  # (B*, cg)

        # axis summaries
        H1, W1 = x1.mean(dim=3), x1.mean(dim=2)   # (B*, cg, H), (B*, cg, W)
        H2, W2 = x2.mean(dim=3), x2.mean(dim=2)   # (B*, cg, H), (B*, cg, W)

        # project channel queries onto axes
        s_h = (H1 * q2.unsqueeze(-1)).sum(1) + (H2 * q1.unsqueeze(-1)).sum(1)  # (B*, H)
        s_w = (W1 * q2.unsqueeze(-1)).sum(1) + (W2 * q1.unsqueeze(-1)).sum(1)  # (B*, W)

        # separable 2D attention and final weight map (before sigmoid gating)
        att_h = torch.softmax(s_h / ema.tau, dim=-1).unsqueeze(-1)  # (B*, H, 1)
        att_w = torch.softmax(s_w / ema.tau, dim=-1).unsqueeze(1)   # (B*, 1, W)
        weights = torch.bmm(att_h, att_w).unsqueeze(1)              # (B*, 1, H, W)

        return weights  # NOTE: not sigmoided here; gating applies sigmoid like EnhancedEMA

    def forward(self, xs):
        """
        xs: tuple/list of two tensors [infrared, visible], each (B, C, H, W)
        returns: torch.Tensor of shape (B, 2C, H, W) = cat([enhanced_ir, enhanced_vis], dim=1)
        """
        x_ir, x_vis = xs[0], xs[1]
        b, c, h, w = self._check_shapes(x_ir, x_vis)
        g = self.groups
        cg = c // g

        # compute per-modality weights using shared EnhancedEMA params
        w_ir = self._compute_group_weights(x_ir)   # (B*G, 1, H, W)
        w_vis = self._compute_group_weights(x_vis) # (B*G, 1, H, W)

        # group views for cross gating
        gi = x_ir.reshape(b * g, cg, h, w)
        gv = x_vis.reshape(b * g, cg, h, w)

        # cross-apply (use sigmoid(weights) as in EnhancedEMA)
        enhanced_vis = (gv * torch.sigmoid(w_ir)).reshape(b, c, h, w)
        enhanced_ir  = (gi * torch.sigmoid(w_vis)).reshape(b, c, h, w)

        # concat in channel dim: [infrared_enhanced, visible_enhanced]
        return torch.cat([enhanced_ir, enhanced_vis], dim=1)


# --------------------------------------------- AuxFeatFuse --------------------------

class AuxFeatFuse(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__()
        self.c_ = c1 // 2 # number of channels in each domain 
        self.c_1 = self.c_ // 2 
        self.c_2 = (self.c_ + c2) // 4 

        self.cv11 = Conv(self.c_, self.c_1, 1, 1) 
        self.cv12 = Conv(self.c_1, self.c_1, 1, 1) 
        self.cv13 = Conv(self.c_1, self.c_1, 1, 1) 
        self.cv14 = Conv(self.c_1, self.c_1, 1, 1) 
        self.cv15 = GhostConv(self.c_1, self.c_, 3, 1) 
        self.cv16 = GhostConv(self.c_1, self.c_, 5, 1) 
        self.cv17 = GhostConv(self.c_1, self.c_, 7, 1) 
        self.cbam11 = CBAM(self.c_) 
        self.cbam12 = CBAM(self.c_) 
        self.cbam13 = CBAM(self.c_) 

        self.cv21 = Conv(self.c_, self.c_1, 1, 1) 
        self.cv22 = Conv(self.c_1, self.c_1, 1, 1) 
        self.cv23 = Conv(self.c_1, self.c_1, 1, 1) 
        self.cv24 = Conv(self.c_1, self.c_1, 1, 1) 
        self.cv25 = GhostConv(self.c_1, self.c_, 3, 1) 
        self.cv26 = GhostConv(self.c_1, self.c_, 5, 1) 
        self.cv27 = GhostConv(self.c_1, self.c_, 7, 1) 
        self.cbam21 = CBAM(self.c_) 
        self.cbam22 = CBAM(self.c_) 
        self.cbam23 = CBAM(self.c_)

        self.cv1 = Conv(self.c_, self.c_2, 1, 1) 
        self.cv2 = Conv(self.c_, self.c_2, 1, 1) 
        self.cv3 = Conv(self.c_, self.c_2, 1, 1) 
        self.cv4 = Conv(3 * self.c_2, c2, 1, 1) 
        

    def forward(self, x):
        C = x.size(1) 
        x_ir, x_vis = x[:, :C // 2], x[:, C // 2:] 

        x11 = self.cv11(x_ir) 
        x12 = self.cv12(x11) 
        x13 = self.cv13(x11 + x12) 
        x14 = self.cv14(x11 + x13) 
        x15 = self.cv15(x12) 
        x16 = self.cv16(x13) 
        x17 = self.cv17(x14) 
        cbam11 = self.cbam11(x15) 
        cbam12 = self.cbam12(x16) 
        cbam13 = self.cbam13(x17) 

        x21 = self.cv21(x_vis) 
        x22 = self.cv22(x21) 
        x23 = self.cv23(x21 + x22) 
        x24 = self.cv24(x21 + x23) 
        x25 = self.cv25(x22) 
        x26 = self.cv26(x23) 
        x27 = self.cv27(x24) 
        cbam21 = self.cbam21(x25) 
        cbam22 = self.cbam22(x26) 
        cbam23 = self.cbam23(x27)

        cv1 = self.cv1(cbam13 + cbam23) 
        cv2 = self.cv2(cbam12 + cbam22) 
        cv3 = self.cv3(cbam11 + cbam21) 

        return self.cv4(torch.cat([cv1, cv2, cv3], 1)) 
    

    
class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)



class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))

# --------------------------------------------- FeatFuse -------------------------- 


class FeatFuse(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__() 
        self.growth_rate = c2 // 4 
        self.Cinit_dense = c2 // 4
        self.cv = Conv(c1, self.Cinit_dense, 1, 1)
        self.dense_block = DenseBlock(3, self.Cinit_dense, self.growth_rate, BottleneckBlock) 
        self.cbam = CBAM(c2)
        self.mbatt = MBatt(c1 // 2, c2, 3) 
        self.cv_mbatt = Conv(c1, c1 // 2, 1, 1) 
        self.cv_out = Conv(2 * c2, c2, 1, 1)

    def forward(self, x):
        branch_1 = self.cv(x) 
        branch_1 = self.dense_block(branch_1)
        branch_1 = self.cbam(branch_1)

        branch_2 = self.cv_mbatt(x) 
        branch_2 = self.mbatt(branch_2) 

        return self.cv_out(torch.cat([branch_1, branch_2], 1)) 
    

