import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import xywh2xyxy
from utils.metrics import bbox_iou
from utils.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist
from utils.tal.assigner import TaskAlignedAssigner
from utils.torch_utils import de_parallel 
from utils.ssim import ssim as ssim_fn


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        
        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


class ComputeLoss:
    # Compute losses
    def __init__(self, model, use_dfl=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.lambda_fusion = 1.0
        self.lambda_fusion_aux = 1.0

        self.lambda_ssim = 3.0
        self.lambda_ssim_aux = 3.0

        self.lambda_l2 = 300.0
        self.lambda_l2_aux = 300.0

        # SSIM config
        self.ssim_default_win = 11      # default target window (will be adapted per scale)
        self.ssim_default_sigma = 1.5   # matches your ssim.py default
        self.ssim_data_range = 1.0      # we normalize to [0,1], so data_range=1.0

        self.assigner = TaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        self.assigner2 = TaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss2 = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def fusion_loss(self, z_vis, z_inf):
        """Calculate fusion loss between visible and infrared features"""
        z_vis = F.softmax(z_vis.view(z_vis.size(0), -1), dim=1)
        z_inf = F.softmax(z_inf.view(z_inf.size(0), -1), dim=1)

        cross_vis_inf = -torch.mean(torch.sum(z_vis * torch.log(z_inf + 1e-10), dim=1))
        cross_inf_vis = -torch.mean(torch.sum(z_inf * torch.log(z_vis + 1e-10), dim=1))

        kl_vis_inf = F.kl_div(torch.log(z_inf + 1e-10), z_vis, reduction='batchmean')
        kl_inf_vis = F.kl_div(torch.log(z_vis + 1e-10), z_inf, reduction='batchmean')

        return cross_vis_inf + cross_inf_vis - kl_vis_inf - kl_inf_vis
    
    def _ssim_loss_pair(self, z_vis: torch.Tensor, z_inf: torch.Tensor) -> torch.Tensor:
        """
        Compute 1 - SSIM between two feature maps (N,C,H,W).
        We normalize to [0,1] with sigmoid and adapt window size to spatial dims.
        Returns a scalar tensor (on the correct device), differentiable.
        """
        # normalize features to [0,1] for stable SSIM (data_range=1.0)
        z_vis = torch.sigmoid(z_vis)
        z_inf = torch.sigmoid(z_inf)

        # pick an odd window size that fits this scale (avoid warnings for small H/W)
        h, w = z_vis.shape[-2], z_vis.shape[-1]
        win = min(self.ssim_default_win, h, w)
        if win % 2 == 0:
            win = max(3, win - 1)  # keep it odd, >=3

        ssim_val = ssim_fn(
            z_vis, z_inf,
            data_range=self.ssim_data_range,
            size_average=True,
            win_size=win,
            win_sigma=self.ssim_default_sigma
        )

        # make sure it's a scalar
        if ssim_val.dim() != 0:
            ssim_val = ssim_val.mean()

        return 1.0 - ssim_val  # turn similarity into a loss 
    
    def fusion_ssim(self, cross_attn_outputs) -> torch.Tensor:
        # return a 0-dim scalar if empty
        if not cross_attn_outputs:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)  # 0-dim scalar
        for cross_tensor in cross_attn_outputs:
            c = cross_tensor.size(1) // 2
            z_inf = cross_tensor[:, :c, :, :]
            z_vis = cross_tensor[:,  c:, :, :]
            total = total + self._ssim_loss_pair(z_vis, z_inf)  # stays scalar
        return total  # shape: []

    def fusion_ssim_aux(self, cross_attn_aux_outputs) -> torch.Tensor:
        if not cross_attn_aux_outputs:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)  # 0-dim scalar
        for cross_tensor in cross_attn_aux_outputs:
            c = cross_tensor.size(1) // 2
            z_inf = cross_tensor[:, :c, :, :]
            z_vis = cross_tensor[:,  c:, :, :]
            total = total + self._ssim_loss_pair(z_vis, z_inf)
        return total  # shape: []
    
    def _l2_loss_pair(self, z_vis: torch.Tensor, z_inf: torch.Tensor) -> torch.Tensor:
        """
        L2/MSE between visible and infrared feature maps (N,C,H,W).
        We normalize to [0,1] with sigmoid for stability across layers.
        Returns a scalar tensor (0-dim).
        """
        z_vis = torch.sigmoid(z_vis)
        z_inf = torch.sigmoid(z_inf)
        return F.mse_loss(z_vis, z_inf, reduction="mean")  # scalar
    
    def fusion_l2(self, cross_attn_outputs) -> torch.Tensor:
        if not cross_attn_outputs:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for cross_tensor in cross_attn_outputs:
            c = cross_tensor.size(1) // 2
            z_inf = cross_tensor[:, :c, :, :]
            z_vis = cross_tensor[:,  c:, :, :]
            total = total + self._l2_loss_pair(z_vis, z_inf)
        return total  # scalar


    def fusion_l2_aux(self, cross_attn_aux_outputs) -> torch.Tensor:
        if not cross_attn_aux_outputs:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for cross_tensor in cross_attn_aux_outputs:
            c = cross_tensor.size(1) // 2
            z_inf = cross_tensor[:, :c, :, :]
            z_vis = cross_tensor[:,  c:, :, :]
            total = total + self._l2_loss_pair(z_vis, z_inf)
        return total  # scalar

    def __call__(self, p, targets, img=None, epoch=0):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, fus
        feats = p[0][1][0] if isinstance(p[0], tuple) else p[0][0]
        feats2 = p[0][1][1] if isinstance(p[0], tuple) else p[0][1]
        cross_attn_outputs = p[1]
        cross_attn_aux_outputs = p[2]
        
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        pred_distri2, pred_scores2 = torch.cat([xi.view(feats2[0].shape[0], self.no, -1) for xi in feats2], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores2 = pred_scores2.permute(0, 2, 1).contiguous()
        pred_distri2 = pred_distri2.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_bboxes2 = self.bbox_decode(anchor_points, pred_distri2)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)
        target_labels2, target_bboxes2, target_scores2, fg_mask2 = self.assigner2(
            pred_scores2.detach().sigmoid(),
            (pred_bboxes2.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        target_bboxes2 /= stride_tensor
        target_scores_sum2 = max(target_scores2.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE
        loss[1] *= 0.25
        loss[1] += self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2 # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)
            loss[0] *= 0.25
            loss[2] *= 0.25
        if fg_mask2.sum():
            loss0_, loss2_, iou2 = self.bbox_loss2(pred_distri2,
                                                   pred_bboxes2,
                                                   anchor_points,
                                                   target_bboxes2,
                                                   target_scores2,
                                                   target_scores_sum2,
                                                   fg_mask2)
            loss[0] += loss0_
            loss[2] += loss2_

        loss_fusion = 0
        for cross_tensor in cross_attn_outputs:
            # Expect channel-wise concat: [infrared | visible]
            c = cross_tensor.size(1) // 2
            z_inf  = cross_tensor[:, :c, :, :]
            z_vis  = cross_tensor[:,  c:, :, :]
            loss_fusion += self.fusion_loss(z_vis, z_inf)

        loss[3] = self.lambda_fusion * loss_fusion 

        # ssim_main = self.fusion_ssim(cross_attn_outputs)
        # loss[3] += self.lambda_ssim * ssim_main

        # l2_main = self.fusion_l2(cross_attn_outputs)
        # loss[3] += self.lambda_l2 * l2_main

        loss_fusion_aux = 0
        for cross_tensor in cross_attn_aux_outputs:
            # Expect channel-wise concat: [infrared | visible]
            c = cross_tensor.size(1) // 2
            z_inf  = cross_tensor[:, :c, :, :]
            z_vis  = cross_tensor[:,  c:, :, :]
            loss_fusion_aux += self.fusion_loss(z_vis, z_inf)

        loss[3] += self.lambda_fusion_aux * loss_fusion_aux * 0.25 

        # ssim_aux = self.fusion_ssim_aux(cross_attn_aux_outputs)
        # loss[3] += self.lambda_ssim_aux * ssim_aux * 0.25

        # l2_aux = self.fusion_l2_aux(cross_attn_aux_outputs)
        # loss[3] += self.lambda_l2_aux * l2_aux * 0.25

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        # loss[3] = 0

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


