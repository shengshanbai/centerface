import torch
import torch.nn as nn
import math


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ESP=1e-12

    def forward(self, pred, target):
        pos_inds = target.eq(1)
        neg_inds = target.lt(1)
        neg_weights = torch.pow(1 - target, 4)

        loss = 0
        pos_loss = torch.log(pred+self.ESP) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred+self.ESP) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss()

    def tranpose_and_gather_feat(self, feat, ind, mask=None):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind.long())
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def forward(self, outputs, batch):
        heat_map, center_off, wh, landmarks = outputs
        _, heat_map_gt, wh_gt, face_index, face_off_gt, face_mask, face_key_off_gt, face_key_mask = batch
        hm_loss = self.focal_loss(heat_map, heat_map_gt)
        wh_loss=0
        off_loss=0
        lm_loss=0
        face_num = face_mask.sum()
        if face_num>0:
            wh_pre = self.tranpose_and_gather_feat(wh, face_index, face_mask)
            dim = wh_gt.size(2)
            mask = face_mask.unsqueeze(2).expand_as(wh_gt)
            wh_gt = wh_gt[mask].view(-1, dim)
            wh_loss = nn.functional.smooth_l1_loss(wh_pre, wh_gt, reduction='sum')

            center_off = self.tranpose_and_gather_feat(center_off, face_index, face_mask)
            dim = face_off_gt.size(2)
            mask = face_mask.unsqueeze(2).expand_as(face_off_gt)
            face_off_gt = face_off_gt[mask].view(-1, dim)
            off_loss = nn.functional.smooth_l1_loss(center_off, face_off_gt, reduction='sum')

            if face_key_mask.sum() > 0:
                face_key_mask=face_key_mask
                landmark_pre = self.tranpose_and_gather_feat(landmarks, face_index)
                lm_loss = nn.functional.smooth_l1_loss(landmark_pre[face_key_mask], face_key_off_gt[face_key_mask])

        if face_num == 0:
            face_num = 1
        loss = hm_loss + (off_loss + 0.1 * wh_loss + 0.1 * lm_loss) / face_num
        return loss
