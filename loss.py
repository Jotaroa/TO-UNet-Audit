"""
loss.py
=======
Loss functions cho NN4TopOptUNet:
  - LovaszHingeLoss     : IoU-based, cho binary segmentation (thay BCE)
  - LovaszSoftmaxLoss   : ban multiclass (kem theo, khong bat buoc dung)
  - ToleranceBandLoss   : volume-constraint (phat khi |pred - target| > epsilon)

Combined loss dung trong train.py:
    total = BCE(sigmoid(logits), y)
          + vol_coeff * ToleranceBandLoss(sigmoid(logits).mean(), y.mean())
          + 0.3 * LovaszHingeLoss(logits, y)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """Gradient cua Lovasz extension cho Jaccard index."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszHingeLoss(nn.Module):
    """Binary segmentation (Lovasz hinge). logits/labels: [B,H,W]."""
    def __init__(self, per_image=False, ignore_index=None):
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        if self.per_image:
            loss_total = 0
            for logit, label in zip(logits, labels):
                loss_total += self._lovasz_hinge_flat(
                    *self._flatten_binary(logit.unsqueeze(0), label.unsqueeze(0)))
            return loss_total / logits.size(0)
        return self._lovasz_hinge_flat(*self._flatten_binary(logits, labels))

    def _flatten_binary(self, logits, labels):
        logits = logits.view(-1)
        labels = labels.view(-1)
        if self.ignore_index is not None:
            valid = (labels != self.ignore_index)
            logits = logits[valid]; labels = labels[valid]
        return logits, labels

    def _lovasz_hinge_flat(self, logits, labels):
        if len(labels) == 0:
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels.float()[perm]
        grad = lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)


class LovaszSoftmaxLoss(nn.Module):
    """Multiclass segmentation (Lovasz softmax). Kem theo de tham khao."""
    def __init__(self, classes='present', per_image=False, ignore_index=None):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, probas, labels):
        if self.per_image:
            loss_total = 0
            for prob, label in zip(probas, labels):
                loss_total += self._lovasz_softmax_flat(
                    *self._flatten_probas(prob.unsqueeze(0), label.unsqueeze(0)))
            return loss_total / probas.size(0)
        return self._lovasz_softmax_flat(*self._flatten_probas(probas, labels))

    def _flatten_probas(self, probas, labels):
        if probas.dim() == 3:
            probas = probas.unsqueeze(0)
        if labels.dim() == 2:
            labels = labels.unsqueeze(0)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        if self.ignore_index is not None:
            valid = (labels != self.ignore_index)
            probas = probas[valid]; labels = labels[valid]
        return probas, labels

    def _lovasz_softmax_flat(self, probas, labels):
        if probas.numel() == 0:
            return probas * 0.
        loss = 0
        C = probas.size(1)
        if self.classes == 'present':
            class_to_sum = labels.unique()
            if self.ignore_index is not None:
                class_to_sum = class_to_sum[class_to_sum != self.ignore_index]
        else:
            class_to_sum = range(C)
        for c in class_to_sum:
            fg = (labels == c).float()
            if self.classes == 'present' and fg.sum() == 0:
                continue
            prob_c = probas[:, 0] if C == 1 else probas[:, c]
            errors = (fg - prob_c).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            loss += torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        return loss / float(len(class_to_sum))


class ToleranceBandLoss(nn.Module):
    """Volume-constraint: phat khi |pred - target| vuot epsilon (tolerance band)."""
    def __init__(self, epsilon=1e-3, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.clamp(diff - self.epsilon, min=0.0) ** 2
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss
