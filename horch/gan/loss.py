import torch
import torch.nn as nn
import torch.nn.functional as F


class NSGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_p):
        return F.binary_cross_entropy_with_logits(
            fake_p, torch.ones_like(fake_p))


class DiscriminatorLoss(nn.Module):
    def __init__(self, real_label=1):
        super().__init__()
        self.real_label = real_label

    def forward(self, real_p, fake_p):
        real_loss = F.binary_cross_entropy_with_logits(
            real_p, torch.full_like(real_p, self.real_label))
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_p, torch.zeros_like(fake_p))
        return real_loss + fake_loss


class ACGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_p, fake_cp, labels):
        loss1 = F.binary_cross_entropy_with_logits(
            fake_p, torch.ones_like(fake_p))
        loss2 = F.cross_entropy(fake_cp, labels)
        return loss1 + loss2


class ACDiscriminatorLoss(nn.Module):
    def __init__(self, real_label=1):
        super().__init__()
        self.real_label = real_label

    def forward(self, real_p, fake_p, real_cp, fake_cp, labels):
        real_loss = F.binary_cross_entropy_with_logits(
            real_p, torch.full_like(real_p, self.real_label))
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_p, torch.zeros_like(fake_p))
        real_c_loss = F.cross_entropy(fake_cp, labels)
        fake_c_loss = F.cross_entropy(real_cp, labels)
        return real_loss + fake_loss + real_c_loss + fake_c_loss
