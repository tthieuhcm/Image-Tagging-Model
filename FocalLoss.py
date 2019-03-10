import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha, (float, int, long)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        # if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, output, target):

        # logpt = F.log_softmax(output)
        # logpt = logpt.gather(1, target)
        # logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())
        # if self.alpha is not None:
        #     if self.alpha.type() != input.data.type():
        #         self.alpha = self.alpha.type_as(input.data)
        #     at = self.alpha.gather(0, target.data.view(-1))
        #     logpt = logpt * Variable(at)

        # loss = -1 * (1 - pt) ** self.gamma * logpt

        pt = target * output + (1 - target) * (1 - output)
        # print(target)
        # print(output)
        loss = -1 * (1 - pt) ** self.gamma * (target * torch.log(output + 1e-12) + (1 - target) * torch.log(1 - output + 1e-12))

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
