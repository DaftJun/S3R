import torch.nn as nn
import torch
from warnings import filterwarnings
filterwarnings('ignore')


class ContLoss(nn.Module):
    def __init__(self, num_classes, device, feat_dims, margin):
        super(ContLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.margin = margin
        self.feat_dim = feat_dims
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, x, labels):
        """
        :param x: feature matrix with shape (batch_size, feat_dims)
        :param labels: ground truth labels with shape (num_classes)
        """
        assert x.size(0) == labels.size(0), "the number of features is not equal to labels"
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(beta=1, mat1=x, mat2=self.centers.t(), alpha=-2)
        distmat = torch.sqrt(distmat)

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        mis_mask = labels.ne(classes.expand(batch_size, self.num_classes))


        # compute similar class loss
        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            # print(value)
            value = value.clamp(min=1e-12, max=1e+12)
            dist.append(value)
        dist = torch.cat(dist)
        loss_1 = dist.mean()

        # compute dissimilar class loss
        ss = distmat[mis_mask]
        num = self.num_classes * batch_size - batch_size
        loss_2 = self.ranking_loss(ss, torch.zeros(num).to(self.device), torch.ones(num).to(self.device))

        return loss_1, loss_2