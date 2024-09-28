import torch
import torchvision.models as models
import torch.nn as nn
import timm
import torch.nn.functional as F

class R_EvidenceNet(nn.Module):
    def __init__(self, model, num_class):
        super(R_EvidenceNet, self).__init__()
        self.classes = num_class
        self.metrics = ['msp', 'entropy', 'vacuity']

        if model=='ResNet34':
            mynet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model=='ResNet18':
            mynet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        mynet = mynet.module if isinstance(mynet, torch.nn.DataParallel) else mynet
        modules = list(mynet.children())[:-1]
        self.mynet = nn.Sequential(*modules)
        self.fc = nn.Linear(mynet.fc.in_features, num_class)
        self.lamb1 = 1
        self.lamb2 = 0.1

    def forward(self, images):
        features = self.mynet(images)
        out = self.fc(features.view(features.size(0), -1))
        return features, out
    
    def criterion(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        def compute_mse(labels_1hot, evidence):
            num_classes = evidence.shape[-1]

            gap = labels_1hot - (evidence + self.lamb2) / \
                (evidence + self.lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + self.lamb2 * num_classes)

            loss_mse = gap.pow(2).sum(-1)

            return loss_mse.mean()


        
        def kl_divergence(output, num_classes, device=None):
            ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
            sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
            first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
            )
            second_term = (
                (alpha - ones)
                .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
                .sum(dim=1, keepdim=True)
            )
            kl = first_term + second_term
            return kl

        alpha = F.softplus(output) + 1
        alpha_ = (alpha - 1) * (1-target) + 1

        mse_loss = compute_mse(target, alpha-1)
        kl_loss = kl_divergence(alpha_, num_classes, device=device).mean()

        return mse_loss + kl_loss
    
    @torch.no_grad()
    def infer_logits(self, features, logits):
        return logits
    
    @torch.no_grad()
    def infer_probs(self, logits):
        alpha = F.softplus(logits) + 1
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probs
    
    