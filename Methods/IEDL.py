import torch
import torchvision.models as models
import torch.nn as nn
import timm
import torch.nn.functional as F

class I_EvidenceNet(nn.Module):
    def __init__(self, model, num_class):
        super(I_EvidenceNet, self).__init__()
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

    def forward(self, images):
        features = self.mynet(images)
        out = self.fc(features.view(features.size(0), -1))
        return features, out
    
    def criterion(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        def edl_digamma_loss(output, target, device=None):
            evidence = torch.exp(output)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            A = torch.sum(target * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
            loss = torch.mean(A)
            return loss
        
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
        
        def compute_fisher_mse(labels_1hot_, evi_alp_):
            evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
            gamma1_alp = torch.polygamma(1, evi_alp_)
            gamma1_alp0 = torch.polygamma(1, evi_alp0_)

            gap = labels_1hot_ - evi_alp_ / evi_alp0_

            loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1).mean()
            loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(-1).mean()
            loss_det_fisher_ = - (torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean()

            return loss_mse_, loss_var_, loss_det_fisher_

        alpha = F.softplus(output)+1
        alpha_ = (alpha - 1) * (1-target) + 1
        loss_mse, loss_var, loss_det_fisher = compute_fisher_mse(target, alpha)
        loss_kl = kl_divergence(alpha_, num_classes, device=device).mean() 

        return loss_mse + loss_var + loss_det_fisher + loss_kl
    
    @torch.no_grad()
    def infer_logits(self, features, logits):
        return logits
    
    @torch.no_grad()
    def infer_probs(self, logits):
        alpha = F.softplus(logits) + 1
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probs

