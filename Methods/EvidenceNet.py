import torch
import torchvision.models as models
import torch.nn as nn
import timm
import torch.nn.functional as F

class EvidenceNet(nn.Module):
    def __init__(self, model, num_class, reg_kl):
        super(EvidenceNet, self).__init__()
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

        self.reg_kl = reg_kl

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
        
        def kl_divergence(alpha, num_classes, device=None):
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

        # edl_digamma_loss + kl
        loss = edl_digamma_loss(output, target, device=device)
        
        alpha = torch.exp(output) + 1
        alpha_ = (alpha - 1) * (1-target) + 1
        kl_reg = kl_divergence(alpha_, num_classes, device=device)

        return loss.mean() + self.reg_kl * kl_reg.mean()
    
    @torch.no_grad()
    def infer_logits(self, features, logits):
        return logits
    
    @torch.no_grad()
    def infer_probs(self, logits):
        evidence = torch.exp(logits)
        alpha = evidence + 1
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probs
    
    
    '''
    @torch.no_grad()
    def infer_metrics(self, features):
        metrics = {}
        for key in self.metrics:
            metrics[key] = getattr(self, f'infer_{key}')(features)
        return metrics

    def infer_probs(self, logits):
        evidence = torch.exp(logits)
        alpha = evidence + 1
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probs
    
    def infer_msp(self, features):
        logits = self.fc(features.view(features.size(0), -1))
        probs = self.infer_probs(logits)
        msp = torch.max(probs, dim=1).values
        return msp
    
    def infer_entropy(self, features):
        logits = self.fc(features.view(features.size(0), -1))
        probs = self.infer_probs(logits)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        return entropy
    
    def infer_vacuity(self, features):
        logits = self.fc(features.view(features.size(0), -1))
        evidence = torch.exp(logits)
        alpha = evidence + 1
        vacuity = self.classes / torch.sum(alpha, dim=1)
        return vacuity
    '''
