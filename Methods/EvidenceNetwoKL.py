import torch
import torchvision.models as models
import torch.nn as nn
import timm
import torch.nn.functional as F

class EvidenceNetwoKL(nn.Module):
    def __init__(self, model, num_class):
        super(EvidenceNetwoKL, self).__init__()
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
        

        # edl_digamma_loss + kl
        loss = edl_digamma_loss(output, target, device=device)

        return torch.mean(loss)
    
    @torch.no_grad()
    def infer_logits(self, features, logits):
        return logits
    
    @torch.no_grad()
    def infer_probs(self, logits):
        evidence = torch.exp(logits)
        alpha = evidence + 1
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probs
    