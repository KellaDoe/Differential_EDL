import torch
import torchvision.models as models
import torch.nn as nn
import timm
import torch.nn.functional as F

class PriorNet_Re(nn.Module):
    def __init__(self, model, num_class):
        super(PriorNet_Re, self).__init__()
        self.classes = num_class
        self.metrics = ['msp', 'entropy']

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
        # hyper-params
        eps = 0.01
        a0 = 20
        eps_var = 1e-6

        def _kl_loss(y_precision, y_alpha, precision, alpha):
            
            loss = torch.lgamma(y_precision + eps_var) - torch.sum(torch.lgamma(y_alpha + eps_var), 1) \
                - torch.lgamma(precision + eps_var) + torch.sum(torch.lgamma(alpha + eps_var), 1)

            l2 = torch.sum((y_alpha - alpha) * (torch.digamma(y_alpha + eps_var) - torch.digamma(alpha + eps_var)), 1)

            return loss + l2

        # prediction processing
        alpha = torch.exp(output)+1
        precision = torch.sum(alpha, dim=1)
        
        # target processing
        y_smooth = (target * (1 - eps*num_classes) + eps)
        y_alpha = y_smooth * a0
        y_precision = torch.sum(y_alpha, dim=1)

        kl = _kl_loss(precision, alpha, y_precision, y_alpha)

        return torch.mean(kl)
    
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
    '''