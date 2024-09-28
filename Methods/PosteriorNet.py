import torch
import torchvision.models as models
import torch.nn as nn
import timm
import numpy as np
import torch.nn.functional as F
import torch.distributions as tdist
from pyro.distributions.transforms.radial import Radial

class Flow(nn.Module):
    def __init__(self, dim, flow_length):
        super(Flow, self).__init__()
        self.dim = dim
        self.flow_length = flow_length

        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)

        self.transforms = nn.Sequential(*(Radial(dim) for _ in range(flow_length)))

    def forward(self, z):
        sum_log_jacobians = 0
        for step, transform in enumerate(self.transforms):
            if torch.any(torch.isnan(z)):
                print('found nan!')
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        import pdb
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x

class PosteriorNet(nn.Module):
    def __init__(self, model, num_class):
        super(PosteriorNet, self).__init__()
        self.classes = num_class
        self.hid_dim = 16
        self.len_flow = 16

        self.metrics = ['msp', 'entropy', 'vacuity']

        if model=='ResNet34':
            mynet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model=='ResNet18':
            mynet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        mynet = mynet.module if isinstance(mynet, torch.nn.DataParallel) else mynet
        modules = list(mynet.children())[:-1]
        self.mynet = nn.Sequential(*modules)

        self.fc = nn.Linear(mynet.fc.in_features, self.hid_dim)
        self.norm = nn.BatchNorm1d(self.hid_dim)
        self.density_estimator = nn.ModuleList([Flow(dim=self.hid_dim, flow_length=self.len_flow) for _ in range(self.classes)])

    def forward(self, inputs):
        features = self.mynet(inputs)
        down_features = self.fc(features.view(features.size(0), -1))
        bn_features = self.norm(down_features)
        log_q_zk = torch.zeros(inputs.shape[0], self.classes).to(features.device.type)
        for c in range(self.classes):
            log_q = self.density_estimator[c].log_prob(bn_features)
            log_q_zk[:, c] = log_q

        return features, log_q_zk

    def criterion(self, output, target, epoch_num, num_classes, annealing_step, device=None):

        evidence = torch.exp(output)*torch.sum(target, dim=0, keepdim=True)
        alpha = evidence + 1
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.classes)
        loss_ece = torch.sum(target * (torch.digamma(alpha_0) - torch.digamma(alpha)), dim=1)  # [n,6]
        return torch.mean(loss_ece)
    
    @torch.no_grad()
    def infer_logits(self, features, logits):
        logits = logits + np.log(logits.size(0)/self.classes)
        return logits
    
    @torch.no_grad()
    def infer_probs(self, logits):
        evidence = torch.exp(logits)
        alpha = evidence + 1
        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probs
    
