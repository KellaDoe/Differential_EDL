import torch
import torch.distributions as dist

def logit2prob(logit):
    evidence = torch.exp(logit)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    return alpha / S

def get_msp(prob):
    return torch.max(prob, dim=1)[0].detach().cpu().numpy()

def get_entropy(prob):
    return -torch.sum(prob*torch.log(prob+1e-6),dim=1).detach().cpu().numpy()

def get_energy(logits):
    return -torch.log(torch.sum(torch.exp(logits), dim=1)).detach().cpu().numpy()

# def get_dirichlet_entropy(logits):
#     evidence = torch.exp(logits)
#     alpha = evidence + 1
#     dirichlet_entropy = dist.Dirichlet(alpha).entropy()
#     return dirichlet_entropy.detach().cpu().numpy()

def get_vacuity(logits):
    evidence = torch.exp(logits)
    alpha = evidence + 1
    vacuity = alpha.shape[-1]/(torch.sum(alpha,dim=1)+alpha.shape[-1])
    return vacuity.detach().cpu().numpy()

def get_mindist(logits):
    # logits:
    return torch.min(logits, dim=1)[0].detach().cpu().numpy()


def get_metric(metric):
    if metric == 'msp':
        return get_msp
    elif metric == 'entropy':
        return get_entropy
    elif metric == 'energy':
        return get_energy
    elif metric == 'vacuity':
        return get_vacuity
    elif metric == 'mindist':
        return get_mindist
    else:
        raise NotImplementedError