from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.autograd import Variable
        
####  label smoothing ####
class LabelSmoothing(nn.Module):
    "Implement label smoothing.  size is the class number"
 
    def __init__(self, size, smoothing=0.0):
 
        super(LabelSmoothing, self).__init__()
 
        #self.criterion = nn.KLDivLoss(size_average=False)
 
        #self.padding_idx = padding_idx
 
        self.confidence = 1.0 - smoothing#if i=y
 
        self.smoothing = smoothing
 
        self.size = size
 
        self.true_dist = None
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, target):
        x = self.softmax(x)
        assert x.size(1) == self.size
        true_dist = x.data.clone()#
        #print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))#otherwise
        #print true_dist
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
 
        self.true_dist = true_dist
        tmp = Variable(true_dist, requires_grad=False)
        return -1.0*((x+1e-10).log() * tmp).sum()/x.size(0)
        #return self.criterion(x.log(), Variable(true_dist, requires_grad=False))
#### BinomialLoss ####   
class BinomialLoss(nn.Module):
    """
    BinomialLoss
    """
    
    def __init__(self, alpha = 2.0, beta = 0.5, C =25):
        super(BinomialLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def forward(self, input, label):
        pairs = torch.eq(label.view(-1,1),label.view(-1,1).transpose(0,1)).float()
        m = 1.0 * pairs + (self.C * (pairs - 1.0))
        W = 1.0 - torch.eye(label.size(0)).float().cuda()#define a matrix, and this can not be propagated by model.cuda()
        W = W * pairs / torch.sum(pairs) + W * (1.0 - pairs) / torch.sum(1.0 - pairs)
        cosine = cosine_sim(input)
        act = self.alpha * (cosine - self.beta) * m
        output = torch.log(torch.exp(-act) + 1.0) * W

        return output.sum()
        
############# Adv network ####        
def Adv_hook(module, grad_in, grad_out):
    return((grad_in[0] * (-1),grad_in[1]))
         
class AdvDivLoss(nn.Module):
    """
    Attention AdvDiverse Loss
    x : is the vector
    """

    def __init__(self, parts=4):
        super(AdvDivLoss, self).__init__()
        self.parts = parts
        self.fc_pre = nn.Sequential(nn.Linear(256, 128, bias=False)) 
        self.fc = nn.Sequential(nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128), nn.BatchNorm1d(128))
        self.fc_pre.register_backward_hook(Adv_hook)
        
    def forward(self, x):
        x = nn.functional.normalize(x)
        x = self.fc_pre(x)
        x = self.fc(x)
        x = nn.functional.normalize(x)
        out = 0
        num = int(x.size(0) / self.parts)
        for i in range(self.parts):
            for j in range(self.parts):
                if i<j:
                    out += ((x[i*num:(i+1)*num,:] - x[j*num:(j+1)*num,:]).norm(dim=1,keepdim=True)).mean()
        return out * 2 / (self.parts*(self.parts - 1))
