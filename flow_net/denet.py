import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from .layers.container import SequentialFlow
from .layers import *
from .layers import base
import torch
import torch.nn as nn
from torch.distributions import Normal

class flow_unit(SequentialFlow):
    def __init__(self,
                 dim
                   ):
        chain = []
        def _actnorm(channel):
            return ActNorm(channel)
        def _lipschitz_layer():
            return base.get_linear
        def _iMonotoneBlock(preact=False):
            return iMonotoneBlock(
                FlowNet(
                    channels=dim,
                    lipschitz_layer=_lipschitz_layer()
                )
            )
        chain.append(_iMonotoneBlock())
        chain.append(_actnorm(dim))
        chain.append(_iMonotoneBlock())
        super(flow_unit, self).__init__(chain)


class FlowNet(nn.Module):
    def __init__(
        self,
        channels: int,
        lipschitz_layer
    ): 
      super(FlowNet, self).__init__()
      out_channels = channels
      nnet = [] 
      nnet.append(lipschitz_layer(channels, out_channels,domain=2, codomain=2))
      nnet.append(base.LeakyLSwish())
      self.nnet=nn.Sequential(*nnet)
    
    def forward(self, x):
        y = self.nnet(x)
        return y
    
    def build_clone(self):
        class FCNetClone(nn.Module):
            def __init__(self, nnet):
                super(FCNetClone, self).__init__()
                self.nnet = nnet

            def forward(self, x):
                y = self.nnet(x)
                return y

        return FCNetClone(self.nnet)

    def build_jvp_net(self, x):
        class FCNetJVP(nn.Module):
            def __init__(self, nnet):
                super(FCNetJVP, self).__init__()
                self.nnet = nnet

            def forward(self, v):
                jv = self.nnet(v)
                return jv

        nnet, y = self.nnet.build_jvp_net(x)
        return FCNetJVP(nnet), y



class denoise_net(nn.Module):
    def __init__(self,
        channels: int,
        num_modules: int,
        ):
        super(denoise_net, self).__init__()
        self.channels = channels
        self.num_modules = num_modules
        self.channel_mask_dim=channels
        flow_net = []
        for i in range(num_modules):
                flow=flow_unit(
                    dim=int(channels)
                )
                flow_net.append(flow)
        self.flow_net=nn.ModuleList(flow_net)
        self.prior= LearnableNormal(channels)

    def f(self, x: Tensor):
        for i in range(self.num_modules):
            x = self.flow_net[i].forward(x)
        return x
    
    def g(self, z: Tensor):
        for i in reversed(range(self.num_modules)):
            z = self.flow_net[i].inverse(z)
        return z
    
    def prob_predictor(self, z: Tensor, log_p_prior: Tensor):
        for i in reversed(range(self.num_modules)):
            _, log_p_post = self.flow_net[i].inverse(z,log_p_prior)
        return log_p_post

    def latent_variable(self, x: Tensor):
        z= self.f(x)
        return z
    
    def reverse(self, z: Tensor):
        x= self.g(z)
        return x
    
    def forward(self, x: Tensor):
        predict_x= self.latent_variable(x)
        log_prob_prior= self.prior(predict_x)
        log_post= self.prob_predictor(predict_x, log_prob_prior)
        return predict_x, log_prob_prior, log_post

class LearnableNormal(nn.Module):
    def __init__(self, dim: int):
        super(LearnableNormal, self).__init__()
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor):
        y=x.detach()
        self.std = torch.exp(self.log_std)
        self.normal = Normal(self.mean, self.std)
        return self.normal.log_prob(y).sum(dim=-1)




