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
        # self.prior= LearnableNormal(channels)
        self.prior = LearnableGaussianMixture(channels, num_centers=10)

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
        # log_prob_prior= self.prior(predict_x)
        log_prior_main, log_prior_mixture = self.prior(predict_x)
        # log_post= self.prob_predictor(predict_x, log_prob_prior)
        log_post= self.prob_predictor(predict_x, log_prior_main)
        # return predict_x, log_prob_prior, log_post
        return predict_x, log_prior_main, log_prior_mixture, log_post

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

class LearnableGaussianMixture(nn.Module):
    def __init__(self, dim: int, num_centers: int = 10):
        super().__init__()
        self.dim = dim
        self.num_centers = num_centers

        # Main center: trained freely via main_log_prob → log_post path
        # self.mu_main = nn.Parameter(torch.randn(dim) * 0.01)
        self.mu_main = nn.Parameter(torch.zeros(dim))  # Initialize main center at zero for stability

        # Offsets for centers 2..M; center 1 is fixed at zero offset by construction
        self.delta_mu = nn.Parameter(torch.randn(num_centers - 1, dim) * 0.01)

        # Intra-class mixture logits
        self.psi = nn.Parameter(torch.zeros(num_centers))

    def log_prob_main(self, z: torch.Tensor) -> torch.Tensor:
        """
        Single Gaussian at mu_main, free gradient to mu_main.
        Replaces LearnableNormal as the prior fed into prob_predictor.
        z: [B, D]
        returns: [B]
        """
        sq_dist = torch.sum((z - self.mu_main) ** 2, dim=-1)
        return -0.5 * sq_dist

    def log_prob_mixture(self, z: torch.Tensor) -> torch.Tensor:
        """
        Full GMM over all intra-class centers.
        mu_main is stop-gradiented — only delta_mu and psi receive gradients.
        z: [B, D]
        returns: [B]
        """
        mu_main_sg = self.mu_main.detach()

        # Center 1: mu_main + 0, Centers 2, ..., M: mu_main + delta_mu_i
        delta_0 = torch.zeros(1, self.dim, device=z.device, dtype=z.dtype)
        all_deltas = torch.cat([delta_0, self.delta_mu], dim=0)        # [M, D]
        all_centers = mu_main_sg.unsqueeze(0) + all_deltas             # [M, D]

        log_c = F.log_softmax(self.psi, dim=0)                         # [M]

        sq_dist = torch.sum(
            (z.unsqueeze(1) - all_centers.unsqueeze(0)) ** 2, dim=-1   # [B, M]
        )

        return torch.logsumexp(-0.5 * sq_dist + log_c, dim=1)          # [B]
    
    def forward(self, x: Tensor):
        log_prob_main = self.log_prob_main(x)
        log_prob_mixture = self.log_prob_mixture(x)
        
        return log_prob_main, log_prob_mixture
