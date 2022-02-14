# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import torch 
import math
import numpy as np

import rnode 
import nflib
import os

import time 

name2class = {}

def register_class(name):
    
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        
        name2class[local_name] = cls
        return cls

    return _register

class Flow(abc.ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def forward(self, x, requires_grad):
        pass 
    
    @abc.abstractmethod
    def reverse(self, z):
        pass 
    
    def dotProduct(self, x, normals, f_vector = None):
        if f_vector is None:
            f_vector = torch.ones_like(x[0:1]).to(self.device)/x.shape[1]
        f_vector = f_vector.view(1,-1)
        if x.shape[0]>128:
            out = self.dotProduct(x[128:], normals[128:], f_vector=f_vector)
            x = x[:128]
            normals = normals[:128]
        else:
            out = None
        if x.is_leaf:
            x.requires_grad = True
        
        flow_out = self.forward(x, require_grad=False)
        
        z = flow_out["z"] # gaussian variable
        
        normal = torch.distributions.Normal(loc=0,scale=1.0)
        
        u = normal.cdf(z.view(-1)).view(*z.shape) #uniform variable
        logpz_c = normal.log_prob(z)
        
        v = (torch.exp(flow_out["logpx"].view(-1,1)-logpz_c)*(u*f_vector))
        
        if z.is_leaf:
            z.requires_grad = True
        flow_out_reverse = self.reverse(z, require_grad=True)
        x_ = flow_out_reverse["x"]
        
        normals.requires_grad = True
        vjp = torch.autograd.grad((x_*normals).sum(),z, create_graph=False,retain_graph=False)[0]
        
        dot_product = (vjp*v).sum(dim=1)
        px = torch.exp(flow_out["logpx"]).view(x.shape[0]) 
        
        if out is None:
            return dot_product.detach(), px.detach(), u.detach()
        else:
            return torch.cat([dot_product.detach(),out[0]]), torch.cat([px.detach(),out[1]]), torch.cat([u.detach(),out[2]])


@register_class("continuous")
class FlowODE(Flow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flow = rnode.RNODE(config.flow_config)
        self.normal = torch.distributions.Normal(loc=0,scale=1.0)

    def forward(self, x, require_grad=False):
        ret = None
        total = x.shape[0]
        current = 0
        for flow in self.flow.flows:
            flow.grad_required = require_grad
        with torch.set_grad_enabled(require_grad):
            while current<total:
                num_pts = min(self.config.flow_config.batch_size,total-current)
                
                z, delta_logp, regstates = self.flow(x[current:current+num_pts].to(self.device))

                uz = self.normal.cdf(z)
                logpz = rnode.standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)
                logdet = -delta_logp

                out = {
                    "z": z,
                    "uz": uz,
                    "logpx": logpz+logdet,
                    "logdet": logdet,
                }

                if ret is None:
                    ret = {i:out[i].to("cuda") for i in out}
                else:
                    ret = {i:torch.cat([ret[i],out[i].to("cuda")]) for i in out}
                
                current += num_pts
            
        return ret

    def reverse(self, z, require_grad=False):
        for flow in self.flow.flows:
            flow.grad_required = require_grad
        x, delta_logp, regstates = self.flow(z, reverse=True)
        return {
            "x": x,
            "logdet":delta_logp
        }

    def sample(self, num_points, trunc=1.0):
        for flow in self.flow.flows:
            flow.grad_required = False
        with torch.no_grad():
            total_points = 0
            ret = None
            while total_points<num_points:
                num_sampled_pts = min(self.config.flow_config.batch_size,num_points-total_points)
                z = torch.randn(num_sampled_pts, self.config.flow_config.in_dims).to(self.device)
                if trunc!=1.0:
                    z = torch.fmod(z,3)
                reversed_out = self.reverse(z)
                if ret is None:
                    ret = {i:reversed_out[i].cpu() for i in reversed_out}
                else:
                    ret = {i: torch.cat([ret[i],reversed_out[i].cpu()],dim=0) for i in reversed_out}
                total_points += num_sampled_pts
        return ret

    def load(self):
        self.flow = rnode.RNODE.load_from_checkpoint(os.path.join(self.config.workdir,"last.ckpt"),config=self.config.flow_config)


@register_class("discrete")
class DiscreteFlow(Flow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flow = nflib.Flow(config.flow_config)
        self.normal = torch.distributions.Normal(loc=0,scale=1.0)

    def forward(self, x, require_grad=False):
        ret = None
        total = x.shape[0]
        current = 0
        with torch.set_grad_enabled(require_grad):
            while current<total:
            
                num_pts = min(self.config.flow_config.batch_size,total-current)
                z, logpz, logdet = self.flow(x[current:current+num_pts].to(self.device))
                uz = self.normal.cdf(z)

                out = {
                    "z": z,
                    "uz": uz,
                    "logpx": logpz+logdet,
                    "logdet": logdet,
                }

                if ret is None:
                    ret = {i:out[i].to("cuda") for i in out}
                else:
                    ret = {i:torch.cat([ret[i],out[i].to("cuda")]) for i in out}
                
                current += num_pts
            
        return ret


    def reverse(self, z, require_grad=True):
        x, logdet = self.flow(z, reverse=True)

        return {
            "x": x,
            "logdet": logdet
        }
    
    def sample(self, num_points, trunc=1.0):
        with torch.no_grad():
            total_points = 0
            ret = None
            while total_points<num_points:
                num_sampled_pts = min(self.config.flow_config.batch_size,num_points-total_points)
                z = torch.randn(num_sampled_pts, self.config.flow_config.in_dims).to(self.device)
                if trunc!=1.0:
                    z = torch.fmod(z,2)
                reversed_out = self.reverse(z)
                if ret is None:
                    ret = {i:reversed_out[i].cpu() for i in reversed_out}
                else:
                    ret = {i: torch.cat([ret[i],reversed_out[i].cpu()],dim=0) for i in reversed_out}
                total_points += num_sampled_pts
        return ret

    def load(self):
        self.flow = nflib.Flow.load_from_checkpoint(os.path.join(self.config.workdir,"last.ckpt"),config=self.config.flow_config)
        for module in self.flow.modules():
            if str(type(module))==str(nflib.ActNorm):
                module.data_dep_init_done = True

