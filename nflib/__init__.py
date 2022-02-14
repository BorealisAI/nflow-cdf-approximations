import torch
import torch.nn as nn 
import math 
import numpy as np

from .flows import (
    ActNorm, AffineHalfFlow, 
    MAF, Invertible1x1Conv,
    NormalizingFlowModel
)

import itertools 
from pytorch_lightning import LightningModule
from torch.distributions import MultivariateNormal

class Flow(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if self.config.flow_type == "glow":
            flows = [Invertible1x1Conv(dim=self.config.in_dims) for i in range(self.config.num_flows)]
            norms = [ActNorm(dim=self.config.in_dims) for _ in flows]
            couplings = [AffineHalfFlow(dim=self.config.in_dims, parity=i%2, nh=self.config.coupling_width) for i in range(len(flows))]
            flows = list(itertools.chain(*zip(norms, flows, couplings))) 
        elif self.config.flow_type == "realnvp":
            flows = [AffineHalfFlow(dim=self.config.in_dims, parity=i%2, nh=self.config.coupling_width) for i in range(self.config.num_flows)]
            norms = [ActNorm(dim=self.config.in_dims) for _ in flows]
            flows = list(itertools.chain(*zip(norms, flows))) 
        elif self.config.flow_type == "maf":
            flows = [MAF(dim=self.config.in_dims, nh=self.config.coupling_width, parity=i%2) for i in range(self.config.num_flows)]
            norms = [ActNorm(dim=self.config.in_dims) for _ in flows]
            flows = list(itertools.chain(*zip(norms, flows))) 
        class StdNormal(nn.Module):
            def __init__(self, mean, cov):
                super().__init__()
                self.mean = mean 
                self.cov = cov 
            
            def log_prob(self, x):
                return MultivariateNormal(self.mean.to(x),self.cov.to(x)).log_prob(x)
            
            def sample(self, x):
                return MultivariateNormal(self.mean,self.cov).sample(x)

        self.prior = StdNormal(mean=torch.zeros(config.in_dims),cov=torch.eye(config.in_dims))
        self.nf = NormalizingFlowModel(self.prior, flows)
        self.automatic_optimization = False 

    def forward(self, x, reverse = False):
        if not(reverse):
            zs, prior_logprob, log_det = self.nf(x)
            return zs[-1], prior_logprob, log_det
        else:
            zs, log_det = self.nf.backward(x)
            return zs[-1], log_det 
    
    def training_step(self, batch, batch_idx):
        batch_idx = self.trainer.global_step
        z, prior_logprob, log_det = self.forward(batch)
        
        logpx = prior_logprob + log_det 
        loss = -logpx.mean()
        
        optim = self.optimizers()
        
        optim.zero_grad() 
        self.manual_backward(loss)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)

        optim.step()
        
        self.log_dict({
            'logpx': (logpx.mean()),
            'det': log_det.mean(),
            'grad_norm': (grad_norm),
        }, prog_bar=True, logger=False,on_step=True,on_epoch=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        batch_idx = self.trainer.global_step
        z, prior_logprob, log_det = self.forward(batch)
        
        logpx = prior_logprob + log_det 
        
        loss = logpx.mean()
        
        self.log_dict({
            'logpx_v': (logpx.mean()),
            'det_v': log_det.mean(),
        }, prog_bar=True, logger=False,on_step=False,on_epoch=True)
        return loss 
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        print()
    
    def on_validation_epoch_end(self):
        print()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optim 
        


