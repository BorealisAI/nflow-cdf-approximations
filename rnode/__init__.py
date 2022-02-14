import torch
import torch.nn as nn 
import math 
import numpy as np 

from .layers import cnf, odefunc
from .layers.wrappers import cnf_regularization as reg_lib
from pytorch_lightning import LightningModule

REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
    "total_deriv": reg_lib.total_derivative,
    "directional_penalty": reg_lib.directional_derivative
}

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def count_nfe(model):
    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, odefunc.ODEfunc):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals

class RNODE(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config 
        self.flows = []
        self.regz_fns = []
        self.regz_coeffs = []
        self.regz_names = []
        for key in config.regularization:
            if config.regularization[key]!=0:
                self.regz_names.append(key)
                self.regz_fns.append(REGULARIZATION_FNS[key])
                self.regz_coeffs.append(config.regularization[key])
        
        net = odefunc.ODEnet(
            hidden_dims = config.odenet.hidden_dims,
            input_shape = (config.in_dims,),
            strides = config.odenet.strides,
            conv = config.odenet.conv,
            layer_type=config.odenet.layer_type,
            nonlinearity=config.odenet.nonlinearity,
            num_squeeze=config.odenet.num_squeeze,
            zero_last_weight=config.odenet.zero_last_weight
        )
        func = odefunc.ODEfunc(
            diffeq=net,
            divergence_fn=config.ode_func.divergence_fn,
            residual=config.ode_func.residual,
            rademacher=config.ode_func.rademacher,
            div_samples=config.ode_func.div_samples
        )
        flow = cnf.CNF(
            odefunc=func,
            T=config.cnf.T,
            train_T=config.cnf.train_T,
            regularization_fns=self.regz_fns,
            solver=config.cnf.solver,
            test_solver=config.cnf.test_solver,
            solver_options=dict(config.cnf.solver_options),
            test_solver_options=dict(config.cnf.test_solver_options),
            atol=config.cnf.atol,
            rtol=config.cnf.rtol,
            test_atol=config.cnf.test_atol,
            test_rtol=config.cnf.test_rtol)
        
        self.flows = nn.ModuleList([flow])
        self.automatic_optimization = False
    
    def forward(self, x, reverse = False):
        logp = None
        reg_states = tuple()
        for flow in self.flows:
            x, logp, reg_states = flow(x, logp, reg_states, reverse = reverse)
        return x, logp, reg_states 

    def training_step(self, batch, batch_idx):
        batch_idx = self.trainer.global_step
        z, delta_logp, reg_states = self.forward(batch)

        reg_states = tuple(torch.mean(rs) for rs in reg_states)

        logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
        logpx = logpz - delta_logp
        
        loss = -logpx.mean() + sum(coeff*reg_state  for coeff,reg_state in zip(self.regz_coeffs,reg_states))
        
        optim = self.optimizers()
        
        optim.zero_grad() 
        self.manual_backward(loss)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)

        optim.step()
        
        self.log_dict({
            'itr':batch_idx, 
            'loss': (loss),
            'logpx': (logpx.mean()),
            'fe': (count_nfe(self)),
            'grad_norm': (grad_norm),
            **{key:(reg_state) for reg_state,key in zip(reg_states,self.regz_names)}
        }, prog_bar=True, logger=False, on_step=True,on_epoch=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        batch_idx = self.trainer.global_step
        z, delta_logp, reg_states = self.forward(batch)

        reg_states = tuple(torch.mean(rs) for rs in reg_states)

        logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
        logpx = logpz - delta_logp

        loss = -logpx.mean()+ sum(coeff*reg_state  for coeff,reg_state in zip(self.regz_coeffs,reg_states))
        
        self.log_dict({
            'logpx_v': (logpx.mean()),
            'det_v': - delta_logp.mean(),
            'fe_v': (count_nfe(self)),
        }, prog_bar=True, logger=False,on_step=False,on_epoch=True)
        return loss 
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        print()

    def on_epoch_end(self):
        print()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optim
