import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint_test

from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', test_solver='dopri5', solver_options = {}, test_solver_options={}, atol=1e-5, rtol=1e-5, test_atol=1e-5,test_rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = test_solver
        self.test_atol = test_atol
        self.test_rtol = test_rtol
        self.solver_options = solver_options
        self.test_solver_options = test_solver_options
        self.grad_required = False 

    def forward(self, z, logpz=None, reg_states=tuple(), integration_times=None, reverse=False):
        if not len(reg_states)==self.nreg and self.training:
            reg_states = tuple(torch.zeros(z.size(0)).to(z) for i in range(self.nreg))

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)
        
        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz) + reg_states,
                integration_times.to(z),
                atol=self.atol,#[self.atol, self.atol] + [1e20] * len(reg_states) if self.solver in ['dopri5', 'bosh3'] else self.atol,
                rtol=self.rtol,#[self.rtol, self.rtol] + [1e20] * len(reg_states) if self.solver in ['dopri5', 'bosh3'] else self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            if self.grad_required:
                if z.is_leaf:
                    z.requires_grad = True
            
                state_t = odeint_test(
                    self.odefunc.forward_map,
                    (z, _logpz),
                    integration_times.to(z),
                    atol=self.test_atol,
                    rtol=self.test_rtol,
                    method=self.test_solver,
                    options=self.test_solver_options
                )
            else:
                state_t = odeint(
                    self.odefunc,
                    (z, _logpz),
                    integration_times.to(z),
                    atol=self.test_atol,
                    rtol=self.test_rtol,
                    method=self.test_solver,
                    options=self.test_solver_options,
                )
        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        reg_states = state_t[2:]

        return z_t, logpz_t, reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
