# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ml_collections

def rnode_config():
    config = ml_collections.ConfigDict()
    
    config.num_steps = 50000
    config.lr = 1e-3
    config.weight_decay = 0
    config.nbits = 8
    config.max_grad_norm = float("inf")
    config.batch_size = 10000
    config.nworkers = 4
    config.in_dims = 2
    
    config.regularization = regularization = ml_collections.ConfigDict()
    regularization.kinetic_energy = 0.0
    regularization.jacobian_norm2 = 0.0
    regularization.total_deriv = 0
    regularization.directional_penalty = 0
    
    config.ode_func = ode_func = ml_collections.ConfigDict()
    ode_func.divergence_fn = "approximate" #"brute_force" #"approximate"
    ode_func.residual = False 
    ode_func.rademacher = False 
    ode_func.div_samples = 1
    
    config.odenet = odenet = ml_collections.ConfigDict()
    odenet.hidden_dims = 64,64
    odenet.conv = False 
    odenet.layer_type = "concat"
    odenet.nonlinearity = "softplus"
    odenet.num_squeeze = 0
    odenet.zero_last_weight = True 
    odenet.strides = None
    odenet.zero_last_weight = True
    
    config.cnf = cnf = ml_collections.ConfigDict()
    cnf.T = 1.0
    cnf.train_T = False
    cnf.solver = "dopri5"
    cnf.solver_options = {}#{"step_size": 0.25}
    cnf.test_solver_options = {}
    cnf.test_solver = "dopri5"
    cnf.atol = 1e-5
    cnf.rtol = 1e-5
    cnf.test_atol = 1e-5
    cnf.test_rtol = 1e-5
    
    return config

def discrete_flow():
    config = ml_collections.ConfigDict()
    
    config.num_flows = 5
    config.num_steps = 50000
    config.lr = 1e-3
    config.weight_decay = 1e-5
    config.nbits = 8
    config.max_grad_norm = float("inf")
    config.batch_size = 10000
    config.nworkers = 4
    config.flow_type = "glow"
    config.in_dims = 2
    config.coupling_width = 32
    
    return config

def eval_config():
    eval = ml_collections.ConfigDict()
    eval.result_file = "results.pkl"
    eval.hull_config = dict(source=["on_sphere"],num_points=20,shape="hull")
    eval.hull_config_loop = dict(radius=[0.5,0.75,1.0])
    eval.num_hulls_per_config = 5
    eval.repeat = False
    return eval