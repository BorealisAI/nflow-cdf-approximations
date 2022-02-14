# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ml_collections
from configs import *

def get_config():
    config = ml_collections.ConfigDict()
    
    config.workdir = None 
    config.flow_type = "continuous"
    config.dataset = "twomoons"
    config.flow_config = rnode_config()
    
    config.flow_config.weight_decay = 0
    config.flow_config.odenet.hidden_dims = 64,64
    config.flow_config.cnf.T = 1.0
    return config




