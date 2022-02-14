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
    config.flow_type = "discrete"
    config.dataset = "hepmass"
    config.flow_config = discrete_flow()

    config.flow_config.in_dims = 21
    config.dims = tuple(range(config.flow_config.in_dims))

    return config




