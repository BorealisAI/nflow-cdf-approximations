# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from datasets import Dataset
import abc
from scipy.spatial import ConvexHull
from flows import Flow 
import datasets
import flows
from estimators import Cube 

class QueryGenerator(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.dataset = datasets.name2class[config.dataset](config,partition="train")
        flow = flows.name2class[config.flow_type](config)
        flow.load()
        flow = flow.eval()
        flow = flow.cuda()
        flow.device = "cuda"
        self.flow = flow
        self.num_dimensions = config.flow_config.in_dims
        pass 
    
    @abc.abstractmethod
    def generate(self, query_options):
        pass


class CHQueryGenerator(QueryGenerator):
    def __init__(self, config):
        super().__init__(config)
        pass 

    def generate(self, query_options):
        """
        query_options:
            source: in_sphere | on_sphere
            shape: hull | cube
            num_points: int
            radius: float
        """
        source = query_options.get("source",["on_sphere"])
        shape = query_options.get("shape","hull")
        num_points = query_options.get("num_points",self.num_dimensions+1)
        all_samples = None 
        
        if "in_sphere" in source:
            trunc = query_options.get("trunc",1.0)
            center = self.flow.sample(1,trunc=trunc)["x"]

            radius = query_options.get("radius",1.0)
            directions = torch.randn(num_points,center.shape[1])
            directions = directions/directions.norm(dim=1,keepdim=True)
            samples = center + directions * radius * torch.rand([directions.shape[0],1])
            if all_samples is None:
                all_samples = samples 
            else:
                all_samples = torch.cat([all_samples,samples],dim=0)
        if "on_sphere" in source:
            trunc = query_options.get("trunc",1.0)
            center = self.flow.sample(1,trunc=trunc)["x"]
            radius = query_options.get("radius",1.0)
            
            directions = torch.randn(num_points,center.shape[1])
            directions = directions/directions.norm(dim=1,keepdim=True)
            samples = center + directions * radius 
            if all_samples is None:
                all_samples = samples 
            else:
                all_samples = torch.cat([all_samples,samples],dim=0)
        
        if shape=="hull":
            return ConvexHull(all_samples.detach().cpu().numpy())
        else:
            mins = all_samples.min(dim=0)[0]
            maxs = all_samples.max(dim=0)[0]
            return Cube(mins, maxs)
