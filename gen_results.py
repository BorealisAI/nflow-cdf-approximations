# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import flows 
from flows import FlowODE
import estimators 
import matplotlib.pyplot as plt
import torch
import numpy as np 
import queries 
import os 
from tqdm import tqdm
import itertools
import pickle 

def gen_summary(config, eval_config, workdir):
    print(eval_config)
    
    flow = flows.name2class[config.flow_type](config)
    flow.load()
    flow = flow.eval()
    flow = flow.cuda()
    flow.device = "cuda"
    
    hull_generator = queries.CHQueryGenerator(config)
    result = {e:{i:[] for i in [100,500,1000,2000,4000]} for e in ["mc","is","bf"]}
    result["correct_estimate"] = []
    result["configs"] = []
    result["hulls"] = []
    
    result_file = f"{workdir}/{eval_config.result_file}"
    if os.path.exists(result_file):
        result = pickle.load(open(result_file,"rb"))
    
    def err(pred,correct):
        return abs(correct-pred)
    
    a = [[{key:val} for val in eval_config.hull_config_loop[key]] for key in eval_config.hull_config_loop]+[[eval_config.hull_config.to_dict()]]
    dictss = list(itertools.product(*a))
    dictss = sum([dictss]*eval_config.num_hulls_per_config,[])
    dictss = dictss[len(result["hulls"]):]
    
    pbar = tqdm(total=len(dictss))
    for dicts in dictss:
        z = {}
        for dict_ in dicts:
            z.update(dict_)
        hull_config = z
        
        tries = 0
        correct_estimate = 0 
        hull = None 

        while hull is None or (correct_estimate<0.01 and tries<5):
            hull = hull_generator.generate(hull_config)
            isestimator = estimators.ImportanceSamplingEstimator(convex_hull=hull, flow=flow)
            correct_estimate = np.mean([isestimator.estimate(100000)[1] for i in range(20)])
            tries += 1
        
        result["configs"].extend([hull_config]*5)
        result["hulls"].append([hull])
        result["correct_estimate"].extend([correct_estimate]*5)

        estimator = estimators.MCEstimator(convex_hull = hull, flow=flow)
        for i in result["mc"]:
            result["mc"][i].extend([(err(estimator.estimate(i)[1],correct_estimate),estimator.time) for _ in range(5)])
        
        estimator = estimators.AdaptiveBoundaryEstimator(convex_hull=hull, flow=flow)
        keys = sorted(list(result["bf"].keys()))
        print(correct_estimate)
        numsamples_probs, ours_volume = estimator.estimate(keys[-1],verbose=True)
        print()
        idx = 0
        for num,prob,time in numsamples_probs:
            if num>=keys[idx]:
                result["bf"][keys[idx]].extend([(err(prob.item(),correct_estimate),time)]*5)
                idx += 1
            if idx == len(keys):
                break
        
        estimator = estimators.ImportanceSamplingEstimator(convex_hull = hull, flow=flow)
        for i in result["is"]:
            result["is"][i].extend([(err(estimator.estimate(i)[1],correct_estimate),estimator.time) for _ in range(5)])
        pbar.update(1)
        import pprint 
        
        with open(result_file,"wb") as f:
            pickle.dump(result,f)
        tabulate(result)
    
    pbar.close()

def tabulate(result):
    from collections import defaultdict
    r = defaultdict(lambda:defaultdict(list))
    flatten = []
    for idx in range(len(result["correct_estimate"])):
        for m in ["mc", "is", "bf"]:
            flatten.append(
                {
                    "R": result["configs"][idx]["radius"],
                    "#Pts": result["configs"][idx]["num_points"],
                    "Target": result["correct_estimate"][idx],
                    "method": m,
                    2000: result[m][2000][idx],
                    4000: result[m][4000][idx],
                }
            )
    
    group_by = ["R", "method"]
    for row in flatten:
        for col in result["mc"].keys():
            if col in row:
                r[tuple(map(row.get,group_by))][str(col)].append([abs(row[col][0]),row[col][1]])
                r[tuple(map(row.get,group_by))][str(col)+"r"].append([abs(row[col][0])/row["Target"],row[col][1]])

    prev = None
    
    for key in sorted(r):
        if prev is None:
            title = "".join(map(lambda x:f"{str(x):^5s}",group_by)) + "".join(map(lambda x:f"{str(x):^15s}",sorted(r[key]))) 
            print(title)
        if key[0]!=prev:
            print("="*len(title))
            prev = key[0]
        print("".join(map(lambda x:f"{str(x):^5s}",key)) 
        + "".join(map(lambda x:f"{np.mean(np.array(r[key][x])[:,0]):>7.4f}±{np.std(np.array(r[key][x])[:,0]):<7.4f}",sorted(r[key]))) 
        )
    print("="*len(title))

