# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict 
import pickle 
import os 
import numpy as np
import glob

def summarize(workdirs,group_by=[],show_fields=[]):
    master = []
    for workdir in workdirs:
        dataset, flow_type, width, depth = workdir.split("_")
        numdims = flow_type[-2]
        flow_type = flow_type[:-2]
        result_file = f"{workdir}/results.pkl"
        if not(os.path.exists(result_file)):
            continue
        results = pickle.load(open(result_file,"rb"))
        num_hulls = len(results["mc"][100])
        for m in ["mc","bf","is"]:
            for i in range(num_hulls):
                hull_config = results["configs"][i]
                target = results["correct_estimate"][i]
                if target<0.01:
                    continue
                if flow_type=="continuous":
                    weight = 3
                else:
                    weight = 1
                for _ in range(weight):
                    master.append({
                        "method": {"mc":"MC","is":"IS","bf":"BF-A"}[m],
                        "radius": hull_config["radius"],
                        "dataset": dataset,
                        "flow_type": flow_type,
                        "numdims": numdims,
                        "width":width,
                        "depth":depth,
                        "100":results[m][100][i][0],
                        "500":results[m][500][i][0],
                        "1000":results[m][1000][i][0],
                        "2000":results[m][2000][i][0],
                        "4000":results[m][4000][i][0],
                        "100r":results[m][100][i][0]/target,
                        "500r":results[m][500][i][0]/target,
                        "1000r":results[m][1000][i][0]/target,
                        "2000r":results[m][2000][i][0]/target,
                        "4000r":results[m][4000][i][0]/target,
                    })
    groups = defaultdict(list)
    for row in master:
        group = tuple(["{:^16s}".format(str(row[key])) for key in group_by])
        groups[group].append(row)
    fmt_string = "{:^16s}    "*(len(group_by)+len(show_fields))
    header = fmt_string.format(*(group_by+show_fields))
    print("="*len(header))
    print(header)
    for key in sorted(groups):
        stats = np.array([[row[c] for c in show_fields] for row in groups[key]])
        means = stats.mean(axis=0)
        stds = stats.std(axis=0)
        linestring = fmt_string.format(*(list(key)+["{:1.5f}±{:1.5f}".format(mean,std) for mean,std in zip(means,stds)]))
        if key[-1].strip().lower()=="is":
            # Assumes that the last key of the group-by is the estimator method!
            print("="*len(header))
        print(linestring)
    print("="*len(header))


summarize(list(glob.glob("workdir/*continuous*"))+list(glob.glob("workdir/*maf*"))+list(glob.glob("workdir/*glow*")),
    group_by=["method"],
    show_fields=["4000","4000r"])

summarize(list(glob.glob("workdir/*continuous*"))+list(glob.glob("workdir/*maf*"))+list(glob.glob("workdir/*glow*")),
    group_by=["numdims","method"],
    show_fields=["4000","4000r"])

summarize(list(glob.glob("workdir/*continuous*"))+list(glob.glob("workdir/*maf*"))+list(glob.glob("workdir/*glow*")),
    group_by=["radius","method"],
    show_fields=["4000","4000r"])

summarize(list(glob.glob("workdir/*continuous*"))+list(glob.glob("workdir/*maf*"))+list(glob.glob("workdir/*glow*")),
    group_by=["flow_type","method"],
    show_fields=["4000","4000r"])

