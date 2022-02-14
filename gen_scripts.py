# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
train_file = open("train_scripts.sh","w")
eval_file = open("eval_scripts.sh","w")
for dataset,total_dims in zip(["gas","hepmass","miniboone","power"],[8,21,43,6]):
    for flow_type in ["glow","maf"]:
        for numdims in [2,3,4,5]:
            for width in [16,32,64]:
                for depth in [3,5,7]:
                    for letter in map(lambda x:chr(x),range(ord('a'),ord('a')+2)):
                        dims = '"'+str(tuple(random.sample(list(range(total_dims)),numdims)))+'"'
                        train_file.write(f"""python main.py --workdir="{dataset}_{flow_type}{numdims}{letter}_{width}_{depth}" --config=configs/discrete_{dataset}.py --mode=train --config.flow_config.in_dims={numdims} --config.flow_config.flow_type={flow_type} --config.flow_config.coupling_width={width} --config.flow_config.num_flows={depth} --config.dims={dims}\n""")
                        eval_file.write(f"""python main.py --mode eval --workdir="{dataset}_{flow_type}{numdims}{letter}_{width}_{depth}"\n""")
    for flow_type in ["continuous"]:
        for numdims in [2,3,4,5]:
            for width in [16,32,64]:
                for depth in [2]:
                    for letter in map(lambda x:chr(x),range(ord('a'),ord('a')+2)):
                        dims = '"'+str(tuple(random.sample(list(range(total_dims)),numdims)))+'"'
                        net = '"'+str(tuple([width]*depth))+'"'
                        train_file.write(f"""python main.py --workdir="{dataset}_{flow_type}{numdims}{letter}_{width}_{depth}" --config=configs/continuous_{dataset}.py --mode=train --config.flow_config.in_dims={numdims} --config.flow_config.odenet.hidden_dims={net} --config.dims={dims}\n""")
                        eval_file.write(f"""python main.py --mode eval --workdir="{dataset}_{flow_type}{numdims}{letter}_{width}_{depth}"\n""")
train_file.close()
eval_file.close()
