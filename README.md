### Introduction:

This repo contains code for a new method for efficient CDF approximations in normalizing flows. 
Tools are included for training a variety of popular normalizing flow models on the UCI tabular datasets, as well as simple toy datasets like the "two moon" 2d dataset.
Trained models can then be evaluated with our method for estimation of CDF in a bounding region. 
Both MC sampling baselines and our own novel approach for CDF approximation are included.

### Example Commands:

In order to train on a 2D slice of Power Dataset which considers the first two dimensions, run:

`python -u main.py --workdir="power_glow2a" --config=configs/discrete_power.py --mode=train --config.flow_config.in_dims=2 --config.flow_config.flow_type=glow --config.dims="(0, 1)"`

In order to generate a results.pkl of the CDF runner:

`python -u main.py --workdir="power_glow2a" --config=configs/discrete_power.py --mode=eval --config.flow_config.in_dims=2 --config.flow_config.flow_type=glow --config.dims="(0, 1)"`

For Continuous, you would instead use the following: 

`python -u main.py --workdir="power_continuous2a" --config=configs/continuous_power.py --mode=train --config.flow_config.in_dims=2 --config.dims="(0, 1)"`

### Training and Evaluation Cycles
Run gen_scripts.py to generate the training and eval scripts. Running the train-scripts will generate the trained normalizing flow checkpoints; subsequently, running the eval-scripts will generate the results.pkl file for each trained normalizing flow model. Finally, the gen_summary.py contains scripts to summarize the results across various evaluation axes. 

### Data:

Follow instructions from https://github.com/gpapamak/maf and place them in data/.

### Dependencies:

- NFLib: Modified and used code shared in https://github.com/karpathy/pytorch-normalizing-flows
- RNODE: Modified and used code shared in https://github.com/cfinlay/ffjord-rnode
- MAF: Modified and used code shared in https://github.com/gpapamak/maf 
- ML-Collections: Apache License; used it for config control -- https://github.com/google/ml_collections
- SortedContainers: Apache License -- http://www.grantjenks.com/docs/sortedcontainers/ 
- Pytorch
- Matplotlib
- Torchdiffeq
- Numpy
- Scipy


