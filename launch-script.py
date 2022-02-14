# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
for i in glob.glob("eval_scripts/*continuous*.sh"):
    if True:#input(f"launch {i}?")=="y":
        print(i,os.system(f"sbatch {i}"))

