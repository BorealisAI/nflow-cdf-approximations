# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import flows 
import datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import os

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import configs 

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
config_flags.DEFINE_config_dict(
  "eval_config", configs.eval_config(), "Eval configuration.", lock_config=True)

flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.mark_flags_as_required(["workdir", "mode"])

def main(_):
    if FLAGS.mode == "train":
        FLAGS.config.workdir = os.path.join("workdir",FLAGS.workdir)
        os.makedirs(FLAGS.config.workdir,exist_ok=True)
        import pickle 
        if os.path.exists(f"{FLAGS.config.workdir}/config.pkl"):
            config = pickle.load(open(f"{FLAGS.config.workdir}/config.pkl","rb"))
            FLAGS.config = config
        else:
            with open(f"{FLAGS.config.workdir}/config.pkl","wb") as f:
                pickle.dump(FLAGS.config,f)
            print(FLAGS.config)
            config = FLAGS.config 

        train_dataset = datasets.name2class[FLAGS.config.dataset](FLAGS.config,partition="train")
        val_dataset = datasets.name2class[FLAGS.config.dataset](FLAGS.config,partition="val")
        test_dataset = datasets.name2class[FLAGS.config.dataset](FLAGS.config,partition="test")

        flow = flows.name2class[FLAGS.config.flow_type](FLAGS.config)
        
        logger = pl.loggers.TensorBoardLogger(save_dir=FLAGS.config.workdir,name="")
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=FLAGS.config.workdir,
            save_last=True
        )
        earlystopping_callback = EarlyStopping("logpx_v",min_delta=0,patience=5,mode="max",check_on_train_epoch_end=False)
        ckpt_file = os.path.join(FLAGS.config.workdir,"last.ckpt")
        trainer = pl.Trainer(
                     gpus=1, 
                     logger=logger,
                     max_steps=FLAGS.config.flow_config.num_steps,
                     callbacks=[checkpoint_callback,
                        earlystopping_callback
                     ],
                     resume_from_checkpoint=ckpt_file if os.path.exists(ckpt_file) else None,
                     val_check_interval=1.0
                     )
        if os.path.exists(ckpt_file):
            flow.load()
        trainer.fit(flow.flow, train_dataloaders= torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=FLAGS.config.flow_config.batch_size, 
            num_workers=FLAGS.config.flow_config.nworkers, pin_memory=False,shuffle=True),
            val_dataloaders=torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=FLAGS.config.flow_config.batch_size, 
            num_workers=FLAGS.config.flow_config.nworkers, pin_memory=False,shuffle=True)
        )
        train_nll = trainer.test(flow.flow, dataloaders=torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=config.flow_config.batch_size,
            num_workers=config.flow_config.nworkers, pin_memory=False,shuffle=True))[0]["logpx_v"]

        val_nll = trainer.test(flow.flow, dataloaders=torch.utils.data.DataLoader(
                dataset=val_dataset, batch_size=config.flow_config.batch_size,
                num_workers=config.flow_config.nworkers, pin_memory=False,shuffle=True))[0]["logpx_v"]

        test_nll = trainer.test(flow.flow, dataloaders=torch.utils.data.DataLoader(
                dataset=test_dataset, batch_size=config.flow_config.batch_size,
                num_workers=config.flow_config.nworkers, pin_memory=False,shuffle=True))[0]["logpx_v"]

        with open(f"workdir/{FLAGS.workdir}/nll.pkl","wb") as f:
            pickle.dump((train_nll,val_nll,test_nll),f)

    elif FLAGS.mode=="eval":
        import gen_results
        import pickle 
        torch.backends.cudnn.benchmark = True
        config = pickle.load(open(f"workdir/{FLAGS.workdir}/config.pkl","rb"))

        gen_results.gen_summary(config,FLAGS.eval_config,f"workdir/{FLAGS.workdir}")
    
    pass 



if __name__ == "__main__":
    app.run(main)




