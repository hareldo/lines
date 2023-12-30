#!/usr/bin/env python3
"""Train L-CNN
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-lr]
"""

import os
import random
import shutil
import os.path as osp
import datetime

import numpy as np
import torch
from docopt import docopt

from models import build_model
from config import C, M
from dataset.dataset import collate
from dataset.dataset import LineDataset
from lr_schedulers import init_lr_scheduler
from trainer import Trainer


def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.to_yaml(osp.join(outdir, "config.yaml"))
    return outdir


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename="config/base.yaml"))
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)

    # WARNING: L-CNN is still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")

    # 1. dataset
    datadir = C.io.datadir
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        LineDataset(datadir, split="train"), batch_size=M.batch_size, shuffle=True, drop_last=True,
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        LineDataset(datadir, split="valid"), batch_size=M.eval_batch_size, **kwargs
    )

    # 2. model
    model = build_model()

    # 3. optimizer
    optim = torch.optim.Adam(model.parameters(), lr=C.optim.lr, weight_decay=C.optim.weight_decay,
                             amsgrad=C.optim.amsgrad)

    outdir = get_outdir(args["--identifier"])
    print("outdir:", outdir)
    if M.backbone in ["hrnet"]:
        shutil.copy("config/w32_384x288_adam_lr1e-3.yaml", f"{outdir}/w32_384x288_adam_lr1e-3.yaml")

    lr_scheduler = init_lr_scheduler(optim, C.optim.lr_scheduler, stepsize=C.optim.lr_decay_epoch,
                                     max_epoch=C.optim.max_epoch)

    trainer = Trainer(device=device, model=model, optimizer=optim, lr_scheduler=lr_scheduler, train_loader=train_loader,
                      val_loader=val_loader, out=outdir)
    trainer.train()


if __name__ == "__main__":
    main()
