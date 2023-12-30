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
import glob
import random
import shutil
import os.path as osp
import datetime

import numpy as np
import torch
from docopt import docopt

from config.config import C, M
from dataset.dataset import collate
from dataset.dataset import LineDataset

from models.stage_1 import FClip
from models import MultitaskHead, hg, hgl, hr
from FClip.lr_schedulers import init_lr_scheduler
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


def build_model():
    if M.backbone == "stacked_hourglass":
        model = hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    elif M.backbone == "hourglass_lines":
        model = hgl(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    elif M.backbone == "hrnet":
        model = hr(
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    else:
        raise NotImplementedError

    model = FClip(model)

    if M.backbone == "hrnet":
        model = torch.nn.DataParallel(model)

    if C.io.model_initialize_file:
        if torch.cuda.is_available():
            checkpoint = torch.load(C.io.model_initialize_file)
        else:
            checkpoint = torch.load(C.io.model_initialize_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        print('=> loading model from {}'.format(C.io.model_initialize_file))

    print("Finished constructing model!")
    return model


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
        "batch_size": M.batch_size,
        "collate_fn": collate,
        "num_workers": C.io.num_workers,
        "pin_memory": True,
    }
    dataname = C.io.dataname
    train_loader = torch.utils.data.DataLoader(
        LineDataset(datadir, split="train", dataset=dataname), batch_size=M.batch_size, shuffle=True, drop_last=True,
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        LineDataset(datadir, split="valid", dataset=dataname), batch_size=M.eval_batch_size, **kwargs
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
