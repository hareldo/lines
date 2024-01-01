import os
import shutil
import os.path as osp
from timeit import default_timer as timer
import cv2
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from tqdm import tqdm

from dataset.input_parsing import WireframeHuangKun
from utils import recursive_to
from config import C, M


class Trainer(object):
    def __init__(self, device, model, optimizer, lr_scheduler, train_loader, val_loader, out, bml=1e1000):
        self.device = device
        self.model = model.to(device)
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = C.model.batch_size
        self.eval_batch_size = C.model.eval_batch_size

        self.validation_interval = C.io.validation_interval

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = bml

        self.loss_labels = None
        self.acc_label = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)
        self.writer = SummaryWriter()
        # self.visual = VisualizeResults()
        # self.printer = ModelPrinter(out)

    def _loss(self, result):
        losses = result["losses"]
        accuracy = result["accuracy"]
        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.acc_label = ["Acc"] + list(accuracy[0].keys())
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels) + len(self.acc_label)])

            # self.printer.loss_head(loss_labels=self.loss_labels+self.acc_label)

        total_loss = 0
        for i in range(self.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss

        for i in range(self.num_stacks):
            for j, name in enumerate(self.acc_label, len(self.loss_labels)):
                if name == "Acc":
                    continue
                if name not in accuracy[i]:
                    assert i != 0
                    continue
                acc = accuracy[i][name].mean()
                self.metrics[i, j] += acc.item()

        return total_loss

    @staticmethod
    def draw_lines(image, lines):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        for (v0, v1) in lines.astype(np.uint16):
            image = cv2.line(image, (v0[1], v0[0]), (v1[1], v1[0]), thickness=3, color=(0, 255, 0))
        return image

    def log_examples(self, images, results, targets, meta):
        _pscale = 512 / C.model.resolution
        example_id = 0
        meta = meta[example_id]

        result = results['heatmaps']
        if "lleng" in result.keys():
            lcmap, lcoff = result["lcmap"][example_id], result["lcoff"][example_id]
            lleng, angle = result["lleng"][example_id], result["angle"][example_id]
            lines, scores = WireframeHuangKun.fclip_torch(lcmap, lcoff, lleng, angle, delta=C.model.delta, nlines=300,
                                                          ang_type=C.model.ang_type, resolution=C.model.resolution)

        lcmap_t, lcoff_t = targets["lcmap"][example_id], targets["lcoff"][example_id]
        lleng_t, angle_t = targets["lleng"][example_id], targets["angle"][example_id]
        lines_t, _ = WireframeHuangKun.fclip_torch(lcmap_t, lcoff_t, lleng_t, angle_t,
                                                   delta=C.model.delta, nlines=meta['num_lines'],
                                                   ang_type=C.model.ang_type,
                                                   resolution=C.model.resolution)

        lines = lines.cpu().numpy() * _pscale
        lines_t = lines_t.cpu().numpy() * _pscale

        image = images[example_id].swapaxes(0, 2).numpy()
        image = image * M.image.stddev + M.image.mean
        image = self.draw_lines(image, lines)
        self.writer.add_image('example', image.transpose((2, 0, 1)), self.epoch)

        image = images[example_id].swapaxes(0, 2).numpy()
        image = image * M.image.stddev + M.image.mean
        image = self.draw_lines(image, lines_t)
        self.writer.add_image('target', image.transpose((2, 0, 1)), self.epoch)

    def validate(self, isviz=True, isnpz=True, isckpt=True):
        self.model.eval()

        if isviz:
            viz = osp.join(self.out, "viz", f"{self.iteration * self.batch_size:09d}")
            osp.exists(viz) or os.makedirs(viz)
        if isnpz:
            npz = osp.join(self.out, "npz", f"{self.iteration * self.batch_size:09d}")
            osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            epoch_tqdm = tqdm(self.val_loader, position=0)
            for batch_idx, (images, targets, meta) in enumerate(epoch_tqdm):
                input_dict = {
                    "image": recursive_to(images, self.device),
                    "target": recursive_to(targets, self.device),
                    "do_evaluation": True,
                }

                result = self.model(input_dict)
                total_loss += self._loss(result)

                H = result["heatmaps"]
                for i in range(images.shape[0]):
                    index = batch_idx * self.eval_batch_size + i
                    if isnpz:
                        npz_dict = {}
                        for k, v in H.items():
                            if v is not None:
                                npz_dict[k] = v[i].cpu().numpy()
                        np.savez(
                            f"{npz}/{index:06}.npz",
                            **npz_dict,
                        )

                    if index >= C.io.visual_num:
                        continue
                    # if isviz:
                    #     self.visual.plot_samples(fn, i, H, target, meta, f"{viz}/{index:06}")
                epoch_tqdm.set_description(
                    f"Val {self.epoch}/{self.max_epoch}, step: {batch_idx + 1}/{len(self.val_loader)}")
                self.log_examples(images, result, targets, meta)

        # self.printer.valid_log(len(self.val_loader), self.epoch, self.iteration, self.batch_size, self.metrics[0])
        self.mean_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("Val/Loss", self.mean_loss, self.epoch)

        if isckpt:
            torch.save(
                {
                    "iteration": self.iteration,
                    "arch": self.model.__class__.__name__,
                    "optim_state_dict": self.optim.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "best_mean_loss": self.best_mean_loss,
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                },
                osp.join(self.out, "checkpoint_lastest.pth.tar"),
            )
            shutil.copy(
                osp.join(self.out, "checkpoint_lastest.pth.tar"),
                osp.join(npz, "checkpoint.pth.tar"),
            )
            if self.mean_loss < self.best_mean_loss:
                self.best_mean_loss = self.mean_loss
                shutil.copy(
                    osp.join(self.out, "checkpoint_lastest.pth.tar"),
                    osp.join(self.out, "checkpoint_best.pth.tar"),
                )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        epoch_tqdm = tqdm(self.train_loader, position=0)
        for batch_idx, (images, targets, _) in enumerate(epoch_tqdm):
            self.optim.zero_grad()
            self.metrics[...] = 0

            input_dict = {
                "image": recursive_to(images, self.device),
                "target": recursive_to(targets, self.device),
                "do_evaluation": False,
            }
            result = self.model(input_dict)

            loss = self._loss(result)
            total_loss += loss.item()
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics[0, :len(self.loss_labels)] = self.avg_metrics[0, :len(self.loss_labels)] * 0.9 + \
                                                              self.metrics[0, :len(self.loss_labels)] * 0.1
                if len(self.loss_labels) < self.avg_metrics.shape[1]:
                    self.avg_metrics[0, len(self.loss_labels):] = self.metrics[0, len(self.loss_labels):]

            epoch_tqdm.set_description(
                f"Train {self.epoch}/{self.max_epoch}- loss: {loss.item()}, step: {batch_idx + 1}/{len(self.train_loader)}")
            # if self.iteration % 4 == 0:
            #     self.printer.train_log(self.epoch, self.iteration, self.batch_size, time, self.avg_metrics)
            #
            #     time = timer()
            # num_images = self.batch_size * self.iteration
            # if num_images % self.validation_interval == 0 or num_images == 60:
            #     # record training loss
            #     if num_images > 0:
            #         self.printer.valid_log(1, self.epoch, self.iteration, self.batch_size, self.avg_metrics[0],
            #                                csv_name="train_loss.csv", isprint=False)
            #         self.validate()
            #         time = timer()
            self.writer.add_scalar("Train/Loss", total_loss / len(epoch_tqdm), self.epoch)
            self.iteration += 1

    def train(self):
        for self.epoch in range(0, self.max_epoch):
            self.train_epoch()
            self.lr_scheduler.step()
            self.validate()
