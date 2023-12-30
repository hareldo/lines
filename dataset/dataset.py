import glob
import os.path

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from config import M
from .input_parsing import WireframeHuangKun
from .transforms import CropAugmentation, ResizeResolution


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch],
    )


class LineDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        filelist = glob.glob(os.path.join(rootdir, split, '*_line.npz'))
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def _get_im_name(self, idx):
        iname = self.filelist[idx][:-10] + ".png"
        return iname

    def __getitem__(self, idx):
        iname = self._get_im_name(idx)
        image_ = io.imread(iname).astype(float)[:, :, :3]

        target = {}

        # step 1 load npz
        lcmap, lcoff, lleng, lpos, angle = WireframeHuangKun.fclip_parsing(self.filelist[idx], M.ang_type)

        # step 2 crop augment
        if self.split == "train":
            if M.crop:

                s = np.random.choice(np.arange(0.9, M.crop_factor, 0.1))
                image_t, lcmap, lcoff, lleng, angle, cropped_lines, cropped_region \
                    = CropAugmentation.random_crop_augmentation(image_, lpos, s)
                image_ = image_t
                lpos = cropped_lines

        # step 3 resize
        if M.resolution < 128:
            image_, lcmap, lcoff, lleng, angle = ResizeResolution.resize(
                lpos=lpos, image=image_, resolu=M.resolution)

        target["lcmap"] = torch.from_numpy(lcmap).float()
        target["lcoff"] = torch.from_numpy(lcoff).float()
        target["lleng"] = torch.from_numpy(lleng).float()
        target["angle"] = torch.from_numpy(angle).float()

        image = (image_ - M.image.mean) / M.image.stddev
        image = np.rollaxis(image, 2).copy()

        return torch.from_numpy(image).float(), target



