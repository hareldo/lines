import numpy as np
import torch
import torch.nn.functional as F


def non_maximum_suppression(a, delta=0., kernel=3):
    ap = F.max_pool2d(a, kernel, stride=1, padding=int((kernel-1)/2))
    mask = (a == ap).float().clamp(min=0.0)
    mask_n = (~mask.bool()).float() * delta
    return a * mask + a * mask_n


class PointParsing:

    @staticmethod
    def jheatmap_torch(jmap, joff, delta=0.8, K=1000, kernel=3, joff_type="raw", resolution=128):
        h, w = jmap.shape
        lcmap = non_maximum_suppression(jmap[None, ...], delta, kernel).reshape(-1)
        score, index = torch.topk(lcmap, k=int(K))

        if joff is not None:
            lcoff = joff.reshape(2, -1)
            if joff_type == "raw":
                y = (index // w).float() + lcoff[0][index] + 0.5
                x = (index % w).float() + lcoff[1][index] + 0.5
            elif joff_type == "gaussian":
                y = (index // w).float() + lcoff[0][index]
                x = (index % w).float() + lcoff[1][index]
            else:
                raise NotImplementedError
        else:
            y = (index // w).float()
            x = (index % w).float()

        yx = torch.cat([y[..., None], x[..., None]], dim=-1).clamp(0, resolution - 1e-6)

        return yx, score, index


class WireframeHuangKun:
    @staticmethod
    def fclip_merge(xy, xy_idx, length_regress, angle_regress, ang_type="radian", resolution=128):
        """
        :param xy: (K, 2)
        :param xy_idx: (K,)
        :param length_regress: (H, W)
        :param angle_regress:  (H, W)
        :param ang_type
        :param resolution
        :return:
        """
        # resolution = OneStageLineParsing.get_resolution()
        xy_idx = xy_idx.reshape(-1)
        lleng_regress = length_regress.reshape(-1)[xy_idx]  # (K,)
        angle_regress = angle_regress.reshape(-1)[xy_idx]   # (K,)

        lengths = lleng_regress * (resolution / 2)
        if ang_type == "cosine":
            angles = angle_regress * 2 - 1
        elif ang_type == "radian":
            angles = torch.cos(angle_regress * np.pi)
        else:
            raise NotImplementedError
        angles1 = -torch.sqrt(1-angles**2)
        direction = torch.cat([angles1[:, None], angles[:, None]], 1)  # (K, 2)
        v1 = (xy + direction * lengths[:, None]).clamp(0, resolution)
        v2 = (xy - direction * lengths[:, None]).clamp(0, resolution)

        return torch.cat([v1[:, None], v2[:, None]], 1)


    @staticmethod
    def fclip_torch(lcmap, lcoff, lleng, angle, delta=0.8, nlines=1000, ang_type="radian", kernel=3, resolution=128):

        xy, score, index = PointParsing.jheatmap_torch(lcmap, lcoff, delta, nlines, kernel, resolution=resolution)
        lines = WireframeHuangKun.fclip_merge(xy, index, lleng, angle, ang_type, resolution=resolution)

        return lines, score

    @staticmethod
    def fclip_parsing(npz_name, ang_type="radian"):
        with np.load(npz_name) as npz:
            lcmap = npz["lcmap"]
            lcoff = npz["lcoff"]
            lpos = npz["lpos"]
            lleng_ = np.clip(npz["lleng"], 0, 64 - 1e-4)
            angle_ = np.clip(npz["angle"], -1 + 1e-4, 1 - 1e-4)

            lleng = lleng_ / 64
            if ang_type == "cosine":
                angle = (angle_ + 1) * lcmap / 2
            elif ang_type == "radian":
                angle = lcmap * np.arccos(angle_) / np.pi
            else:
                raise NotImplementedError

            return lcmap, lcoff, lleng, lpos, angle
