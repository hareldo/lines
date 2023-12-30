import numpy as np


class WireframeHuangKun:
    @staticmethod
    def fclip_parsing(npz_name, ang_type="radian"):
        with np.load(npz_name) as npz:
            lcmap_ = npz["lcmap"]
            lcoff_ = npz["lcoff"]
            lleng_ = np.clip(npz["lleng"], 0, 64 - 1e-4)
            angle_ = np.clip(npz["angle"], -1 + 1e-4, 1 - 1e-4)

            lcmap = lcmap_
            lcoff = lcoff_
            lleng = lleng_ / 64
            if ang_type == "cosine":
                angle = (angle_ + 1) * lcmap_ / 2
            elif ang_type == "radian":
                angle = lcmap * np.arccos(angle_) / np.pi
            else:
                raise NotImplementedError

            return lcmap, lcoff, lleng, angle
