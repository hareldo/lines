import torch
from config import C, M
from .stage_1 import FClip
from .hourglass_pose import hg
from .pose_hrnet import get_pose_net as hr
from .hourglass_line import hgl
from .heads import MultitaskHead


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
