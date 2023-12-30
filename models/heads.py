from config import M
import torch.nn as nn
import torch


class LineHead(nn.Module):
    def __init__(self, input_channels, m, output_channels):
        super(LineHead, self).__init__()
        ks = M.line_kernel

        self.branch1 = nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=(1, ks), padding=(0, int(ks/2))),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=(1, ks), padding=(0, int(ks/2))),
                    )
        self.branch2 = nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
        self.branch3 = nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=(ks, 1), padding=(int(ks/2), 0)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=(ks, 1), padding=(int(ks/2), 0)),
                    )

        self.merge = nn.Conv2d(int(3 * output_channels), output_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x4 = torch.cat([x1, x2, x3], dim=1)

        return self.merge(x4)


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        heads_size = sum(self._get_head_size(), [])
        heads_net = M.head_net
        for k, (output_channels, net) in enumerate(zip(heads_size, heads_net)):
            if net == "raw":
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
                print(f"{k}-th head, head type {net}, head output {output_channels}")

            elif net == "raw_upsampler":
                heads.append(
                    nn.Sequential(
                        nn.UpsamplingBilinear2d(size=(M.resolution, M.resolution)),
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
                print(f"{k}-th head, head type {net}, head output {output_channels}")
            elif net == "mask":
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, m, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(m),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
                print(f"{k}-th head, head type {net}, head output {output_channels}")

            elif net == "line":
                heads.append(
                    LineHead(input_channels, m, output_channels)
                )
                print(f"{k}-th head, head type {net}, head output {output_channels}")
            else:
                raise NotImplementedError
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(self._get_head_size(), []))

    @staticmethod
    def _get_head_size():

        M_dic = M.to_dict()
        head_size = []
        for h in M_dic['head']['order']:
            head_size.append([M_dic['head'][h]['head_size']])

        return head_size

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)

