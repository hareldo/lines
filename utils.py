from torch import Tensor


def recursive_to(input, device):
    if isinstance(input, Tensor):
        return input.to(device)
    if isinstance(input, dict):
        for name in input:
            if isinstance(input[name], Tensor):
                input[name] = input[name].to(device)
        return input
    if isinstance(input, list):
        for i, item in enumerate(input):
            input[i] = recursive_to(item, device)
        return input
    assert False
