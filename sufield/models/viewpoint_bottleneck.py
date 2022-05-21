from torch import nn

class ViewpointBottleneck(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()
        