import torch
from torch import nn

from ..datasets import transforms as t
from .modules.loss import BarlowTwinsLoss, VICRegLoss
from .res16unet import Res16UNet34C

import pointnet2._ext as p2


class ViewpointBottleneck(nn.Module):

    def __init__(self, arch, criterion, split_transform) -> None:
        super().__init__()
        self.encoder = Res16UNet34C(3, 20, [1 for _ in range(8)], 0.02)
        self.encoder.final = nn.Identity()
        self.criterion = BarlowTwinsLoss()
        self.train_split_transform = t.SplitCompose(
            sync_transform=[],
            random_transform=[
                t.RandomRotation(),
                t.RandomTranslation(),
                t.RandomScaling(),
                t.ToSparseTensor(),
            ],
        )
        self.val_split_trainsform = t.Composes([
            t.ToSparseTensor()
        ])
        self.fc = None

    def forward(self, input):
        if self.training:
            (tfield_a, _, _), (tfield_b, _, _), *_ = self.train_split_transform(*input)
            feats_a = self.encoder(tfield_a).slice(tfield_a)
            feats_b = self.encoder(tfield_b).slice(tfield_b)

            indices = p2.furthest_point_sampling(feats_a.C[:, 1:].unsqueeze(0).contiguous(), 1024).reshape(1024).long()

            ds_feats_a = torch.index_select(feats_a.F, 0, indices)
            ds_feats_b = torch.index_select(feats_b.F, 0, indices)

            return self.criterion(ds_feats_a, ds_feats_b)
        else:
            tfield, _, target, *_ = self.val_split_trainsform(*input)
            feats = self.encoder(tfield).slice(tfield)
            logits = self.fc(feats)

            # TODO evaluate validation